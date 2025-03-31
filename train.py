#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene
from models import (
    optimizationParamTypeCallbacks,
    gaussianModel
)

from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams
import matplotlib.pyplot as plt
import json
import time
import numpy as np

# 检查是否可以使用TensorBoard进行训练可视化
try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(gs_type, dataset: ModelParams, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint,
             debug_from, save_xyz):
    """
    主训练函数，实现高斯点云模型的训练过程
    
    参数:
        gs_type: 高斯点云类型
        dataset: 包含数据集相关参数的ModelParams对象
        opt: 优化器配置
        pipe: 渲染管线参数
        testing_iterations: 在哪些迭代次数进行测试评估
        saving_iterations: 在哪些迭代次数保存模型
        checkpoint_iterations: 在哪些迭代次数保存检查点
        checkpoint: 如果提供，从此检查点恢复训练
        debug_from: 从哪个迭代开始启用调试
        save_xyz: 是否保存点云坐标信息
    """
    # 记录训练开始时间
    time_start = time.process_time()
    init_time = time.time()
    first_iter = 0
    
    # 准备输出目录和日志记录器
    tb_writer = prepare_output_and_logger(dataset)
    
    # 计算视频帧数并初始化高斯点云模型
    frames = len(os.listdir(f'{dataset.source_path}/original'))
    gaussians = gaussianModel[gs_type](dataset.sh_degree, dataset.poly_degree, frames) # poly_degree: 7 多项式阶数

    # 创建场景并设置高斯点云训练
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    
    # 如果提供了检查点，恢复训练状态
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    # 设置背景颜色
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 创建CUDA事件用于计时
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    
    # 初始化指数移动平均值，用于日志记录
    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0 
    
    # 创建进度条
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    
    # 获取训练用的相机视角
    viewpoint_cameras = scene.getTrainCameras()
    
    # 主训练循环
    for iteration in range(first_iter, opt.iterations + 1):
        # 创建目录用于保存点云坐标，并在指定迭代次数保存
        os.makedirs(f"{scene.model_path}/xyz", exist_ok=True)
        if save_xyz and (iteration % 5000 == 1 or iteration == opt.iterations):
            torch.save(gaussians.get_xyz, f"{scene.model_path}/xyz/{iteration}.pt")

        # 记录迭代开始时间
        iter_start.record()

        # 更新学习率
        gaussians.update_learning_rate(iteration)

        # 每1000次迭代增加球谐函数的级别，直到达到最大级别
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # 选择随机相机批次进行训练
        if not viewpoint_stack:
            # 将相机分成批次，并随机排列
            chunks = int(len(viewpoint_cameras) / opt.batch_size)
            viewpoint_stack = list(np.array_split(np.random.permutation(len(viewpoint_cameras)), chunks))
            viewpoint_stack = [(i, cam) for i, cam in enumerate(viewpoint_stack)]

        # 从视角堆栈中随机选择一批相机
        viewpoint_cams = []
        idx = randint(0, len(viewpoint_stack) - 1)
        idx, idxs = viewpoint_stack.pop(idx)
        for i in idxs:
           viewpoint_cams.append(viewpoint_cameras[i]) 

        # 开始渲染
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # 设置背景（随机或固定）
        bg = torch.rand((3), device="cuda") if opt.random_background else background
        outputs = []
            
        # 对每个相机视角进行渲染
        for viewpoint_cam in viewpoint_cams:
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
            # 获取真实图像作为ground truth
            render_pkg["gt"] = viewpoint_cam.get_image(bg, opt.random_background).cuda()
            outputs.append(render_pkg)

        # 合并批次中所有视角的数据
        data = {}
        for k in outputs[0].keys():
            if k == "viewspace_points":
                data[k] = [output[k] for output in outputs]
            elif k in ["visibility_filter", "radii"]:
                data[k] = [output[k] for output in outputs]
            elif k in ["render", "gt", "mask"]:
                data[k] = torch.stack([output[k] for output in outputs], dim=0)

        # 创建可见性掩码
        data['mask_t'] = torch.stack(data['visibility_filter'], dim=-1).any(dim=1)
        
        # 计算损失
        Ll1 = l1_loss(data["render"], data["gt"])  # L1损失
        # 总损失 = L1损失权重 * L1损失 + SSIM权重 * (1-SSIM)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(data["render"], data["gt"]))
        # 计算PSNR指标
        psnr_ = psnr(data["render"], data["gt"]).mean().double()
        # 反向传播
        loss.backward()

        # 记录迭代结束时间
        iter_end.record()

        with torch.no_grad():
            # 更新进度条显示
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
            total_point = gaussians._xyz.shape[0]
            if iteration % 100 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",
                                          "psnr": f"{psnr_:.{2}f}",
                                          "point":f"{total_point}"})
                progress_bar.update(100)
            if iteration == opt.iterations:
                progress_bar.close()

            # 记录日志并保存模型
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # 点云密度控制：添加和修剪点
            if iteration < 15000:
                # 跟踪图像空间中的最大半径用于剪枝
                radii_batch = torch.stack(data['radii'], dim=-1).max(dim=-1)[0]
                visibility_filter_batch = data["mask_t"]
                gaussians.max_radii2D[visibility_filter_batch] = torch.max(
                    gaussians.max_radii2D[visibility_filter_batch],
                    radii_batch[visibility_filter_batch]
                )
                xyscreen = data['viewspace_points']
                # 添加密度统计信息
                gaussians.add_densification_stats(xyscreen, visibility_filter_batch)
                    
                # 定期执行密度化和剪枝操作
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.01, scene.cameras_extent,
                                                size_threshold)
                
                # 定期重置不透明度
                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
                
            # 优化器步骤
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            # 保存检查点
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
    
    # 保存训练时间信息
    time_elapsed = time.process_time() - time_start
    time_dict = {}
    time_dict["time"] = time_elapsed  # CPU时间
    time_dict["elapsed"] = time.time() - init_time  # 墙钟时间

    with open(scene.model_path + f"/time.json", 'w') as fp:
        json.dump(time_dict, fp, indent=True)

def prepare_output_and_logger(args):
    """
    准备输出目录和日志记录器
    
    参数:
        args: 包含模型路径和其他参数的对象
        
    返回:
        tb_writer: TensorBoard写入器，如果TensorBoard不可用则为None
    """
    # 如果没有指定模型路径，则创建一个唯一的路径
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # 创建输出文件夹
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # 创建TensorBoard写入器
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs):
    """
    记录训练报告和测试结果
    
    参数:
        tb_writer: TensorBoard写入器
        iteration: 当前迭代次数
        Ll1: L1损失
        loss: 总损失
        l1_loss: L1损失函数
        elapsed: 迭代耗时
        testing_iterations: 测试迭代列表
        scene: 场景对象
        renderFunc: 渲染函数
        renderArgs: 渲染参数
    """
    # 记录训练损失到TensorBoard
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # 在指定迭代次数上测试模型并记录结果
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        config = {'name': 'test', 'cameras': scene.getTestCameras()}

        l1_test = 0.0
        psnr_test = 0.0
        psnrs = []
        times = []
        # 对每个测试视角进行渲染和评估
        for idx, viewpoint in enumerate(config['cameras']):
            image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
            # 将前5个视角的渲染结果保存到TensorBoard
            if tb_writer and (idx < 5):
                tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                        image[None], global_step=iteration)
                if iteration == testing_iterations[0]:
                    tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                            gt_image[None], global_step=iteration)

            # 计算测试损失和PSNR
            l1_test += l1_loss(image, gt_image).mean().double()
            psnrs.append(psnr(image, gt_image).mean().double().cpu())
            times.append(viewpoint.time)
            psnr_test += psnrs[-1]
            
        # 计算平均指标
        psnr_test /= len(config['cameras'])
        l1_test /= len(config['cameras'])
        
        # 绘制PSNR随时间的变化图
        plt.plot(times, psnrs, 'o')
        plt.ylabel("PSNR")
        plt.xlabel("Frame")
        if not os.path.isdir(f"{scene.model_path}/plots/"):
            os.makedirs(f"{scene.model_path}/plots/")
        plt.savefig(f"{scene.model_path}/plots/{str(iteration)}.png")
        plt.clf()

        # 输出评估结果
        num_gaussians = scene.gaussians.get_xyz.shape[0]
        poly_degree = scene.gaussians._w1.shape[-1] // 2
        print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} Num Points {} Poly Degree {}".format(iteration, config['name'], l1_test, psnr_test, num_gaussians, poly_degree))
        
        # 记录到TensorBoard
        if tb_writer:
            tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
            tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        # 记录直方图和点数到TensorBoard
        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # 设置命令行参数解析器
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--gs_type', type=str, default="gs")
    parser.add_argument('--camera', type=str, default="mirror")
    parser.add_argument("--distance", type=float, default=1.0)
    parser.add_argument("--num_pts", type=int, default=100_000)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[5_000, 10_000, 15_000, 20_000, 25_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--save_xyz", action='store_true')
    parser.add_argument("--poly_degree", type=int, default=7)
    parser.add_argument("--batch_size", type=int, default=3)

    # 解析模型参数
    lp = ModelParams(parser)
    args, _ = parser.parse_known_args(sys.argv[1:])
    lp.gs_type = args.gs_type
    lp.camera = args.camera
    lp.distance = args.distance
    lp.num_pts = args.num_pts
    lp.poly_degree = args.poly_degree
    
    # 解析优化器和管线参数
    op = optimizationParamTypeCallbacks[args.gs_type](parser)
    pp = PipelineParams(parser)
    args = parser.parse_args(sys.argv[1:])
    op.batch_size = args.batch_size

    # 确保最后一次迭代也被保存
    args.save_iterations.append(args.iterations)

    print("torch cuda: ", torch.cuda.is_available())
    print("Optimizing " + args.model_path)
    # 初始化系统状态(RNG)
    safe_state(args.quiet)

    # 设置异常检测
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # 开始训练
    # extract()方法将命令行参数解析为对象属性，可以得到一个更干净简洁的参数对象
    training(
        args.gs_type,
        lp.extract(args), op.extract(args), pp.extract(args),
        args.test_iterations, args.save_iterations, args.checkpoint_iterations,
        args.start_checkpoint, args.debug_from, args.save_xyz
    )

    # 训练完成
    print("\nTraining complete.")
