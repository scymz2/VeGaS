import torch
import numpy as np
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render  # 注意这里使用的是普通的renderer
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from models import gaussianModelRender


def modify_func(means3D: torch.Tensor, # 高斯中心点，形状为 num_gauss x 3，means3D[:,1] = 0
                scales: torch.Tensor, # 高斯尺度，形状为 num_gauss x 3，scales[:,1] = eps
                rotations: torch.Tensor, # 旋转四元数，形状为 num_gauss x 4，表示3D空间中的2D旋转
                time: float):
    # 用于修改高斯分布参数的函数，但当前实现只是返回原始参数
    return means3D, scales, rotations

def render_set( model_path,
                iteration,
                views, 
                gaussians, 
                pipeline, 
                background, 
                interp,  # 插值数量
                extension):
    render_path = os.path.join(model_path, f"render")

    makedirs(render_path, exist_ok=True)
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # 对每个视角进行插值渲染
        for i in range(interp):
            rendering = render(view, gaussians, pipeline, background, interp=interp, interp_idx=i, modify_func=modify_func)["render"].cpu()
            # 保存渲染结果，文件名包含插值索引
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + "_" + str(i) + extension))

def render_sets(dataset : ModelParams,
                iteration : int, 
                pipeline : PipelineParams, 
                skip_train : bool,  # 是否跳过训练集
                skip_test : bool,   # 是否跳过测试集
                interp : int,       # 插值数量
                extension: str):
    with torch.no_grad():
        # 只初始化一个高斯模型，没有前景/背景之分
        gaussians = gaussianModelRender['gs'](dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        # 设置背景颜色
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        # 只渲染测试视角，没有前景/背景组合
        render_set(dataset.model_path, scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, interp, extension)

# 主函数：解析命令行参数并执行渲染
if __name__ == "__main__":
    # 设置命令行参数解析器
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)  # 使用的模型迭代次数
    parser.add_argument('--camera', type=str, default="mirror")  # 相机类型
    parser.add_argument("--distance", type=float, default=1.0)  # 相机距离
    parser.add_argument("--num_pts", type=int, default=100_000)  # 点数量
    parser.add_argument("--skip_train", action="store_false")  # 是否跳过训练集
    parser.add_argument("--skip_test", action="store_true")   # 是否跳过测试集 
    parser.add_argument("--quiet", action="store_true")      # 静默模式
    parser.add_argument("--poly_degree", type=int, default=1)  # 多项式次数
    parser.add_argument("--interp", type=int, default=1)      # 插值数量
    parser.add_argument("--extension", type=str, default=".png")  # 输出文件扩展名

    args = get_combined_args(parser)
    model.gs_type = "gs"
    model.camera = args.camera
    model.distance = args.distance
    model.num_pts = args.num_pts
    model.poly_degree = args.poly_degree

    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.interp, args.extension)