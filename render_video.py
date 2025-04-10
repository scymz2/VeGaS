import torch
import numpy as np
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer.bg_fg_renderer import render  # 注意这里使用了bg_fg_renderer
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from models import gaussianModelRender
import copy

def modify_mesh(triangles: torch.Tensor, # 三角网格数据，形状为 num_gaussians x 3 x 3，triangles[:,:,1] = 0
                time: float  # 时间参数
                ):
    # 用于修改网格，但当前实现只是返回原始三角网格，没有实际修改
    return triangles


def render_set_combine(model_path, views, gaussians_fg, gaussians_fg_mesh, gaussians_bg, pipeline, background, extension):
    # 渲染路径和伪网格路径设置
    render_path = os.path.join(model_path, "render")
    obj_path = os.path.join(model_path, "pseudomesh")
    pipeline.obj_path = obj_path

    makedirs(render_path, exist_ok=True)
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # 判断是否存在对应时间点的obj文件，若存在则使用mesh模型渲染，否则使用普通高斯模型
        if os.path.exists(os.path.join(obj_path, f"{view.time}.obj")):
            rendering = render(view, gaussians_fg_mesh, gaussians_bg, pipeline, background, modify_mesh=modify_mesh)["render"].cpu()
        else:
            rendering = render(view, gaussians_fg, gaussians_bg, pipeline, background)["render"].cpu()
        # 保存渲染结果
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + extension))


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, extension: str):
    with torch.no_grad():
        parser = copy.deepcopy(dataset)
        
        # 初始化前景高斯模型和前景网格高斯模型
        gaussians_fg = gaussianModelRender['gs'](dataset.sh_degree)
        gaussians_fg_mesh = gaussianModelRender['pgs'](dataset.sh_degree)  # 使用pgs类型的模型处理网格
        scene_fg = Scene(dataset, gaussians_fg, load_iteration=iteration, shuffle=False)

        # 如果有背景模型，则加载背景高斯模型
        if parser.bg_model != "":
            print("loading background model...")
            dataset.model_path = dataset.bg_model
            gaussians_bg = gaussianModelRender['gs'](dataset.sh_degree)
            scene_bg = Scene(dataset, gaussians_bg, load_iteration=iteration, shuffle=False)
        else:
            print("no background model...")
            gaussians_bg = None
            scene_bg = None

        # 设置背景颜色
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        # 渲染前景和背景组合场景
        render_set_combine(dataset.model_path, scene_fg.getTestCameras(), gaussians_fg, gaussians_fg_mesh, gaussians_bg, pipeline, background, extension)

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
    parser.add_argument('--gs_type', type=str, default="gs")  # 高斯模型类型
    parser.add_argument("--quiet", action="store_true")  # 静默模式
    parser.add_argument("--start_frame", type=int, default=5)  # 起始帧
    parser.add_argument("--bg_model", type=str, default="")  # 背景模型路径
    parser.add_argument("--scale", type=float, default=1)  # 缩放因子
    parser.add_argument("--bg_stop_frame", type=int, default=-1)  # 背景停止帧
    parser.add_argument("--extension", type=str, default=".png")  # 输出文件扩展名
    
    args = get_combined_args(parser)
    model.gs_type = args.gs_type
    model.camera = args.camera
    model.distance = args.distance
    model.num_pts = args.num_pts
    model.bg_model = args.bg_model
    pipeline.start_frame = args.start_frame
    pipeline.scale = args.scale
    pipeline.bg_stop_frame = args.bg_stop_frame

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.extension)