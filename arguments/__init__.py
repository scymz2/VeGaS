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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 0
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._depths = ""
        self._resolution = -1
        self._white_background = False
        self.train_test_exp = False
        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000                  # 总训练迭代次数
        self.position_lr_init = 0.00016           # 高斯位置参数的初始学习率
        self.position_lr_final = 0.0000016        # 高斯位置参数的最终学习率
        self.position_lr_delay_mult = 0.01        # 控制位置学习率衰减延迟的乘数
        self.position_lr_max_steps = 30_000       # 位置学习率调度的最大步数
        self.feature_lr = 0.0025                  # 高斯特征/颜色属性的学习率
        self.opacity_lr = 0.025                   # 高斯不透明度属性的学习率
        self.scaling_lr = 0.005                   # 高斯缩放(大小)参数的学习率
        self.rotation_lr = 0.001                  # 高斯旋转参数的学习率
        self.exposure_lr_init = 0.01              # 曝光优化的初始学习率
        self.exposure_lr_final = 0.001            # 曝光优化的最终学习率
        self.exposure_lr_delay_steps = 0          # 曝光学习率调度延迟的步数
        self.exposure_lr_delay_mult = 0.0         # 控制曝光学习率衰减延迟的乘数
        self.percent_dense = 0.01                 # 密度控制的百分比阈值
        self.lambda_dssim = 0.2                   # 结构相似性指数测量(SSIM)损失的权重
        self.densification_interval = 100         # 执行高斯点云加密的迭代间隔
        self.opacity_reset_interval = 3000        # 重置不透明度值的迭代间隔
        self.densify_from_iter = 500              # 开始进行加密过程的迭代次数
        self.densify_until_iter = 15_000          # 停止加密过程的迭代次数
        self.densify_grad_threshold = 0.0002      # 决定是否加密的梯度阈值
        self.depth_l1_weight_init = 1.0           # 深度L1损失的初始权重
        self.depth_l1_weight_final = 0.01         # 深度L1损失的最终权重
        self.random_background = False            # 是否在训练中使用随机背景
        self.optimizer_type = "default"           # 训练使用的优化器类型
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
