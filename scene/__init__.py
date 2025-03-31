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
import random
import json
from utils.system_utils import searchForMaxIteration
from models.scenes import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from pathlib import Path

class Scene:
    """
    Scene类负责管理3D场景的表示，包括高斯模型和相机配置。
    
    主要功能:
    1. 场景初始化: 根据不同的输入类型(Colmap、Blender、Mirror、单张图像)加载场景数据
    2. 高斯模型管理: 创建或加载高斯点云模型
    3. 相机管理: 处理训练和测试相机，支持不同分辨率缩放
    4. 场景持久化: 保存点云和相机信息到文件系统
    5. 数据加载: 支持从之前的训练迭代中恢复模型状态
    
    该类作为高斯溅射(Gaussian Splatting)渲染系统的核心组件，
    连接输入数据、相机参数和高斯模型，为训练和渲染提供必要的场景上下文。
    """

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], point_or_mesh: str = "point"):
        """
        初始化Scene对象，设置场景参数并加载场景数据
        
        参数:
            args: 模型和训练相关的参数配置
            gaussians: 高斯模型对象，用于表示3D点云
            load_iteration: 加载特定迭代次数的模型，若为-1则加载最新模型
            shuffle: 是否随机打乱训练和测试相机的顺序
            resolution_scales: 支持的分辨率缩放比例列表
            point_or_mesh: 场景表示类型，"point"表示点云，"mesh"表示网格
        """
        # 设置基本属性
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.point_or_mesh = point_or_mesh
        print("Distance: {}".format(args.distance))
        
        # 处理模型加载逻辑
        if load_iteration:
            if load_iteration == -1:
                # 搜索最新的迭代模型
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        # 初始化相机字典
        self.train_cameras = {}
        self.test_cameras = {}

        # 根据不同的相机类型加载场景信息
        if args.camera == "mirror":
            # 镜像相机模式 - 用于特定的视图合成任务
            scene_info = sceneLoadTypeCallbacks["Mirror"](
                args.source_path, args.white_background, args.eval, args.distance, args.num_pts
            )
        elif args.camera == "one":
            # 单图像模式 - 从单一视角生成场景
            scene_info = sceneLoadTypeCallbacks["Image"](
                args.source_path, args.white_background, args.eval, args.distance, args.num_pts
            )
        else:
            # 自动检测场景类型
            if os.path.exists(os.path.join(args.source_path, "sparse")):
                    # Colmap格式 - 基于SfM的场景重建
                    scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
            elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
                    print("Found transforms_train.json file, assuming Blender data set!")
                    # Blender格式 - 用于合成数据集
                    scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, args.num_pts)
            else:
                assert False, "Could not recognize scene type!"

        # 如果不是加载已有模型，则需要保存输入点云和相机信息
        if not self.loaded_iter:
            # 保存点云数据
            if args.gs_type == "gs_multi_mesh":
                # 多网格模式 - 保存多个PLY文件
                for i, ply_path in enumerate(scene_info.ply_path):
                    with open(ply_path, 'rb') as src_file, open(os.path.join(self.model_path, f"input_{i}.ply") , 'wb') as dest_file:
                        dest_file.write(src_file.read())
            else:
                # 单点云/网格模式 - 保存单个PLY文件
                with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                    dest_file.write(src_file.read())
            
            # 保存相机信息到JSON文件
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        # 打乱相机顺序以提高训练稳定性
        if shuffle:
            random.shuffle(scene_info.train_cameras)  # 多分辨率一致的随机打乱
            random.shuffle(scene_info.test_cameras)  # 多分辨率一致的随机打乱

        # 记录场景尺度，用于归一化
        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # 为每个分辨率缩放创建相机列表
        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        # 加载或创建高斯模型
        if self.loaded_iter:
            # 从现有模型中加载高斯点云
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                  "point_cloud",
                                                  "iteration_" + str(self.loaded_iter),
                                                  "point_cloud.ply"))
            self.gaussians.point_cloud = scene_info.point_cloud
        else:
            # 从点云创建新的高斯模型
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        """
        保存当前迭代的点云模型
        
        参数:
            iteration: 当前迭代次数，用于命名保存的文件
        """
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        """
        获取指定分辨率的训练相机列表
        
        参数:
            scale: 分辨率缩放比例，默认为1.0(原始分辨率)
        返回:
            对应分辨率的训练相机列表
        """
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        """
        获取指定分辨率的测试相机列表
        
        参数:
            scale: 分辨率缩放比例，默认为1.0(原始分辨率)
        返回:
            对应分辨率的测试相机列表
        """
        return self.test_cameras[scale]