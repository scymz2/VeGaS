import os
from scene.dataset_readers import (
    readCamerasFromTransforms, CameraInfo,
    getNerfppNorm, BasicPointCloud, SH2RGB, storePly, fetchPly, SceneInfo
)
import numpy as np
from PIL import Image, ImageOps
from utils.graphics_utils import focal2fov, fov2focal
from pathlib import Path
import math
from glob import glob

"""
数据集读取工具模块

该模块提供用于加载和处理不同类型场景数据的函数，包括：
1. 单幅图像场景加载
2. 镜像图像场景加载
3. 相机变换矩阵创建

这些函数为高斯溅射渲染系统提供场景初始化和相机设置所需的数据。
"""

# 默认相机视野角度（弧度）
camera_angle_x = 0.6911112070083618

def create_transform_matrix(distance):
    """
    创建相机的变换矩阵，将相机位置设置为指定的距离
    该矩阵用于将相机坐标系转换为世界坐标系
    该函数假设相机在z轴上朝向原点，并在y轴上朝上。
    变换矩阵的形式为：
    |  -1  0  0  0 |
    |  0  0 1 distance |
    |  0  1  0  0 |
    |  0  0  0  1 |
    这个矩阵的含义为：反转x轴，原来的y轴替换为z轴，并且向正方向移动distance单位，原来的z轴替换为y轴。
    参数:
        distance: 相机与原点的距离，决定场景的尺度
                 (默认值在train.py中为1.0)
    
    返回:
        4x4的变换矩阵，表示相机位置和方向
    """
    transform_matrix = [
        [-np.sign(distance) ,0.0,0.0,0.0],
        [.0,0.0, np.sign(distance), distance ],
        [0.0, 1.0, 0.0,0.0],
        [0.0,0.0, 0.0,1.0]
    ]
    return transform_matrix

def readImage(path, white_background, eval, distance, num_pts, extension=".png"):
    """
    从单幅图像创建3D场景表示
    
    参数:
        path: 包含图像文件的目录
        white_background: 是否使用白色背景(True)或透明背景(False)
        eval: 评估模式标志
        distance: 相机距离原点的距离(决定场景的尺度)
                 (默认值在train.py中为1.0)
        num_pts: 初始化生成的随机3D点的数量
                (默认值在train.py中为100,000)
        extension: 要查找的图像文件扩展名
    
    返回:
        包含相机参数、点云和归一化数据的SceneInfo对象
    """
    print("Creating Training Transform")
    # 创建训练用的相机配置(单一视角，距离为-distance)
    train_cam_infos = CreateCamerasTransforms(
        path, white_background, [-distance], extension
    )
    print("Creating Test Transform")
    # 使用相同的相机配置进行测试
    test_cam_infos = CreateCamerasTransforms(
        path, white_background, [-distance], extension
    )

    # 获取NeRF++兼容的场景归一化参数，计算相机的平均中心位置和场景的1.1倍半径
    nerf_normalization = getNerfppNorm(train_cam_infos)
    ply_path = os.path.join(path, "points3d.ply")
    
    # 根据相机参数计算场景的可见边界
    camera = train_cam_infos[0]
    # 根据FOV和距离计算垂直边界
    top = distance * math.tan(camera.FovY * 0.5)
    # 根据宽高比计算水平边界
    aspect_ratio = camera.width / camera.height
    right = top * aspect_ratio
    print(f"Generating random point cloud ({num_pts})...")

    # 在计算的视锥体边界内生成随机点
    # 点在z=0平面上创建(形成平面点云)
    xyz = np.random.uniform(low=[-right, 0, -top], high=[right, 0, top], size=(num_pts, 3))
    # 为每个点生成随机颜色
    shs = np.random.random((num_pts, 3)) / 255.0
    # 创建带有点、颜色和零法线的点云对象
    pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

    # 将点云保存为PLY文件
    storePly(ply_path, xyz, SH2RGB(shs) * 255)
    # 尝试读回点云(确保格式兼容性)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    # 将所有场景信息打包到SceneInfo对象中
    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
        maxtime=1.0,
    )
    return scene_info


def readMirrorImages(path, white_background, eval, distance, num_pts, extension=".png"):
    """
    从图像和其镜像创建3D场景表示
    
    参数:
        path: 包含图像文件的目录
        white_background: 是否使用白色背景(True)或透明背景(False)
        eval: 评估模式标志
        distance: 相机距离原点的距离(决定场景的尺度)
                 (默认值在train.py中为1.0)
        num_pts: 初始化生成的随机3D点的数量
                (默认值在train.py中为100,000)
        extension: 要查找的图像文件扩展名
    
    返回:
        包含相机参数、点云和归一化数据的SceneInfo对象
    """
    print("Creating Training Transforms")
    # 创建训练用的相机配置，包括原始视角和镜像视角
    train_cam_infos = CreateCamerasTransforms(
        path, white_background, [-distance, distance], extension
    )
    print("Creating Test Transforms")
    # 仅使用原始视角进行测试
    test_cam_infos = CreateCamerasTransforms(
        path, white_background, [-distance], extension
    )

    # 获取场景归一化参数
    nerf_normalization = getNerfppNorm(train_cam_infos)
    ply_path = os.path.join(path, "points3d.ply")

    # 由于此数据集没有colmap数据，我们从随机点开始
    camera = train_cam_infos[0]
    top = distance * math.tan(camera.FovY * 0.5)
    aspect_ratio = camera.width / camera.height
    right = top * aspect_ratio
    print(f"Generating random point cloud ({num_pts})...")
    # 在合成Blender场景边界内创建随机点
    xyz = np.random.uniform(low=[-right, 0, -top], high=[right, 0, top], size=(num_pts, 3))
    shs = np.random.random((num_pts, 3)) / 255.0
    pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

    # 保存和读取点云
    storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    # 打包场景信息
    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
        maxtime=1.0,
    )

    return scene_info

def CreateCamerasTransforms(path: str, white_background, distances, extension=".png"):
    """
    为给定路径下的图像创建相机信息
    
    参数:
        path: 包含图像文件的目录路径
        white_background: 是否使用白色背景
        distances: 不同相机位置的距离列表
                  (在train.py中默认设置为[-1.0]或[-1.0, 1.0])
        extension: 图像文件扩展名
    
    返回:
        包含相机信息对象的列表
    """
    cam_infos = []

    # 获取所有匹配扩展名的原始图像
    filepaths = glob(f"{path}/original/*{extension}")
    num_frames = len(filepaths)
    # 提取索引并按顺序排序
    filepaths = [(int(os.path.basename(filepath).replace(extension, "")), filepath) for filepath in filepaths]
    filepaths.sort()
    
    # 处理每个图像文件
    for idx, original in filepaths:
        fovx = camera_angle_x
        cam_name_init = original
        cam_name_mirror = original.replace("original", "mirror")

        # 为每个指定的距离创建相机
        for i in range(len(distances)):
            distance = distances[i]
            if i == 0:
                cam_name = cam_name_init
            if i == 1:
                cam_name = cam_name_mirror
                # 如果镜像图像不存在，则创建它
                if not os.path.exists(cam_name):
                    # 保存镜像图像
                    im = Image.open(cam_name_init)
                    im_flip = ImageOps.mirror(im)
                    im_flip.save(cam_name_mirror)

            # NeRF中的'transform_matrix'是相机到世界的变换
            c2w = np.array(create_transform_matrix(distance))
            # 从OpenGL/Blender相机坐标系(Y向上，Z向后)转换到COLMAP坐标系(Y向下，Z向前)
            c2w[:3, 1:3] *= -1

            # 获取世界到相机的变换并设置R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])  # 由于CUDA代码中的'glm'，R被转置存储
            T = w2c[:3, 3]

            # 加载图像
            image_path = cam_name
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            # 处理RGBA图像
            im_data = np.array(image.convert("RGBA"))

            # 设置背景色
            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

            # 处理透明度
            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")
            
            # 计算FOV
            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx
            
            # 创建相机信息对象
            cam_infos.append(
                CameraInfo(
                    uid=i, 
                    R=R, 
                    T=T, 
                    FovY=FovY, 
                    FovX=FovX, 
                    image=image,
                    image_path=image_path, 
                    image_name=image_name, 
                    width=image.size[0],
                    height=image.size[1], 
                    time=idx,
                    mask=None,
                    norm_data = norm_data
                )
            )
    return cam_infos