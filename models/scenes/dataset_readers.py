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


camera_angle_x = 0.6911112070083618

def create_transform_matrix(distance):
    transform_matrix = [
        [-np.sign(distance) ,0.0,0.0,0.0],
        [.0,0.0, np.sign(distance), distance ],
        [0.0, 1.0, 0.0,0.0],
        [0.0,0.0, 0.0,1.0]
    ]
    return transform_matrix

def readImage(path, white_background, eval, distance, num_pts, extension=".png"):
    """
    Creates a 3D scene representation from a single image.
    
    Parameters:
    - path: Directory containing the image files
    - white_background: Whether to use white background (True) or transparent (False)
    - eval: Evaluation mode flag
    - distance: Distance of the camera from the origin (determines scene scale)
    - num_pts: Number of random 3D points to generate for initialization
    - extension: Image file extension to look for
    
    Returns:
    - SceneInfo object containing camera parameters, point cloud, and normalization data
    """
    print("Creating Training Transform")
    # Create camera configuration for training (single view at -distance)
    train_cam_infos = CreateCamerasTransforms(
        path, white_background, [-distance], extension
    )
    print("Creating Test Transform")
    # Use the same camera configuration for testing
    test_cam_infos = CreateCamerasTransforms(
        path, white_background, [-distance], extension
    )

    # Get scene normalization parameters for NeRF++ compatibility
    nerf_normalization = getNerfppNorm(train_cam_infos)
    ply_path = os.path.join(path, "points3d.ply")
    
    # Calculate the visible bounds of the scene based on camera parameters
    camera = train_cam_infos[0]
    # Calculate vertical bound based on FoV and distance
    top = distance * math.tan(camera.FovY * 0.5)
    # Calculate horizontal bound based on aspect ratio
    aspect_ratio = camera.width / camera.height
    right = top * aspect_ratio
    print(f"Generating random point cloud ({num_pts})...")

    # Generate random points within the calculated frustum bounds
    # Points are created in a planar distribution at z=0 (creates a flat point cloud)
    xyz = np.random.uniform(low=[-right, 0, -top], high=[right, 0, top], size=(num_pts, 3))
    # Generate random colors for each point
    shs = np.random.random((num_pts, 3)) / 255.0
    # Create point cloud object with points, colors, and zero normals
    pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

    # Save point cloud to PLY file
    storePly(ply_path, xyz, SH2RGB(shs) * 255)
    # Try to read it back (ensures format compatibility)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    # Package all scene information into a SceneInfo object
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
    print("Creating Training Transforms")
    train_cam_infos = CreateCamerasTransforms(
        path, white_background, [-distance, distance], extension
    )
    print("Creating Test Transforms")
    test_cam_infos = CreateCamerasTransforms(
        path, white_background, [-distance], extension
    )

    nerf_normalization = getNerfppNorm(train_cam_infos)
    ply_path = os.path.join(path, "points3d.ply")

    # Since this data set has no colmap data, we start with random points
    camera = train_cam_infos[0]
    top = distance * math.tan(camera.FovY * 0.5)
    aspect_ratio = camera.width / camera.height
    right = top * aspect_ratio
    print(f"Generating random point cloud ({num_pts})...")
    # We create random points inside the bounds of the synthetic Blender scenes
    xyz = np.random.uniform(low=[-right, 0, -top], high=[right, 0, top], size=(num_pts, 3))
    shs = np.random.random((num_pts, 3)) / 255.0
    pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

    storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

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
    cam_infos = []

    filepaths = glob(f"{path}/original/*{extension}")
    num_frames = len(filepaths)
    filepaths = [(int(os.path.basename(filepath).replace(extension, "")), filepath) for filepath in filepaths]
    filepaths.sort()
    for idx, original in filepaths:
        fovx = camera_angle_x
        cam_name_init = original
        cam_name_mirror = original.replace("original", "mirror")

        for i in range(len(distances)):
            distance = distances[i]
            if i == 0:
                cam_name = cam_name_init
            if i == 1:
                cam_name = cam_name_mirror
                if not os.path.exists(cam_name):
                    # save mirror image
                    im = Image.open(cam_name_init)
                    im_flip = ImageOps.mirror(im)
                    im_flip.save(cam_name_mirror)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(create_transform_matrix(distance))
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = cam_name
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")
            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx
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