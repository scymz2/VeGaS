#
# Copyright (C) 2024, Gmum
# Group of Machine Learning Research. https://gmum.net/
# All rights reserved.
#
# The Gaussian-splatting software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
# For inquiries contact  george.drettakis@inria.fr
#
# The Gaussian-mesh-splatting is software based on Gaussian-splatting, used on research.
# This Games software is free for non-commercial, research and evaluation use
#

from arguments import OptimizationParams
from scene.gaussian_model import GaussianModel
from models.flat_splatting.scene.points_gaussian_model import PointsGaussianModel


optimizationParamTypeCallbacks = {
    "gs": OptimizationParams,
}

gaussianModel = {
    "gs": GaussianModel,
}

gaussianModelRender = {
    "gs": GaussianModel,
    "pgs": PointsGaussianModel
}

# GaussianModel表示完整的3D体积高斯，能在各个方向自由缩放。
# PointsGaussianModel表示扁平化（限制了Y轴方向的缩放）的高斯体，适合基于点/面的渲染，通过限制某一维度的缩放使高斯体变得的扁平。
