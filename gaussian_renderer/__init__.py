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

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # 创建一个与模型xyz坐标相同形状的零张量，并设置需要梯度，用于计算屏幕空间的均值
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    # 计算视角相机的视场角度的正切值
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    # 设置光栅化的配置参数
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),  # 设置图像的高度，从视角相机获取
        image_width=int(viewpoint_camera.image_width),    # 设置图像的宽度，从视角相机获取
        tanfovx=tanfovx,                                  # 设置水平视场角的正切值，用于视角变换
        tanfovy=tanfovy,                                  # 设置垂直视场角的正切值，用于视角变换
        bg=bg_color,                                      # 设置背景颜色，这是渲染时使用的背景色
        scale_modifier=scaling_modifier,                  # 设置缩放修正因子，影响整体场景的缩放程度
        viewmatrix=viewpoint_camera.world_view_transform, # 设置视图矩阵，用于从世界坐标转换到视角坐标
        projmatrix=viewpoint_camera.full_proj_transform,  # 设置投影矩阵，用于从视角坐标投影到2D屏幕坐标
        sh_degree=pc.active_sh_degree,                    # 设置活动的球谐函数的阶数，用于控制球谐函数的复杂度
        campos=viewpoint_camera.camera_center,            # 设置相机的位置，用于从相机的视角进行渲染
        prefiltered=False                                 # 设置是否预过滤，通常用于性能优化
        # debug=pipe.debug                                # 设置是否启用调试模式，用于调试渲染过程
    )
    # 创建光栅化器对象，传入上述设置的光栅化参数
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    # 获取高斯模型的中心坐标、不透明度等信息
    means3D = pc.get_xyz            # 获取高斯模型的3D坐标中心
    means2D = screenspace_points    # 用于屏幕空间梯度计算的2D坐标点
    opacity = pc.get_opacity        # 获取高斯模型的不透明度

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    # 根据配置选择预计算3D协方差或者从缩放/旋转计算协方差
    scales = None          # 初始化缩放系数变量
    rotations = None       # 初始化旋转系数变量
    cov3D_precomp = None   # 初始化预计算的3D协方差变量
    
    # 检查管道设置中是否指定使用Python进行协方差计算
    if pipe.compute_cov3D_python:
        # 如果指定，则调用高斯模型的get_covariance方法获取预计算的协方差
        # 传入缩放修正因子以调整协方差的计算
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        # 如果未指定使用预计算协方差，则获取模型的缩放和旋转参数
        # 这些参数将在光栅化器中用来实时计算3D协方差
        scales = pc.get_scaling     # 获取缩放参数
        rotations = pc.get_rotation # 获取旋转参数

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    # 根据是否覆盖颜色和配置选择预计算颜色或从球谐函数转换颜色
    shs = None              # 初始化球谐函数变量，用于存储或计算颜色信息
    colors_precomp = None   # 初始化预计算颜色变量
    # 检查是否有覆盖颜色的输入，即是否有外部直接指定的颜色数组
    if override_color is None:
        # 如果没有指定覆盖颜色，检查是否需要在Python中从球谐函数计算颜色
        if pipe.convert_SHs_python:
            # 转换并重塑高斯模型的特征，以适应球谐函数到RGB颜色的计算
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            # 计算从高斯模型中心到相机中心的方向
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            # 标准化这些方向向量
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            # 评估球谐函数，转换为RGB颜色
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            # 确保所有颜色值都不低于0，并做适当偏移
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            # 如果不需要在Python中计算颜色，则直接使用高斯模型的特征作为球谐函数数据
            shs = pc.get_features
    else:
        # 如果提供了覆盖颜色，则直接使用这些颜色，不进行计算
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # 使用光栅化器渲染可见的高斯模型到图像，并获取它们在屏幕上的半径
    rendered_image, radii, _ = rasterizer(
        means3D = means3D,          # 3D中心坐标，定义了高斯模型在3D空间中的位置
        means2D = means2D,          # 屏幕空间坐标，用于计算模型在屏幕上的投影位置
        shs = shs,                  # 球谐函数数据，用于光照和颜色的计算
        colors_precomp = colors_precomp,  # 预计算的颜色数据，如果提供则用于直接渲染
        opacities = opacity,        # 模型的不透明度，影响渲染的透明和遮挡关系
        scales = scales,            # 模型的缩放参数，影响模型的大小和形状
        rotations = rotations,      # 模型的旋转参数，定义模型的朝向
        cov3D_precomp = cov3D_precomp)  # 预计算的3D协方差，如果提供则用于定义模型的形变

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    # 返回渲染的图像和相关信息
    return {
        "render": rendered_image,                # 返回渲染得到的2D图像
        "viewspace_points": screenspace_points,  # 返回高斯模型在屏幕空间的坐标点
        "visibility_filter" : radii > 0,         # 返回一个布尔数组，表示哪些高斯模型在视锥内且半径大于0，因此是可见的
        "radii": radii                           # 返回每个高斯模型在屏幕上的半径大小
    }
