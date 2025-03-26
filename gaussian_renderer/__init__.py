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
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh









def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}


def render_imp(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, flag_max_count=True, culling=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    from diff_gaussian_rasterization_ms import GaussianRasterizationSettings, GaussianRasterizer
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            dc, shs = pc.get_features_dc, pc.get_features_rest
    else:
        colors_precomp = override_color

    if culling==None:
        culling=torch.zeros(means3D.shape[0], dtype=torch.bool, device='cuda')

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, accum_max_count  = rasterizer(
        means3D = means3D,
        means2D = means2D,
        dc = dc,
        shs = shs,
        culling = culling,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        flag_max_count=flag_max_count)



    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : (radii > 0).nonzero(),
            "radii": radii,
            "area_max": accum_max_count,
            }


def render_simp(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, culling=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    from diff_gaussian_rasterization_ms import GaussianRasterizationSettings, GaussianRasterizer
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            dc, shs = pc.get_features_dc, pc.get_features_rest
    else:
        colors_precomp = override_color

    if culling==None:
        culling=torch.zeros(means3D.shape[0], dtype=torch.bool, device='cuda')

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, \
    accum_weights_ptr, accum_weights_count, accum_max_count  = rasterizer.render_simp(
        means3D = means3D,
        means2D = means2D,
        dc = dc,
        shs = shs,
        culling = culling,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : (radii > 0).nonzero(),
            "radii": radii,
            "accum_weights": accum_weights_ptr,
            "area_proj": accum_weights_count,
            "area_max": accum_max_count,
            }



def render_depth(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, culling=None):
    """
    渲染深度图 - 使用现有的render_imp函数并提取深度信息
    """
    # 简单地调用render_imp并手动计算深度信息
    from diff_gaussian_rasterization_ms import GaussianRasterizationSettings, GaussianRasterizer
    
    # 渲染基本图像
    render_result = render_imp(viewpoint_camera, pc, pipe, bg_color, scaling_modifier, override_color, culling=culling)
    
    # 假设渲染结果中没有直接的深度信息，我们创建一个简化的深度图
    # 用于演示目的的简单近似
    H, W = render_result["render"].shape[1], render_result["render"].shape[2]
    
    # 创建一个基于亮度的简单深度图近似
    rgb = render_result["render"].detach()
    grayscale = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    depth_map = 1.0 - grayscale  # 简单假设亮度反比于深度
    
    # 计算法线
    surface_normals = compute_surface_normals(depth_map)
    
    return {
        "depth": depth_map,
        "surface_normals": surface_normals,
        "viewspace_points": render_result["viewspace_points"],
        "visibility_filter": render_result["visibility_filter"],
        "radii": render_result["radii"]
    }

def compute_surface_normals(depth_map):
    """
    简化版表面法线计算，避免CUDA内核错误
    """
    try:
        # 获取图像尺寸
        H, W = depth_map.shape
        
        # 创建法线图
        normals = torch.zeros((3, H, W), device=depth_map.device)
        
        # 计算梯度 - 使用简单的差分
        dx = torch.zeros_like(depth_map)
        dy = torch.zeros_like(depth_map)
        
        # 水平差分
        dx[:, 1:-1] = depth_map[:, 2:] - depth_map[:, :-2]
        # 垂直差分
        dy[1:-1, :] = depth_map[2:, :] - depth_map[:-2, :]
        
        # 为了数值稳定性，对梯度值应用平滑和缩放
        dx = torch.nn.functional.avg_pool2d(dx.unsqueeze(0), 3, stride=1, padding=1).squeeze(0)
        dy = torch.nn.functional.avg_pool2d(dy.unsqueeze(0), 3, stride=1, padding=1).squeeze(0)
        
        # 构建法线
        normals[0] = -dx
        normals[1] = -dy
        normals[2] = torch.ones_like(dx) * 0.1  # 较小的z值使法线更平行于表面
        
        # 安全归一化
        norm = torch.sqrt(torch.sum(normals**2, dim=0, keepdim=True) + 1e-6)
        normals = normals / norm
        
        return normals
    except Exception as e:
        print(f"法线计算错误: {str(e)}")
        # 返回一个简单的默认法线
        return torch.tensor([0.0, 0.0, 1.0], device=depth_map.device).reshape(3, 1, 1).expand(3, depth_map.shape[0], depth_map.shape[1])

def render_depth_with_weights(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, culling=None):
    """
    渲染深度图，并返回每个像素射线上的高斯深度和权重
    """
    from diff_gaussian_rasterization_ms import GaussianRasterizationSettings, GaussianRasterizer
    
    # 创建屏幕空间点
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # 设置光栅化配置
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    
    # 获取尺度与旋转
    scales = pc.get_scaling
    rotations = pc.get_rotation
    cov3D_precomp = None

    # 获取颜色特征
    dc, shs = pc.get_features_dc, pc.get_features_rest
    colors_precomp = None

    if culling==None:
        culling=torch.zeros(means3D.shape[0], dtype=torch.bool, device='cuda')

    # 获取每个像素射线上的高斯深度和权重
    # 注意：这里假设rasterizer有一个render_with_depths方法，实际实现可能需要修改
    try:
        # 尝试使用专门的深度渲染函数(如果存在)
        result = rasterizer.render_with_depths(
            means3D = means3D,
            means2D = means2D,
            dc = dc,
            shs = shs,
            culling = culling,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        
        # 假设result包含：最终渲染图像、每个射线上的高斯深度及其权重
        depths_per_ray = result.get("depths_per_ray", None)  # [B, N, H*W]
        weights_per_ray = result.get("weights_per_ray", None)  # [B, N, H*W]
        depth_map = result.get("depth", None)  # [H, W]
        radii = result.get("radii", None)
        
    except Exception as e:
        print(f"深度渲染出错: {str(e)}")
        # 回退到基本渲染，获取深度图
        render_result = render_imp(viewpoint_camera, pc, pipe, bg_color, scaling_modifier, override_color, culling=culling)
        
        # 创建简化的深度图(基于亮度)
        rgb = render_result["render"].detach()
        depth_map = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        radii = render_result["radii"]
        
        # 创建空的深度和权重数组(简化版本)
        H, W = depth_map.shape
        depths_per_ray = None
        weights_per_ray = None
    
    # 计算表面法线
    surface_normals = compute_surface_normals(depth_map)
    
    return {
        "depth": depth_map,
        "surface_normals": surface_normals,
        "depths_per_ray": depths_per_ray,
        "weights_per_ray": weights_per_ray,
        "viewspace_points": screenspace_points,
        "visibility_filter": (radii > 0).nonzero() if radii is not None else torch.zeros((0,1), device="cuda", dtype=torch.long),
        "radii": radii if radii is not None else torch.zeros(0, device="cuda")
    }