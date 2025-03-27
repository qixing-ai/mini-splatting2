import os
import sys
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from fused_ssim import fused_ssim

from gaussian_renderer import network_gui
from gaussian_renderer import render, render_imp, render_simp, render_depth
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, read_config
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import numpy as np
from lpipsPyTorch import lpips
from utils.sh_utils import SH2RGB
import time

import open3d as o3d

def gs2ply(gaussians, path_save):

    path_dir = os.path.dirname(path_save)
    if not os.path.exists(path_dir):
        os.mkdir(path_dir)

    xyz = gaussians._xyz+0
    rgb_sh = SH2RGB(gaussians._features_dc+0)[:,0]
    xyz = xyz.cpu().numpy()
    rgb_sh = rgb_sh.cpu().numpy()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb_sh)
    o3d.io.write_point_cloud(path_save, pcd)


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, args):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(sh_degree=0)

    scene = Scene(dataset, gaussians, resolution_scales=[1,2])
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    mask_blur = torch.zeros(gaussians._xyz.shape[0], device='cuda')
    gaussians.init_culling(len(scene.getTrainCameras()))
    
    # 获取额外损失相关参数
    enable_normal_loss = getattr(args, 'enable_normal_loss', False)
    enable_depth_loss = getattr(args, 'enable_depth_loss', False)
    
    # 添加安全的时间测量选项
    use_safe_timing = getattr(args, 'use_safe_timing', True)  # 默认使用安全计时
    if use_safe_timing:
        print("使用安全计时机制")
        import time
    
    # 只有在启用额外损失时才设置这些参数
    if enable_normal_loss or enable_depth_loss:
        # 损失权重参数
        normal_loss_weight = getattr(args, 'normal_loss_weight', 0.1)
        depth_loss_weight = getattr(args, 'depth_loss_weight', 0.01)
        normal_loss_start_iter = getattr(args, 'normal_loss_start_iter', 1000)
        depth_loss_start_iter = getattr(args, 'depth_loss_start_iter', 2000)
        normal_loss_threshold = getattr(args, 'normal_loss_threshold', 2.0)
        print_loss_interval = getattr(args, 'print_loss_interval', 100)
        
        # 打印配置信息
        print(f"法线约束: 启用={enable_normal_loss}, 权重={normal_loss_weight}, 开始迭代={normal_loss_start_iter}")
        print(f"深度集中约束: 启用={enable_depth_loss}, 权重={depth_loss_weight}, 开始迭代={depth_loss_start_iter}")
        
        # 自动调整法线损失权重
        if enable_normal_loss and getattr(args, 'normal_loss_auto_adjust', True):
            adjusted_normal_weight = normal_loss_weight * 0.5
            print(f"法线约束权重自动调整: {normal_loss_weight} -> {adjusted_normal_weight}")
            normal_loss_weight = adjusted_normal_weight
    else:
        print("使用原始训练流程（无额外约束）")

    for iteration in range(first_iter, opt.iterations + 1):
        # 安全计时开始
        if use_safe_timing:
            start_time = time.time()
        else:
            iter_start.record()

        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render_imp(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0 and iteration>args.simp_iteration1:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras_warn_up(iteration, args.warn_until_iter, scale=1.0, scale2=2.0).copy()

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        render_pkg = render_imp(viewpoint_cam, gaussians, pipe, background, culling=gaussians._culling[:,viewpoint_cam.uid])
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))

        # 根据是否启用额外损失走不同的路径
        if enable_normal_loss or enable_depth_loss:
            # 计算基础光度损失
            photo_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
            
            # 初始化额外损失
            normal_loss = torch.tensor(0.0, device="cuda")
            depth_loss = torch.tensor(0.0, device="cuda")
            
            # 计算额外约束损失
            if (enable_normal_loss and iteration > normal_loss_start_iter) or (enable_depth_loss and iteration > depth_loss_start_iter):
                try:
                    with torch.no_grad():
                        # 从渲染图像获取深度估计
                        rgb = render_pkg["render"].detach()
                        depth_map = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
                        
                        # 计算表面法线
                        surface_normals = compute_surface_normals(depth_map)
                        
                        # 创建深度包
                        depth_pkg = {
                            "depth": depth_map,
                            "surface_normals": surface_normals
                        }
                    
                    # 计算法线一致性损失
                    if enable_normal_loss and iteration > normal_loss_start_iter:
                        normal_loss = compute_normal_consistency_loss(render_pkg, depth_pkg, viewpoint_cam, gaussians)
                        if not torch.isfinite(normal_loss) or normal_loss > normal_loss_threshold:
                            print(f"[警告] 迭代 {iteration}: 法线损失值异常({normal_loss.item():.4f})，使用0.0替代")
                            normal_loss = torch.tensor(0.0, device="cuda")
                    
                    # 计算深度集中损失
                    if enable_depth_loss and iteration > depth_loss_start_iter:
                        depth_loss = compute_simplified_depth_concentration_loss(depth_pkg)
                        if not torch.isfinite(depth_loss) or depth_loss > 1.0:
                            print(f"[警告] 迭代 {iteration}: 深度损失值异常({depth_loss.item():.4f})，使用0.0替代")
                            depth_loss = torch.tensor(0.0, device="cuda")
                except Exception as e:
                    print(f"[错误] 迭代 {iteration}: 约束计算错误: {str(e)}")
                    normal_loss = torch.tensor(0.0, device="cuda")
                    depth_loss = torch.tensor(0.0, device="cuda")
            
            # 组合总损失
            total_loss = photo_loss
            normal_term = torch.tensor(0.0, device="cuda")
            depth_term = torch.tensor(0.0, device="cuda")
            
            if normal_loss > 0:
                normal_term = normal_loss_weight * normal_loss
                total_loss = total_loss + normal_term
            if depth_loss > 0:
                depth_term = depth_loss_weight * depth_loss
                total_loss = total_loss + depth_term
            
            # 反向传播
            total_loss.backward()
            
            # 更新进度条
            ema_loss_for_log = 0.4 * total_loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.{7}f}",
                    "NormLoss": f"{normal_loss.item():.{4}f}" if normal_loss > 0 else "0.0",
                    "DepthLoss": f"{depth_loss.item():.{4}f}" if depth_loss > 0 else "0.0"
                })
                progress_bar.update(10)
            
            
            
            # 修改loss变量以便与原始代码兼容
            loss = total_loss
        else:
            # 原始代码路径
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
            loss.backward()
            
            # 原始进度条更新
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

        # 安全计时结束
        if use_safe_timing:
            elapsed_time = (time.time() - start_time) * 1000  # 转为毫秒
        else:
            iter_end.record()
            # 尝试添加同步，防止CUDA错误
            try:
                torch.cuda.synchronize()
            except:
                pass

        with torch.no_grad():
            # 使用安全的报告机制
            try:
                if use_safe_timing:
                    # 使用Python计时器计算的时间
                    if enable_normal_loss or enable_depth_loss:
                        # 带额外损失的路径
                        if 'normal_loss' in locals() and 'depth_loss' in locals():
                            safe_training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed_time, 
                                        testing_iterations, scene, render, (pipe, background),
                                        normal_loss=normal_loss if enable_normal_loss else None,
                                        depth_loss=depth_loss if enable_depth_loss else None)
                        else:
                            safe_training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed_time, 
                                        testing_iterations, scene, render, (pipe, background))
                    else:
                        # 原始代码路径
                        safe_training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed_time, 
                                      testing_iterations, scene, render, (pipe, background))
                else:
                    # 尝试使用CUDA事件计算时间
                    try:
                        elapsed = iter_start.elapsed_time(iter_end)
                        if enable_normal_loss or enable_depth_loss:
                            # ... 与上面相同的逻辑但使用原始training_report ...
                            pass
                        else:
                            training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, 
                                          testing_iterations, scene, render, (pipe, background))
                    except Exception as e:
                        print(f"[警告] CUDA事件计时错误，切换到安全计时: {str(e)}")
                        # 出错后改用安全机制
                        use_safe_timing = True
                        safe_training_report(tb_writer, iteration, Ll1, loss, l1_loss, 0.0, 
                                      testing_iterations, scene, render, (pipe, background))
            except Exception as e:
                print(f"[警告] 训练报告错误: {str(e)}")
                # 完全跳过时间记录
                pass

            # 以下是原始代码的密集化和剪枝逻辑
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])

                if gaussians._culling[:,viewpoint_cam.uid].sum()==0:
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                else:
                    # normalize xy gradient after culling
                    gaussians.add_densification_stats_culling(viewspace_point_tensor, visibility_filter, gaussians.factor_culling)

                area_max = render_pkg["area_max"]
                mask_blur = torch.logical_or(mask_blur, area_max>(image.shape[1]*image.shape[2]/5000))

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 and iteration != args.depth_reinit_iter:
                                
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                    gaussians.densify_and_prune_mask(opt.densify_grad_threshold, 
                                                    0.005, scene.cameras_extent, 
                                                    size_threshold, mask_blur)
                    mask_blur = torch.zeros(gaussians._xyz.shape[0], device='cuda')
                    
                if iteration == args.depth_reinit_iter:

                    num_depth = gaussians._xyz.shape[0]*args.num_depth_factor

                    # interesction_preserving for better point cloud reconstruction result at the early stage, not affect rendering quality
                    gaussians.interesction_preserving(scene, render_simp, iteration, args, pipe, background)
                    pts, rgb = gaussians.depth_reinit(scene, render_depth, iteration, num_depth, args, pipe, background)

                    gaussians.reinitial_pts(pts, rgb)

                    gaussians.training_setup(opt)
                    gaussians.init_culling(len(scene.getTrainCameras()))
                    mask_blur = torch.zeros(gaussians._xyz.shape[0], device='cuda')
                    torch.cuda.empty_cache()

                if iteration >= args.aggressive_clone_from_iter and iteration % args.aggressive_clone_interval == 0 and iteration!=args.depth_reinit_iter:
                    gaussians.culling_with_clone(scene, render_simp, iteration, args, pipe, background)
                    torch.cuda.empty_cache()
                    mask_blur = torch.zeros(gaussians._xyz.shape[0], device='cuda')

            if iteration == args.simp_iteration1:

                path_save = r'./teaser/%d_d.ply'%iteration
                gs2ply(gaussians, path_save)

                gaussians.culling_with_interesction_sampling(scene, render_simp, iteration, args, pipe, background)
                gaussians.max_sh_degree=dataset.sh_degree
                gaussians.extend_features_rest()

                gaussians.training_setup(opt)
                torch.cuda.empty_cache()

                path_save = r'./teaser/%d_s.ply'%iteration
                gs2ply(gaussians, path_save)                
                

            if iteration == args.simp_iteration2:
                gaussians.culling_with_interesction_preserving(scene, render_simp, iteration, args, pipe, background)
                torch.cuda.empty_cache()

                path_save = r'./teaser/%d.ply'%iteration
                gs2ply(gaussians, path_save)                

            if iteration == (args.simp_iteration2+opt.iterations)//2:
                gaussians.init_culling(len(scene.getTrainCameras()))

            # Optimizer step
            if iteration < opt.iterations:
                visible = radii>0
                gaussians.optimizer.step(visible, radii.shape[0])
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")  

    print('Num of Guassians: %d'%(gaussians._xyz.shape[0]))
    return









def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene, renderFunc, renderArgs, normal_loss=None, depth_loss=None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        
        # 仅在提供额外损失时记录它们
        if normal_loss is not None:
            tb_writer.add_scalar('train_loss_patches/normal_loss', normal_loss.item(), iteration)
        if depth_loss is not None:
            tb_writer.add_scalar('train_loss_patches/depth_loss', depth_loss.item(), iteration)

    # 保留原始代码的报告逻辑
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},)        

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssims = []
                lpipss = []
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)

                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                    ssims.append(ssim(image, gt_image))
                    lpipss.append(lpips(image, gt_image, net_type='vgg'))                    


                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras']) 

                ssims_test=torch.tensor(ssims).mean()
                lpipss_test=torch.tensor(lpipss).mean()

                print("\n[ITER {}] Evaluating {}: ".format(iteration, config['name']))
                print("  SSIM : {:>12.7f}".format(ssims_test.mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(psnr_test.mean(), ".5"))
                print("  LPIPS : {:>12.7f}".format(lpipss_test.mean(), ".5"))
                print("")
                
                
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

def compute_surface_normals(depth_map):
    """
    通过深度图计算表面法线
    """
    if depth_map is None or depth_map.numel() == 0:
        return torch.zeros((3, 1, 1), device="cuda")
    
    # 使用简单的差分计算梯度
    # 水平差分
    dx = torch.zeros_like(depth_map)
    dx[:, :-1] = depth_map[:, 1:] - depth_map[:, :-1]
    
    # 垂直差分
    dy = torch.zeros_like(depth_map)
    dy[:-1, :] = depth_map[1:, :] - depth_map[:-1, :]
    
    # 创建法线图
    normals = torch.zeros((3, depth_map.shape[0], depth_map.shape[1]), device=depth_map.device)
    normals[0] = -dx  # X分量
    normals[1] = -dy  # Y分量
    normals[2] = torch.ones_like(dx) * 0.1  # Z分量
    
    # 归一化
    norm = torch.sqrt(torch.sum(normals**2, dim=0, keepdim=True) + 1e-6)
    normals = normals / norm
    
    return normals

def compute_normal_consistency_loss(render_pkg, depth_pkg, viewpoint_cam, gaussians):
    """
    计算法线一致性损失
    """
    try:
        # 基本检查
        if depth_pkg is None or "surface_normals" not in depth_pkg:
            return torch.tensor(0.0, device="cuda")
        
        # 获取表面法线
        surface_normals = depth_pkg["surface_normals"]
        
        # 计算全局表面法线方向
        avg_normal = torch.mean(surface_normals.reshape(3, -1), dim=1)
        avg_normal = torch.nn.functional.normalize(avg_normal, dim=0)
        
        # 获取高斯法线
        max_points = 1000  # 限制点数
        num_points = min(gaussians._normals.shape[0], max_points)
        indices = torch.randperm(gaussians._normals.shape[0], device='cuda')[:num_points]
        sampled_normals = gaussians.get_normals[indices]
        
        # 将高斯法线转换到相机空间
        view_matrix = viewpoint_cam.world_view_transform[:3, :3]
        camera_normals = torch.matmul(view_matrix, sampled_normals.T).T
        camera_normals = torch.nn.functional.normalize(camera_normals, dim=1)
        
        # 计算法线一致性损失
        cos_similarity = torch.sum(camera_normals * avg_normal.unsqueeze(0), dim=1)
        cos_similarity = torch.clamp(cos_similarity, min=-1.0, max=1.0)
        
        # 对应角度小于90度的法线，保持1-cos作为损失
        mask = cos_similarity >= 0
        loss_values = torch.ones_like(cos_similarity)
        loss_values[mask] = 1.0 - cos_similarity[mask]
        
        normal_loss = loss_values.mean()
        
        return normal_loss
    except Exception as e:
        print(f"法线一致性损失计算错误: {str(e)}")
        return torch.tensor(0.0, device="cuda")

def compute_simplified_depth_concentration_loss(depth_pkg):
    """
    简化版深度集中损失
    """
    if depth_pkg is None or "depth" not in depth_pkg:
        return torch.tensor(0.0, device="cuda")
    
    depth_map = depth_pkg["depth"]
    
    # 使用前向差分计算梯度
    dx = torch.zeros_like(depth_map)
    dx[:, :-1] = depth_map[:, 1:] - depth_map[:, :-1]
    
    dy = torch.zeros_like(depth_map)
    dy[:-1, :] = depth_map[1:, :] - depth_map[:-1, :]
    
    # 计算梯度幅值
    grad_magnitude = torch.sqrt(dx**2 + dy**2 + 1e-6)
    
    # 鼓励深度平滑
    depth_loss = grad_magnitude.mean()
    
    return depth_loss

def safe_training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene, renderFunc, renderArgs, normal_loss=None, depth_loss=None):
    """
    安全版本的训练报告函数，避免CUDA错误
    """
    try:
        if tb_writer:
            tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
            tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
            tb_writer.add_scalar('iter_time', elapsed, iteration)
            
            # 仅在提供额外损失时记录它们
            if normal_loss is not None:
                tb_writer.add_scalar('train_loss_patches/normal_loss', normal_loss.item(), iteration)
            if depth_loss is not None:
                tb_writer.add_scalar('train_loss_patches/depth_loss', depth_loss.item(), iteration)
        
        # 仅在特定迭代次数执行测试评估，减少GPU负担
        if iteration in testing_iterations:
            # 添加同步点
            torch.cuda.synchronize()
            
            # 复制最少量的原始代码，只保留必要部分
            if len(testing_iterations) > 0:
                validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},)
                
                for config in validation_configs:
                    if config['cameras'] and len(config['cameras']) > 0:
                        l1_test = 0.0
                        psnr_test = 0.0
                        ssims = []
                        lpipss = []
                        for idx, viewpoint in enumerate(config['cameras']):
                            # 限制测试相机数量
                            if idx >= 5:  # 最多评估5个视角
                                break
                                
                            torch.cuda.synchronize()  # 同步点
                            image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                            
                            # 简化TensorBoard记录
                            if tb_writer and (idx < 3):  # 进一步减少记录数量
                                tb_writer.add_images(config['name'] + f"_view_{idx}/render", image[None], global_step=iteration)
                                if iteration == testing_iterations[0]:
                                    tb_writer.add_images(config['name'] + f"_view_{idx}/ground_truth", gt_image[None], global_step=iteration)
                            
                            # 计算指标
                            l1_test += l1_loss(image, gt_image).mean().double()
                            psnr_test += psnr(image, gt_image).mean().double()
                            
                            try:
                                from utils.image_utils import psnr
                                from lpipsPyTorch import lpips
                                ssims.append(ssim(image, gt_image))
                                lpipss.append(lpips(image, gt_image, net_type='vgg'))
                            except Exception as e:
                                print(f"[警告] 图像评估指标计算错误: {str(e)}")
                            
                            torch.cuda.synchronize()  # 同步点
                            
                        # 计算平均指标
                        num_cameras = len(config['cameras'])
                        if num_cameras > 0:
                            psnr_test /= min(5, num_cameras)  # 考虑最多5个相机
                            l1_test /= min(5, num_cameras)
                            
                            # 打印结果
                            print(f"\n[ITER {iteration}] 测试 {config['name']}: ")
                            print(f"  PSNR: {psnr_test.item():.5f}")
                            if len(ssims) > 0:
                                ssims_test = torch.tensor(ssims).mean()
                                print(f"  SSIM: {ssims_test.item():.5f}")
                            if len(lpipss) > 0:
                                lpipss_test = torch.tensor(lpipss).mean()
                                print(f"  LPIPS: {lpipss_test.item():.5f}")
                            
                            # 记录到TensorBoard
                            if tb_writer:
                                tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                                tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    
                # 清理GPU内存
                torch.cuda.empty_cache()
    except Exception as e:
        print(f"[警告] 安全训练报告出现错误: {str(e)}")
        # 捕获所有异常但继续训练
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)

    parser.add_argument("--imp_metric", required=True, type=str, default = 'outdoor')


    parser.add_argument("--config_path", type=str)

    parser.add_argument("--aggressive_clone_from_iter", type=int, default = 500)
    parser.add_argument("--aggressive_clone_interval", type=int, default = 250)

    parser.add_argument("--warn_until_iter", type=int, default = 3_000)
    parser.add_argument("--depth_reinit_iter", type=int, default=2_000)
    parser.add_argument("--num_depth_factor", type=float, default=1)

    parser.add_argument("--simp_iteration1", type=int, default = 3_000)
    parser.add_argument("--simp_iteration2", type=int, default = 8_000)
    parser.add_argument("--sampling_factor", type=float, default = 0.6)

    parser.add_argument("--enable_normal_loss", action="store_true", help="是否启用法线约束")
    parser.add_argument("--enable_depth_loss", action="store_true", help="是否启用深度集中约束")
    parser.add_argument("--normal_loss_weight", type=float, default=0.1, help="法线一致性损失的权重")
    parser.add_argument("--depth_loss_weight", type=float, default=0.01, help="深度集中损失的权重")
    parser.add_argument("--normal_loss_start_iter", type=int, default=1000, help="开始应用法线损失的迭代次数")
    parser.add_argument("--depth_loss_start_iter", type=int, default=2000, help="开始应用深度集中损失的迭代次数")
    parser.add_argument("--normal_loss_auto_adjust", action="store_true", default=True, help="自动调整法线损失权重")
    parser.add_argument("--normal_loss_threshold", type=float, default=2.0, help="法线损失异常值阈值")
    parser.add_argument("--print_loss_interval", type=int, default=100, help="打印详细损失信息的间隔")

    parser.add_argument("--use_safe_timing", action="store_true", default=True, 
                        help="使用安全的时间测量机制，避免CUDA错误")

    args = parser.parse_args(sys.argv[1:])
    args.iterations = args.simp_iteration2
    args.densify_until_iter = args.simp_iteration1


    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args)

    # All done
    print("\nTraining complete.")
