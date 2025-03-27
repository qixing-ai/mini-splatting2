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
    
    # 获取损失相关参数
    enable_normal_loss = getattr(args, 'enable_normal_loss', False)
    enable_depth_loss = getattr(args, 'enable_depth_loss', False)
    
    # 重要修复：总是获取max_points_limit参数
    max_points_limit = getattr(args, 'max_points_limit', 1000000)
    
    # 内存管理参数
    memory_cleanup_interval = getattr(args, 'memory_cleanup_interval', 100)
    
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
    
    # 打印点数限制信息
    print(f"高斯点数量上限：{max_points_limit}")
    
    # 修复：禁用过于频繁的CUDA事件记录，改为使用Python的时间测量
    use_cuda_events = False  # 关闭CUDA事件计时，使用Python时间代替
    start_time = time.time()

    for iteration in range(first_iter, opt.iterations + 1):
        # 修复：只在需要时记录CUDA事件
        if use_cuda_events:
            iter_start.record()
        else:
            # 使用Python时间替代CUDA事件
            current_time = time.time()
            elapsed_ms = (current_time - start_time) * 1000  # 转换为毫秒
            start_time = current_time
        
        # 内存管理 - 减少频率并确保同步
        if iteration % memory_cleanup_interval == 0:
            # 重要：在清理缓存前同步
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            # 重要：清理后也同步
            torch.cuda.synchronize()
        
        # 限制高斯点数量 - 对所有训练模式有效，但减少频率
        if gaussians._xyz.shape[0] > max_points_limit and iteration % 10 == 0:  # 降低剪枝频率
            print(f"[警告] 迭代 {iteration}: 高斯点数量({gaussians._xyz.shape[0]})超过限制({max_points_limit})，执行额外剪枝")
            
            # 同步GPU，确保之前的操作完成
            torch.cuda.synchronize()
            
            # 计算重要性分数
            importance = gaussians.get_opacity.squeeze()
            _, indices = torch.sort(importance, descending=True)
            keep_indices = indices[:max_points_limit]
            
            # 创建掩码标记要保留的点
            mask = torch.zeros(gaussians._xyz.shape[0], dtype=torch.bool, device="cuda")
            mask[keep_indices] = True
            
            # 剪枝
            gaussians.prune_points(~mask)
            
            # 重置mask_blur以匹配新的点数
            mask_blur = torch.zeros(gaussians._xyz.shape[0], device='cuda')
            
            # 同步GPU，确保剪枝操作完成
            torch.cuda.synchronize()
            
            print(f"[内存管理] 迭代 {iteration}: 剪枝后高斯点数量: {gaussians._xyz.shape[0]}")

        # 图形界面处理（保持不变）
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

        # 更新学习率
        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0 and iteration>args.simp_iteration1:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras_warn_up(iteration, args.warn_until_iter, scale=1.0, scale2=2.0).copy()

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        if (iteration - 1) == debug_from:
            pipe.debug = True

        render_pkg = render_imp(viewpoint_cam, gaussians, pipe, background, culling=gaussians._culling[:,viewpoint_cam.uid])
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        photo_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        if enable_normal_loss or enable_depth_loss:
            normal_loss = torch.tensor(0.0, device="cuda")
            depth_loss = torch.tensor(0.0, device="cuda")
            
            if (enable_normal_loss and iteration > normal_loss_start_iter) or (enable_depth_loss and iteration > depth_loss_start_iter):
                try:
                    with torch.no_grad():
                        rgb = render_pkg["render"].detach()
                        depth_map = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
                        
                        surface_normals = compute_surface_normals(depth_map)
                        
                        depth_pkg = {
                            "depth": depth_map,
                            "surface_normals": surface_normals
                        }
                    
                    if enable_normal_loss and iteration > normal_loss_start_iter:
                        normal_loss = compute_normal_consistency_loss(render_pkg, depth_pkg, viewpoint_cam, gaussians)
                        if not torch.isfinite(normal_loss) or normal_loss > normal_loss_threshold:
                            print(f"[警告] 迭代 {iteration}: 法线损失值异常({normal_loss.item():.4f})，使用0.0替代")
                            normal_loss = torch.tensor(0.0, device="cuda")
                    
                    if enable_depth_loss and iteration > depth_loss_start_iter:
                        depth_loss = compute_simplified_depth_concentration_loss(depth_pkg)
                        if not torch.isfinite(depth_loss) or depth_loss > 1.0:
                            print(f"[警告] 迭代 {iteration}: 深度损失值异常({depth_loss.item():.4f})，使用0.0替代")
                            depth_loss = torch.tensor(0.0, device="cuda")
                except Exception as e:
                    print(f"[错误] 迭代 {iteration}: 约束计算错误: {str(e)}")
                    normal_loss = torch.tensor(0.0, device="cuda")
                    depth_loss = torch.tensor(0.0, device="cuda")
                    torch.cuda.empty_cache()

            total_loss = photo_loss
            normal_term = torch.tensor(0.0, device="cuda")
            depth_term = torch.tensor(0.0, device="cuda")
            
            if normal_loss > 0:
                normal_term = normal_loss_weight * normal_loss
                total_loss = total_loss + normal_term
            if depth_loss > 0:
                depth_term = depth_loss_weight * depth_loss
                total_loss = total_loss + depth_term

            total_loss.backward()
            
            ema_loss_for_log = 0.4 * total_loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.{7}f}",
                    "NormLoss": f"{normal_loss.item():.{4}f}" if normal_loss > 0 else "0.0",
                    "DepthLoss": f"{depth_loss.item():.{4}f}" if depth_loss > 0 else "0.0"
                })
                progress_bar.update(10)
            
            if iteration % print_loss_interval == 0:
                print(f"[迭代 {iteration}] 光度损失: {photo_loss.item():.4f}, "
                    f"法线损失: {normal_loss.item():.4f} (权重项: {normal_term.item():.4f}), "
                    f"深度损失: {depth_loss.item():.4f} (权重项: {depth_term.item():.4f}), "
                    f"总损失: {total_loss.item():.4f}")
        else:
            loss = photo_loss
            loss.backward()
            
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

        # 修复：只在使用CUDA事件时记录结束事件
        if use_cuda_events:
            iter_end.record()
            # 重要修复：确保CUDA操作同步
            torch.cuda.synchronize()

        with torch.no_grad():
            # 记录和保存
            try:
                # 使用Python时间或CUDA事件，取决于设置
                if use_cuda_events:
                    elapsed_time = iter_start.elapsed_time(iter_end) if iteration > 1 else 0.0
                else:
                    elapsed_time = elapsed_ms  # 使用Python计时器的结果
                
                if enable_normal_loss or enable_depth_loss:
                    training_report(tb_writer, iteration, Ll1, total_loss, l1_loss, elapsed_time, 
                                testing_iterations, scene, render, (pipe, background),
                                normal_loss=normal_loss if enable_normal_loss else None,
                                depth_loss=depth_loss if enable_depth_loss else None)
                else:
                    training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed_time, 
                                  testing_iterations, scene, render, (pipe, background))
            except Exception as e:
                print(f"[警告] 迭代 {iteration}: 时间计算错误: {str(e)}")
                # 使用零替代
                if enable_normal_loss or enable_depth_loss:
                    training_report(tb_writer, iteration, Ll1, total_loss, l1_loss, 0.0, 
                                testing_iterations, scene, render, (pipe, background),
                                normal_loss=normal_loss if enable_normal_loss else None,
                                depth_loss=depth_loss if enable_depth_loss else None)
                else:
                    training_report(tb_writer, iteration, Ll1, loss, l1_loss, 0.0, 
                                  testing_iterations, scene, render, (pipe, background))

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if iteration < opt.densify_until_iter:
                mask_blur = torch.zeros(gaussians._xyz.shape[0], device='cuda')
                
                if 'area_max' in render_pkg:
                    area_max = render_pkg["area_max"]
                    valid_range = min(mask_blur.shape[0], area_max.shape[0])
                    mask_blur[:valid_range] = area_max[:valid_range] > (image.shape[1]*image.shape[2]/5000)
                
                if gaussians._xyz.shape[0] > max_points_limit and iteration % opt.densification_interval == 0:
                    print(f"[警告] 迭代 {iteration}: 密集化前高斯点数量({gaussians._xyz.shape[0]})超过限制({max_points_limit})，执行额外剪枝")
                    
                    importance = gaussians.get_opacity.squeeze()
                    _, indices = torch.sort(importance, descending=True)
                    keep_indices = indices[:max_points_limit]
                    
                    prune_mask = torch.ones(gaussians._xyz.shape[0], dtype=torch.bool, device="cuda")
                    prune_mask[keep_indices] = False
                    
                    gaussians.prune_points(prune_mask)
                    mask_blur = torch.zeros(gaussians._xyz.shape[0], device='cuda')
                    print(f"[内存管理] 迭代 {iteration}: 剪枝后高斯点数量: {gaussians._xyz.shape[0]}")
                
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 and iteration != args.depth_reinit_iter:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    
                    gaussians.densify_and_prune_mask(
                        opt.densify_grad_threshold, 
                        0.005, 
                        scene.cameras_extent, 
                        size_threshold, 
                        mask_blur
                    )

                if iteration == args.depth_reinit_iter:
                    mask_blur = torch.zeros(gaussians._xyz.shape[0], device='cuda')
                    print(f"[信息] 迭代 {iteration}: 深度重初始化后重置mask_blur，新形状: {mask_blur.shape[0]}")

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

            if iteration < opt.iterations:
                visible = render_pkg["visibility_filter"]>0
                gaussians.optimizer.step(visible, render_pkg["visibility_filter"].shape[0])
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
        # 只有在提供额外损失时才记录它们
        if normal_loss is not None:
            tb_writer.add_scalar('train_loss_patches/normal_loss', normal_loss.item(), iteration)
        if depth_loss is not None:
            tb_writer.add_scalar('train_loss_patches/depth_loss', depth_loss.item(), iteration)

    # Report test and samples of training set
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

def compute_normal_consistency_loss(render_pkg, depth_pkg, viewpoint_cam, gaussians):
    """
    计算法线一致性损失 - 修复异常值问题
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
        
        # 获取高斯法线 - 随机采样以提高效率
        max_points = 1000  # 限制采样点数
        num_points = min(gaussians._normals.shape[0], max_points)
        indices = torch.randperm(gaussians._normals.shape[0], device='cuda')[:num_points]
        sampled_normals = gaussians.get_normals[indices]
        
        # 将高斯法线转换到相机空间
        view_matrix = viewpoint_cam.world_view_transform[:3, :3]
        camera_normals = torch.matmul(view_matrix, sampled_normals.T).T
        camera_normals = torch.nn.functional.normalize(camera_normals, dim=1)
        
        # 计算法线一致性损失 - 使用余弦相似度
        cos_similarity = torch.sum(camera_normals * avg_normal.unsqueeze(0), dim=1)
        
        # 修复：使用修改后的损失计算方式
        # 限制余弦值到[-1, 1]范围
        cos_similarity = torch.clamp(cos_similarity, min=-1.0, max=1.0)
        
        # 对应角度小于90度的法线，保持1-cos作为损失
        # 对于角度大于90度的法线，使用恒定值1.0作为损失（避免惩罚过大）
        mask = cos_similarity >= 0  # 角度小于90度
        loss_values = torch.ones_like(cos_similarity)
        loss_values[mask] = 1.0 - cos_similarity[mask]  # 角度小于90度的部分使用1-cos
        
        normal_loss = loss_values.mean()
        
        return normal_loss
    except Exception as e:
        print(f"法线一致性损失计算错误: {str(e)}")
        return torch.tensor(0.0, device="cuda")

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
    normals[2] = torch.ones_like(dx) * 0.1  # Z分量（较小值使法线更加平行于表面）
    
    # 归一化
    norm = torch.sqrt(torch.sum(normals**2, dim=0, keepdim=True) + 1e-6)
    normals = normals / norm
    
    return normals

def compute_depth_concentration_loss(depth_pkg):
    """
    计算射线上高斯深度集中损失
    depth_pkg: 包含depths_per_ray和weights_per_ray的字典
    """
    # 检查是否有有效的深度和权重数据
    if depth_pkg is None or "depths_per_ray" not in depth_pkg or depth_pkg["depths_per_ray"] is None:
        return torch.tensor(0.0, device="cuda")
    
    depths_per_ray = depth_pkg["depths_per_ray"]  # [B, N, H*W]
    weights_per_ray = depth_pkg["weights_per_ray"]  # [B, N, H*W]
    
    # 射线数量
    num_rays = depths_per_ray.shape[-1]
    # 每个射线上的高斯数量
    num_gaussians_per_ray = depths_per_ray.shape[1]
    
    # 初始化总损失
    total_loss = torch.tensor(0.0, device="cuda")
    
    # 对每个射线计算深度集中损失
    for ray_idx in range(num_rays):
        # 获取当前射线上的深度和权重
        ray_depths = depths_per_ray[0, :, ray_idx]  # [N]
        ray_weights = weights_per_ray[0, :, ray_idx]  # [N]
        
        # 过滤有效的深度和权重(权重不为零)
        valid_mask = ray_weights > 0
        valid_depths = ray_depths[valid_mask]
        valid_weights = ray_weights[valid_mask]
        
        # 如果射线上没有有效高斯，跳过
        if valid_depths.shape[0] <= 1:
            continue
        
        # 计算深度差的绝对值
        depth_diffs = torch.abs(valid_depths.unsqueeze(1) - valid_depths.unsqueeze(0))  # [M, M]
        
        # 计算权重乘积矩阵
        weight_products = valid_weights.unsqueeze(1) * valid_weights.unsqueeze(0)  # [M, M]
        
        # 计算加权深度差
        weighted_diffs = depth_diffs * weight_products
        
        # 对矩阵上三角部分(不包括对角线)求和
        upper_tri_mask = torch.triu(torch.ones_like(weighted_diffs), diagonal=1).bool()
        ray_loss = weighted_diffs[upper_tri_mask].sum()
        
        # 添加到总损失
        total_loss += ray_loss
    
    # 除以射线数量，得到平均损失
    if num_rays > 0:
        total_loss = total_loss / num_rays
    
    return total_loss

def compute_simplified_depth_concentration_loss(depth_pkg):
    """
    简化版深度集中损失 - 使用深度图的平滑性作为代理
    """
    if depth_pkg is None or "depth" not in depth_pkg:
        return torch.tensor(0.0, device="cuda")
    
    depth_map = depth_pkg["depth"]  # [H, W]
    
    # 使用简单的前向差分计算梯度，避免额外的维度转换
    # 水平差分
    dx = torch.zeros_like(depth_map)
    dx[:, :-1] = depth_map[:, 1:] - depth_map[:, :-1]
    
    # 垂直差分
    dy = torch.zeros_like(depth_map)
    dy[:-1, :] = depth_map[1:, :] - depth_map[:-1, :]
    
    # 计算梯度幅值
    grad_magnitude = torch.sqrt(dx**2 + dy**2 + 1e-6)
    
    # 鼓励深度平滑(梯度小)
    depth_loss = grad_magnitude.mean()
    
    return depth_loss

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

    parser.add_argument("--normal_loss_weight", type=float, default=0.1, help="法线一致性损失的权重")
    parser.add_argument("--depth_loss_weight", type=float, default=0.01, help="深度集中损失的权重")
    parser.add_argument("--enable_normal_loss", action="store_true", help="是否启用法线约束")
    parser.add_argument("--enable_depth_loss", action="store_true", help="是否启用深度集中约束")
    parser.add_argument("--normal_loss_start_iter", type=int, default=1000, help="开始应用法线损失的迭代次数")
    parser.add_argument("--depth_loss_start_iter", type=int, default=2000, help="开始应用深度集中损失的迭代次数")
    parser.add_argument("--max_points_limit", type=int, default=1000000, help="高斯点数量上限")

    parser.add_argument("--normal_loss_auto_adjust", action="store_true", default=True, help="自动调整法线损失权重")
    parser.add_argument("--normal_loss_threshold", type=float, default=2.0, help="法线损失异常值阈值")
    parser.add_argument("--print_loss_interval", type=int, default=100, help="打印详细损失信息的间隔")
    parser.add_argument("--memory_cleanup_interval", type=int, default=100, help="内存清理间隔")

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
