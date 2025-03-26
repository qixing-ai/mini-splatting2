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

    # 修改默认值，使法线约束立即生效
    normal_loss_weight = getattr(args, 'normal_loss_weight', 0.05)
    enable_normal_loss = getattr(args, 'enable_normal_loss', True)  # 默认开启
    normal_loss_start_iter = getattr(args, 'normal_loss_start_iter', 500)  # 开始较早
    
    # 打印法线约束设置信息
    print(f"法线约束设置: 启用={enable_normal_loss}, 权重={normal_loss_weight}, 开始迭代={normal_loss_start_iter}")
    
    # 添加内存管理参数
    memory_cleanup_interval = 100  # 每隔多少次迭代清理一次内存
    max_points_limit = getattr(args, 'max_points_limit', 1000000)  # 限制高斯点数量
    
    for iteration in range(first_iter, opt.iterations + 1):
        # 每隔一定迭代次数强制清理GPU内存
        if iteration % memory_cleanup_interval == 0:
            torch.cuda.empty_cache()
            print(f"[内存管理] 迭代 {iteration}: 强制清理GPU缓存")
        
        # 限制高斯点数量以防止内存爆炸
        if gaussians._xyz.shape[0] > max_points_limit:
            print(f"[警告] 迭代 {iteration}: 高斯点数量({gaussians._xyz.shape[0]})超过限制({max_points_limit})，执行额外剪枝")
            # 计算重要性分数
            importance = gaussians.get_opacity.squeeze()
            # 保留最重要的点
            _, indices = torch.sort(importance, descending=True)
            keep_indices = indices[:max_points_limit]
            
            # 创建掩码标记要保留的点
            mask = torch.zeros(gaussians._xyz.shape[0], dtype=torch.bool, device="cuda")
            mask[keep_indices] = True
            
            # 剪枝
            gaussians.prune_points(~mask)
            print(f"[内存管理] 迭代 {iteration}: 剪枝后高斯点数量: {gaussians._xyz.shape[0]}")
        
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

        iter_start.record()

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

        # 计算原始光度损失
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        photo_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        
        # 计算法线损失
        normal_loss = torch.tensor(0.0, device="cuda")
        if enable_normal_loss and iteration > normal_loss_start_iter:
            try:
                with torch.no_grad():
                    # 获取渲染图像的灰度值作为简单深度估计
                    rgb = render_pkg["render"].detach()
                    depth_map = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
                    
                    # 使用修复后的函数计算表面法线
                    surface_normals = compute_surface_normals(depth_map)
                    
                    # 创建深度包
                    depth_pkg = {
                        "depth": depth_map,
                        "surface_normals": surface_normals
                    }
                
                # 计算法线一致性损失
                normal_loss = compute_normal_consistency_loss(render_pkg, depth_pkg, viewpoint_cam, gaussians)
                
                if not torch.isfinite(normal_loss) or normal_loss.numel() > 1:
                    normal_loss = torch.tensor(0.0, device="cuda")
            except Exception as e:
                print(f"[错误] 迭代 {iteration}: 法线计算错误: {str(e)}")
                normal_loss = torch.tensor(0.0, device="cuda")
        
        # 构建最终损失 - 始终使用光度损失
        if normal_loss > 0 and torch.isfinite(normal_loss):
            # 先反向传播光度损失
            photo_loss.backward(retain_graph=True)  # 保留计算图以便继续
            # 再单独反向传播法线损失
            (normal_loss_weight * normal_loss).backward()
        else:
            # 只有光度损失
            photo_loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * photo_loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.{7}f}",
                    "NormLoss": f"{normal_loss.item():.{7}f}"
                })
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, photo_loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # # Densification
            if iteration < opt.densify_until_iter:
                # 每次都创建新的mask_blur，确保尺寸匹配
                mask_blur = torch.zeros(gaussians._xyz.shape[0], device='cuda')
                
                if 'area_max' in render_pkg:
                    area_max = render_pkg["area_max"]
                    # 只处理有效范围内的元素
                    valid_range = min(mask_blur.shape[0], area_max.shape[0])
                    mask_blur[:valid_range] = area_max[:valid_range] > (image.shape[1]*image.shape[2]/5000)
                
                # 在密集化之前，应用点数量限制
                if gaussians._xyz.shape[0] > max_points_limit and iteration % opt.densification_interval == 0:
                    print(f"[警告] 迭代 {iteration}: 密集化前高斯点数量({gaussians._xyz.shape[0]})超过限制({max_points_limit})，执行额外剪枝")
                    
                    # 计算重要性分数
                    importance = gaussians.get_opacity.squeeze()
                    _, indices = torch.sort(importance, descending=True)
                    keep_indices = indices[:max_points_limit]
                    
                    # 创建掩码
                    prune_mask = torch.ones(gaussians._xyz.shape[0], dtype=torch.bool, device="cuda")
                    prune_mask[keep_indices] = False
                    
                    # 剪枝
                    gaussians.prune_points(prune_mask)
                    # 重置mask_blur
                    mask_blur = torch.zeros(gaussians._xyz.shape[0], device='cuda')
                    print(f"[内存管理] 迭代 {iteration}: 剪枝后高斯点数量: {gaussians._xyz.shape[0]}")
                
                # 密集化和剪枝
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 and iteration != args.depth_reinit_iter:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    
                    # 传递mask_blur参数
                    gaussians.densify_and_prune_mask(
                        opt.densify_grad_threshold, 
                        0.005, 
                        scene.cameras_extent, 
                        size_threshold, 
                        mask_blur
                    )

                if iteration == args.depth_reinit_iter:
                    # 在深度重新初始化后重置mask_blur
                    mask_blur = torch.zeros(gaussians._xyz.shape[0], device='cuda')
                    print(f"[信息] 迭代 {iteration}: 深度重初始化后重置mask_blur，新形状: {mask_blur.shape[0]}")

                if iteration >= args.aggressive_clone_from_iter and iteration % args.aggressive_clone_interval == 0 and iteration!=args.depth_reinit_iter:
                    gaussians.culling_with_clone(scene, render_simp, iteration, args, pipe, background)
                    torch.cuda.empty_cache()
                    mask_blur = torch.zeros(gaussians._xyz.shape[0], device='cuda')
                    # print(gaussians._xyz.shape)

            if iteration == args.simp_iteration1:

                path_save = r'./teaser/%d_d.ply'%iteration
                gs2ply(gaussians, path_save)

                gaussians.culling_with_interesction_sampling(scene, render_simp, iteration, args, pipe, background)
                gaussians.max_sh_degree=dataset.sh_degree
                gaussians.extend_features_rest()

                gaussians.training_setup(opt)
                torch.cuda.empty_cache()
                # print(gaussians._xyz.shape)

                path_save = r'./teaser/%d_s.ply'%iteration
                gs2ply(gaussians, path_save)                
                

            if iteration == args.simp_iteration2:
                gaussians.culling_with_interesction_preserving(scene, render_simp, iteration, args, pipe, background)
                torch.cuda.empty_cache()
                # print(gaussians._xyz.shape)

                path_save = r'./teaser/%d.ply'%iteration
                gs2ply(gaussians, path_save)                

            if iteration == (args.simp_iteration2+opt.iterations)//2:
                gaussians.init_culling(len(scene.getTrainCameras()))



            # Optimizer step
            if iteration < opt.iterations:
                visible = render_pkg["visibility_filter"]>0
                gaussians.optimizer.step(visible, render_pkg["visibility_filter"].shape[0])
                # gaussians.optimizer.step()
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

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

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
    计算法线一致性损失 - 简化版本
    """
    try:
        # 基本检查
        if depth_pkg is None or "surface_normals" not in depth_pkg:
            return torch.tensor(0.0, device="cuda")
        
        # 获取表面法线
        surface_normals = depth_pkg["surface_normals"]
        
        # 计算全局表面法线方向 (平均值)
        avg_normal = torch.mean(surface_normals.reshape(3, -1), dim=1)
        avg_normal = torch.nn.functional.normalize(avg_normal, dim=0)
        
        # 获取随机采样的高斯点法线
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
        normal_loss = (1.0 - cos_similarity).mean()
        
        return normal_loss
    except Exception as e:
        print(f"法线一致性损失计算错误: {str(e)}")
        return torch.tensor(0.0, device="cuda")

def compute_surface_normals(depth_map):
    """
    通过深度图计算表面法线 - 避免使用pad函数
    """
    if depth_map is None or depth_map.numel() == 0:
        return torch.zeros((3, 1, 1), device="cuda")
    
    # 获取图像尺寸
    H, W = depth_map.shape
    
    # 创建法线图
    normals = torch.zeros((3, H, W), device=depth_map.device)
    
    # 使用简单的滑动差分计算梯度
    # 水平差分(中心差分)
    dx = torch.zeros_like(depth_map)
    dx[:, 1:-1] = depth_map[:, 2:] - depth_map[:, :-2]  # 中心点的梯度
    dx[:, 0] = depth_map[:, 1] - depth_map[:, 0]  # 左边界
    dx[:, -1] = depth_map[:, -1] - depth_map[:, -2]  # 右边界
    
    # 垂直差分(中心差分)
    dy = torch.zeros_like(depth_map)
    dy[1:-1, :] = depth_map[2:, :] - depth_map[:-2, :]  # 中心点的梯度
    dy[0, :] = depth_map[1, :] - depth_map[0, :]  # 上边界
    dy[-1, :] = depth_map[-1, :] - depth_map[-2, :]  # 下边界
    
    # 平滑梯度以减少噪声
    kernel_size = 3
    dx = torch.nn.functional.avg_pool2d(
        dx.unsqueeze(0),  # 添加批次维度
        kernel_size=kernel_size, 
        stride=1, 
        padding=kernel_size//2
    ).squeeze(0)
    
    dy = torch.nn.functional.avg_pool2d(
        dy.unsqueeze(0),  # 添加批次维度
        kernel_size=kernel_size, 
        stride=1, 
        padding=kernel_size//2
    ).squeeze(0)
    
    # 构建法线
    normals[0] = -dx  # X分量
    normals[1] = -dy  # Y分量
    normals[2] = torch.ones_like(dx) * 0.1  # Z分量(减小以增强XY平面细节)
    
    # 归一化
    norm = torch.sqrt(torch.sum(normals**2, dim=0, keepdim=True) + 1e-6)
    normals = normals / norm
    
    return normals

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

    parser.add_argument("--normal_loss_weight", type=float, default=0.05, help="法线一致性损失的权重")
    parser.add_argument("--enable_normal_loss", action="store_true", help="是否启用法线约束")
    parser.add_argument("--normal_loss_start_iter", type=int, default=500, help="开始应用法线损失的迭代次数")
    parser.add_argument("--max_points_limit", type=int, default=1000000, help="高斯点数量上限，超过此值将进行额外剪枝")

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
