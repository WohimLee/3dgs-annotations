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
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    # 初始化训练的起始迭代次数
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset) # 准备输出和日志记录器
    # 创建一个高斯模型，初始化需要数据集的球谐函数阶数
    gaussians = GaussianModel(dataset.sh_degree)
    # 创建场景对象，包含数据集和高斯模型
    scene = Scene(dataset, gaussians) 
    # 配置高斯模型的训练设置
    gaussians.training_setup(opt)
    if checkpoint: # 如果提供了检查点路径，则从检查点中恢复模型
        (model_params, first_iter) = torch.load(checkpoint) # 加载模型参数和起始迭代次数
        gaussians.restore(model_params, opt)    # 恢复模型状态
    # 设置背景颜色，默认为白色或黑色，取决于数据集是否指定白色背景
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda") # 将背景颜色转换为CUDA张量
    # 创建CUDA事件，用于计算迭代的开始和结束时间，以便进行性能评估
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    # 初始化视角堆栈变量，用于存储不同视角的信息
    viewpoint_stack = None
    # 初始化用于日志记录的指数移动平均损失
    ema_loss_for_log = 0.0
    # 使用tqdm库创建一个进度条，范围是从当前迭代到总迭代次数，显示训练进度
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    # 起始迭代次数自增，以避免重复执行第一次迭代
    first_iter += 1
    
    # 进入主训练循环，从first_iter开始，一直到opt.iterations + 1
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None: # 如果网络接口未连接，则尝试连接
            network_gui.try_connect()
        while network_gui.conn != None: # 如果网络接口已连接，进入内部循环等待处理指令
            try:
                net_image_bytes = None # 初始化接收到的图像字节为None
                # 从网络接口接收指令，包括自定义相机参数、是否进行训练、管道配置等
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                # 如果接收到自定义相机参数，则进行渲染
                if custom_cam != None:
                    # 调用render函数进行渲染，获取渲染结果
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    # 对渲染结果进行处理，转换为适合网络传输的字节格式
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                # 将渲染后的图像字节发送回网络接口
                network_gui.send(net_image_bytes, dataset.source_path)
                # 如果需要进行训练，并且迭代次数没有达到上限或不需要保持活动状态，则中断内循环
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            # 如果在网络通信或处理过程中出现异常，则断开连接
            except Exception as e:
                network_gui.conn = None
        # 记录迭代开始的时间点
        iter_start.record()
        # 更新高斯模型的学习率
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0: # 每1000次迭代增加一次球谐函数的阶数，直到达到最大值
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack: # 如果视点堆栈为空，则从场景中获取训练用的相机视点并复制
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1)) # 从视点堆栈中随机选取一个相机视点

        # Render 如果当前迭代次数减1等于指定的调试起始点，则开启调试模式
        if (iteration - 1) == debug_from:
            pipe.debug = True
        # 根据选项随机生成背景或使用预设背景
        bg = torch.rand((3), device="cuda") if opt.random_background else background
        # 进行渲染，获取渲染包含的多个结果
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss 获取对应视点的原始图像，并将其转移到CUDA设备上
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image) # 计算L1损失
        # 计算总损失，结合L1损失和SSIM损失
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward() # 对损失进行反向传播

        iter_end.record() # 记录迭代结束的时间点

        with torch.no_grad():
            # Progress bar 更新指数移动平均（EMA）损失以平滑显示在进度条上
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0: # 每10次迭代更新一次进度条，并显示当前的平滑损失
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations: # 当迭代到达指定的最大次数时，关闭进度条
                progress_bar.close()

            # Log and save 调用训练报告函数记录当前的训练状态，包括损失和迭代时间等
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations): # 在指定的迭代次数保存模型状态
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter: # 细化和修剪高斯模型，以更好地适应数据
                # Keep track of max radii in image-space for pruning
                # 更新图像空间中的最大半径，用于后续的修剪操作
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter) # 添加细化统计数据
                
                # 根据设定的间隔执行细化和修剪操作
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                # 在特定迭代时重置高斯模型的不透明度
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step 执行优化器的步骤，更新模型参数，并重置梯度
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                
            # 在指定的迭代次数保存检查点
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

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

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
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
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
