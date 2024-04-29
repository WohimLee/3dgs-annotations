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
import sys
from datetime import datetime
import numpy as np
import random

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

# 定义一个函数，从输入的矩阵L中提取下三角和对角线元素
def strip_lowerdiag(L):
    
    # 创建一个形状为(L的第一个维度, 6)的张量，初始化为0，数据类型为float，放置在CUDA设备上
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")
    # 从矩阵L中提取对角线和下三角的元素，并分别赋值到uncertainty张量的相应位置
    uncertainty[:, 0] = L[:, 0, 0]  # 提取第一行第一列的元素，即对角元素
    uncertainty[:, 1] = L[:, 0, 1]  # 提取第一行第二列的元素
    uncertainty[:, 2] = L[:, 0, 2]  # 提取第一行第三列的元素
    uncertainty[:, 3] = L[:, 1, 1]  # 提取第二行第二列的元素，即对角元素
    uncertainty[:, 4] = L[:, 1, 2]  # 提取第二行第三列的元素
    uncertainty[:, 5] = L[:, 2, 2]  # 提取第三行第三列的元素，即对角元素
    return uncertainty # 返回包含提取元素的张量

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

# 定义一个函数，根据四元数r构建一个3x3的旋转矩阵R
def build_rotation(r):
    # 计算四元数的模（长度），以保证它是单位四元数
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])
    # 标准化四元数，确保其长度为1
    q = r / norm[:, None]
    # 创建一个形状为(q的第一个维度, 3, 3)的零矩阵R，数据类型为float，放置在CUDA设备上
    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    # 提取标准化后的四元数的各个分量
    r = q[:, 0] # 实部
    x = q[:, 1] # i部
    y = q[:, 2] # j部
    z = q[:, 3] # k部

    # 根据四元数的各个分量，计算旋转矩阵R的各个元素
    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R # 返回构建好的旋转矩阵R

# 定义一个函数，根据缩放向量s和旋转参数r构建一个变换矩阵L
def build_scaling_rotation(s, r):
    # 创建一个形状为(s的第一个维度, 3, 3)的零矩阵L，数据类型为float，放置在CUDA设备上
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    # 调用build_rotation函数，根据旋转参数r构建一个旋转矩阵R
    R = build_rotation(r)

    # 设置矩阵L的对角线元素为缩放向量s的各个分量，形成一个缩放矩阵
    L[:,0,0] = s[:,0]   # 第一行第一列元素，对应x轴的缩放因子
    L[:,1,1] = s[:,1]   # 第二行第二列元素，对应y轴的缩放因子
    L[:,2,2] = s[:,2]   # 第三行第三列元素，对应z轴的缩放因子
    # 将旋转矩阵R与缩放矩阵L进行矩阵乘法，得到同时包含旋转和缩放信息的变换矩阵L
    L = R @ L
    return L    # 返回构建好的变换矩阵L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))
