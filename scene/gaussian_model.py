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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

class GaussianModel:
    # 设置高斯模型的激活函数和其他转换函数
    def setup_functions(self):
        # 定义根据缩放和旋转参数构建协方差矩阵的函数
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            # 通过缩放和旋转参数构建矩阵L
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            # 计算协方差矩阵，L乘以其转置
            actual_covariance = L @ L.transpose(1, 2)
            # 对协方差矩阵进行对称化处理
            symm = strip_symmetric(actual_covariance)
            return symm # 对协方差矩阵进行对称化处理

        # 设置缩放参数的激活函数为指数函数
        self.scaling_activation = torch.exp
        # 设置缩放参数的逆激活函数为对数函数
        self.scaling_inverse_activation = torch.log
        # 设置协方差参数的激活函数
        self.covariance_activation = build_covariance_from_scaling_rotation
        # 设置不透明度参数的激活函数为sigmoid函数
        self.opacity_activation = torch.sigmoid
        # 设置不透明度参数的逆激活函数
        self.inverse_opacity_activation = inverse_sigmoid
        # 设置旋转参数的激活函数为归一化函数
        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0 # 激活的球谐函数阶数
        self.max_sh_degree = sh_degree  # 最大球谐函数阶数
        # 初始化各种属性
        self._xyz = torch.empty(0)              # xyz坐标
        self._features_dc = torch.empty(0)      #TODO 直流特征
        self._features_rest = torch.empty(0)    #TODO 其他特征
        self._scaling = torch.empty(0)          # 缩放参数
        self._rotation = torch.empty(0)         # 旋转参数
        self._opacity = torch.empty(0)          # 不透明度参数
        self.max_radii2D = torch.empty(0)       #TODO 2D最大半径
        self.xyz_gradient_accum = torch.empty(0)    # xyz梯度累积
        self.denom = torch.empty(0)             #TODO 归一化因子
        self.optimizer = None                   # 优化器
        self.percent_dense = 0                  # 密度百分比
        self.spatial_lr_scale = 0               # 空间学习率比例
        self.setup_functions()                  # 初始化激活函数和其他一些后面用到的函数

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        # 设置密集度百分比
        self.percent_dense = training_args.percent_dense
        # 初始化xyz梯度累积张量为零，形状与xyz坐标数量一致，部署在CUDA设备上
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # 初始化分母张量为零，形状与xyz坐标数量一致，部署在CUDA设备上
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # 构造一个列表，包含不同参数组及其对应的学习率和名称
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        # 初始化优化器为Adam，传入参数组列表，初始学习率设为0，epsilon为1e-15，以提高数值稳定性
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        # 配置xyz坐标的学习率调度器参数，使用指数衰减函数
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        # 使用inverse_sigmoid函数计算新的不透明度值。将当前的不透明度值与0.01进行比较，取较小的值，
        # 然后应用inverse_sigmoid函数。这一步是为了确保不透明度不会低于一定阈值（接近0但不为0），
        # inverse_sigmoid是sigmoid函数的逆函数，用于将不透明度的sigmoid值转换回其原始的线性范围。
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        
        # 将新计算的不透明度值替换为可优化的张量，这一步通常涉及将新的不透明度张量设置为可由优化器更新的状态
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        
        # 更新类的内部 _opacity 属性
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        # 初始化一个空字典，用于存储被替换的张量
        optimizable_tensors = {}
        
        # 遍历优化器中的所有参数组
        for group in self.optimizer.param_groups:
            # 检查参数组的名称是否与给定的名称匹配
            if group["name"] == name:
                # 获取当前参数的优化状态，如果不存在则返回None
                stored_state = self.optimizer.state.get(group['params'][0], None)
                
                # 初始化优化状态中的梯度平均(exp_avg)和平方梯度平均(exp_avg_sq)为零张量，
                # 它们的形状与新的张量相同
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                # 删除优化器状态中当前参数的旧状态
                del self.optimizer.state[group['params'][0]]
                
                # 将新的张量封装为nn.Parameter，并设置为需要梯度，
                # 然后替换原有参数组中对应的参数
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                
                # 将更新后的状态信息重新关联到新的参数上
                self.optimizer.state[group['params'][0]] = stored_state
                # 将新的参数保存到optimizable_tensors字典中，用其组名作为键
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors  # 返回包含已替换参数的字典

    def _prune_optimizer(self, mask):
        # 初始化一个字典，用于存储更新后的参数张量
        optimizable_tensors = {}
        # 遍历优化器中的所有参数组
        for group in self.optimizer.param_groups:
            # 获取当前参数的优化状态
            stored_state = self.optimizer.state.get(group['params'][0], None)
            # 如果存在优化状态，则对状态进行修剪
            if stored_state is not None:
                # 使用提供的掩码修剪优化器状态中的梯度平均和平方梯度平均
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                # 删除当前参数的旧状态
                del self.optimizer.state[group['params'][0]]
                # 创建新的参数，将原有参数中有效的部分（根据掩码）重新封装为可训练参数
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                # 更新优化器状态，重新关联修剪后的新参数
                self.optimizer.state[group['params'][0]] = stored_state
                # 将更新后的参数存储在返回字典中
                optimizable_tensors[group["name"]] = group["params"][0]
            # 如果不存在优化状态，直接修剪并更新参数
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors # 返回包含所有更新后参数的字典

    def prune_points(self, mask):
        '''
        负责根据提供的掩码修剪（即删除）不再需要的点。这是模型优化的一部分，有助于移除冗余或不符合条件的数据点
        '''
        valid_points_mask = ~mask
        # 调用 _prune_optimizer 方法，传入有效点掩码以修剪优化器中的参数
        optimizable_tensors = self._prune_optimizer(valid_points_mask)
        # 更新模型的主要张量属性为修剪后的张量
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # 修剪与点相关的其他属性张量
        # 梯度累积张量也根据有效点掩码进行修剪
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        # 归一化因子（denom）和最大半径（max_radii2D）张量也按照相同的掩码进行修剪
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        # 初始化一个字典，用来存储更新后的参数张量
        optimizable_tensors = {}
        # 遍历优化器中的所有参数组
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1 # 确保每个参数组中只有一个参数（这是一个设计前提）
            # 从输入字典中获取与当前参数组名称相匹配的张量
            extension_tensor = tensors_dict[group["name"]]
            # 获取当前参数的优化状态
            stored_state = self.optimizer.state.get(group['params'][0], None)
            # 如果存在优化状态，则更新状态并合并张量
            if stored_state is not None:
                # 更新梯度平均和平方梯度平均状态，添加新张量对应的零初始化部分
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                # 删除当前参数的旧状态
                del self.optimizer.state[group['params'][0]]
                # 创建新的参数，将旧参数和新张量拼接，并设置为需要梯度
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                # 重新设置更新后的参数状态
                self.optimizer.state[group['params'][0]] = stored_state
                # 将更新后的参数存储在返回字典中
                optimizable_tensors[group["name"]] = group["params"][0]
            # 如果不存在优化状态（较不常见），直接创建新的参数并更新
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors # 返回包含所有更新后参数的字典

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        # 将新点的所有相关属性组织成一个字典
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}
        # 将新点的数据添加到优化器中，并获取更新后的张量
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        
        # 更新模型的各个属性，以包括新添加的点
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # 重置与这些新点相关的梯度累积和归一化因子
        # 这确保新点被正确地融入模型的训练过程中，没有过时的梯度信息
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # 重置最大2D半径数组，为新点计算新的半径值
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0] # 获取初始点的数量
        # Extract points that satisfy the gradient condition
        # 初始化一个全零的梯度填充张量
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze() # 将实际梯度填充到前部
        # 根据梯度阈值选择点
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        # 进一步筛选这些点，确保它们的缩放值大于场景范围与密度百分比的乘积
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        # 计算选中点的标准差，用于后续生成新点
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda") # 创建均值为0的向量
        
        # 生成新点的坐标，基于正态分布随机偏移
        samples = torch.normal(mean=means, std=stds)
        # 对选中点的旋转参数进行重复，准备应用于新点
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        # 通过旋转和平移变换生成新点的坐标
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        # 计算新点的缩放参数
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        # 重复旧点的旋转参数
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        # 重复旧点的特征
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        # 重复旧点的不透明度
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        # 调用后处理函数整合新点
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
        # 创建修剪滤波器，用于移除被替代的旧点
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        # 从梯度张量中选出符合梯度阈值条件的点
        # torch.norm(grads, dim=-1) 计算每个点的梯度的范数
        # 比较梯度范数是否大于或等于设定的阈值 grad_threshold
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        # 进一步筛选符合缩放限制的点，即缩放因子必须小于或等于场景范围乘以密集百分比（percent_dense）
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        # 使用筛选出的掩码获取符合条件的点的属性
        new_xyz = self._xyz[selected_pts_mask]               # 选出符合条件的坐标
        new_features_dc = self._features_dc[selected_pts_mask]  # 选出符合条件的直流特征
        new_features_rest = self._features_rest[selected_pts_mask]  # 选出符合条件的其他特征
        new_opacities = self._opacity[selected_pts_mask]     # 选出符合条件的不透明度
        new_scaling = self._scaling[selected_pts_mask]       # 选出符合条件的缩放参数
        new_rotation = self._rotation[selected_pts_mask]     # 选出符合条件的旋转参数

        # 调用densification_postfix处理复制后的点
        # 该函数可能涉及将新点添加到模型数据中，进行必要的更新和重新计算
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        # 计算梯度，即xyz坐标的梯度累积除以归一化因子
        grads = self.xyz_gradient_accum / self.denom
        # 修正梯度中的NaN值为0，防止计算错误
        grads[grads.isnan()] = 0.0
        
        # 调用densify_and_clone方法进行密集化和克隆操作，增加密集度
        self.densify_and_clone(grads, max_grad, extent)
        # 调用densify_and_split方法进行密集化和分裂操作，进一步优化密集度
        self.densify_and_split(grads, max_grad, extent)

        # 创建修剪掩码，基于不透明度小于最小不透明度阈值的条件
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        
        # 如果设置了最大屏幕尺寸限制
        if max_screen_size:
            # 计算点在视空间中的最大半径是否超过最大屏幕尺寸
            big_points_vs = self.max_radii2D > max_screen_size
            # 计算点的缩放值是否超过场景范围的10%
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            # 更新修剪掩码，包括超过屏幕尺寸和缩放尺寸的点
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        # 执行修剪操作，移除不需要的高斯点
        self.prune_points(prune_mask)
        # 清空CUDA缓存，优化内存使用
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        # 更新xyz_gradient_accum：增加视空间点张量的梯度范数
        # torch.norm 计算梯度的L2范数，这里只计算前两维（:2），通常对应x和y方向
        # dim=-1 指定在最后一个维度上计算范数，keepdim=True 保持输出维度与输入一致
        # update_filter 是一个布尔掩码，指定哪些点的统计信息需要更新
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        
        # 更新denom（归一化分母）：为相应的点增加1
        # 这是一个简单的计数器，用于后续计算平均梯度或进行其他统计分析
        self.denom[update_filter] += 1