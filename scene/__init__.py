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
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene: # 定义一个类Scene，用来表示和处理3D渲染场景

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
 
        初始化Scene类
        :param args: 包含模型路径等配置的参数对象
        :param gaussians: 一个高斯模型对象
        :param load_iteration: 指定加载模型的迭代次数, 默认为None
        :param shuffle: 是否打乱数据, 默认为True
        :param resolution_scales: 分辨率比例列表，默认为[1.0]
        """
        self.model_path = args.model_path # 设置模型路径
        self.loaded_iter = None # 初始化加载的迭代次数为None
        self.gaussians = gaussians  # 设置高斯模型

        if load_iteration:  # 如果指定了加载迭代次数, 默认None, 跳过
            if load_iteration == -1: # 如果加载迭代次数为-1，搜索模型路径下最大迭代次数
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:   # 否则直接使用指定的迭代次数
                self.loaded_iter = load_iteration
            # 打印加载模型的迭代次数
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        # 初始化训练和测试用的相机字典
        self.train_cameras = {}
        self.test_cameras = {}

        # 检查场景源路径下是否存在'sparse'目录
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            # 如果存在，则使用 Colmap 数据集设置，并调用相应的加载函数
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        # 检查场景源路径下是否存在'transforms_train.json'文件
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            # 如果存在，则使用 Blender 数据集设置，并调用相应的加载函数
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else: # 如果两者都不存在，则断言错误，场景类型无法识别
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter: # 检查是否已加载场景的某个迭代
            # 将PLY文件从源路径复制到模型路径
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            # 初始化相机信息列表
            json_cams = []
            camlist = []
            # 如果场景信息中包含测试相机，则添加到相机列表
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            # 如果场景信息中包含训练相机，则添加到相机列表
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
                
            # 将每个相机的信息转换为JSON格式并保存
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            # 将相机信息写入JSON文件
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)
        # 如果设置为打乱数据
        if shuffle:
            # 打乱训练和测试相机列表
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling
        # 从场景信息中获取并设置场景范围
        self.cameras_extent = scene_info.nerf_normalization["radius"]
        # 根据不同的分辨率比例加载相机信息
        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            # 加载并设置训练相机
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            # 加载并设置测试相机
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
        # 如果已加载场景的某个迭代
        if self.loaded_iter:
            # 从对应的迭代文件夹加载高斯模型的PLY文件
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            # 根据点云数据和场景范围创建高斯模型
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]