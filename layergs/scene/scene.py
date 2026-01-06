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
import numpy as np
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader

from layergs.utils.system_utils import searchForMaxIteration
from layergs.utils.camera_utils import cameraList_from_camInfos, camera_to_JSON, create_cam, create_rotate_cameras
from layergs.data.dataset_readers import sceneLoadTypeCallbacks
from layergs.scene.gaussian_model import GaussianModel
from layergs.data.humangaussian_dataset import (
    ExperimentConfig, load_config, parse_structured,
    RandomCameraDataModuleConfig, RandomCameraIterableDataset, RandomCameraDataset
)
from arguments import ModelParams

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], cfg_file=None, rot_num_views=18):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        self.use_opengl_camera = True   # whehter to use opengl camera or blender camera

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.rot_cameras = {}

        rotation = hasattr(args, 'rotation')
        if rotation:
            rotation = args.rotation

        use_original_colmap = getattr(args, 'use_original_colmap', False)
        sparse_folder = getattr(args, 'sparse_folder', 'sparse')

        if args.data_type == 'pytorch3d':
            scene_info = sceneLoadTypeCallbacks["Pytorch3d"](args.source_path, args.images, args.eval, args.scan_name, rotation=rotation)
        elif args.data_type == 'colmap' and os.path.exists(os.path.join(args.source_path, sparse_folder)):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, rotation=rotation, use_original_colmap=use_original_colmap, sparse_folder=sparse_folder)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, f"Could not recognize scene type `{args.data_type}`!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            if scene_info.rot_cameras:
                camlist.extend(scene_info.rot_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            print("Loading Rotation Cameras")
            self.rot_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.rot_cameras, resolution_scale, args)
        
        # self.rotate_cameras = create_rotate_cameras(num_views=rot_num_views)
        self.rotate_cameras = create_rotate_cameras(num_views=90)


        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)


        # Use the code from HumanGaussian here
        extras = ["system.prompt_processor.prompt='A man wearing a white tank top and a gray trunks'"]
        n_gpus = 1
        if cfg_file is not None:
            config_file = cfg_file    # this is for the inpainting
        else:
            config_file = 'config/inpaint.yaml'    # this is for the inpainting

        cfg: ExperimentConfig
        cfg = load_config(config_file, cli_args=extras, n_gpus=n_gpus)
        random.seed(cfg.seed)
        self.data_cfg = parse_structured(RandomCameraDataModuleConfig, cfg.data)
        self.human_gaussian_cameras_dataset = RandomCameraIterableDataset(self.data_cfg, self.use_opengl_camera)
        self.human_gaussian_cameras_dataloader = DataLoader(
            self.human_gaussian_cameras_dataset,
            num_workers=0,  # type: ignore
            batch_size=self.data_cfg.batch_size,
            collate_fn=self.human_gaussian_cameras_dataset.collate,
        )

        self.system_cfg = cfg.system

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
    def getHumanGaussianCameras(self):
        return self.human_gaussian_cameras_dataloader

    def getHumanGaussianCameras_with_bs(self, bs=8):
        self.data_cfg.batch_size = bs
        self.human_gaussian_cameras_dataset = RandomCameraIterableDataset(self.data_cfg, self.use_opengl_camera)
        self.human_gaussian_cameras_dataloader = DataLoader(
            self.human_gaussian_cameras_dataset,
            num_workers=0,  # type: ignore
            batch_size=self.data_cfg.batch_size,
            collate_fn=self.human_gaussian_cameras_dataset.collate,
        )
        return self.human_gaussian_cameras_dataloader

    def getHumanGaussianCamerasValTest(self, data_type):
        dataset = RandomCameraDataset(self.data_cfg, data_type, self.use_opengl_camera)
        return DataLoader(
            dataset,
            num_workers=0,
            batch_size=1,
            collate_fn=dataset.collate,
        )
    
    def getRotationCameras(self, scale=1.0):
        return self.rot_cameras[scale]
    
    def getPureRotationCameras(self, scale=1.0):
        return self.rotate_cameras



    def getCustomCameras(self, image_size=1024, yaw=180):
        old_cam = self.train_cameras[1.0][0]
        R = np.eye(3)
        R = Rotation.from_euler('z', angles=180, degrees=True).as_matrix() @ R
        R = Rotation.from_euler('y', angles=yaw, degrees=True).as_matrix() @ R  # front
        # R = Rotation.from_euler('y', angles=0, degrees=True).as_matrix() @ R  # back
        T = np.array([0, -0.4, 2])

        # R = old_cam.R
        # T = old_cam.T
        import torch
        # if old_cam.image_width != image_size
        #     old_cam.
        image = torch.ones((3, image_size, image_size))
        custom_cam = create_cam(old_cam, R, T, image)
        return [custom_cam]