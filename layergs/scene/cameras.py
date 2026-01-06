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
from torch import nn

from layergs.utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov, fov2focal

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, seg_image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")        

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        self.seg_image = seg_image


        if gt_alpha_mask is not None:
            self.gt_alpha_mask = gt_alpha_mask.to(self.data_device)
            # self.original_image *= gt_alpha_mask.to(self.data_device)     # TODO: adapt this for customized monocular video
        else:
            if self.seg_image is not None:
                self.gt_alpha_mask = torch.zeros_like(self.seg_image['sapiens']).to(self.data_device)
                self.gt_alpha_mask[torch.where(self.seg_image['sapiens'] != 0)] = 1
                self.gt_alpha_mask = self.gt_alpha_mask[None]
            self.original_image *= torch.ones((1, self.image_height, self.image_width)).to(self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

class HumanGaussianCamera(nn.Module):
    def __init__(self, c2w, FoVy, height, width,
                 trans=torch.tensor([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(HumanGaussianCamera, self).__init__()
        FoVx = focal2fov(fov2focal(FoVy, height), width)
        # FoVx = focal2fov(fov2focal(FoVy, width), height)

        w2c = torch.inverse(c2w)

        # rectify...
        w2c[1:3, :3] *= -1
        w2c[:3, 3] *= -1

        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_height =height
        self.image_width = width

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")


        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans.float()
        self.scale = scale

        self.world_view_transform = w2c.transpose(0, 1).float().cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).float().cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0).float()
        self.camera_center = self.world_view_transform.inverse()[3, :3].float()
        # print('self.camera_center',self.camera_center)

