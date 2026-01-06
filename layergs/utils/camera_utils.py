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

from layergs.scene.cameras import Camera
import numpy as np
from layergs.utils.general_utils import PILtoTorch
from layergs.utils.graphics_utils import fov2focal
import torch
from pytorch3d.renderer import (
    look_at_view_transform
)
import tqdm
import math

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    seg_image_dict = {}
    for k, v in cam_info.seg_image.items():
        if v is None:
            seg_image = None
        elif k in ['sam_upper', 'sam_lower', 'ori_img', 'seg_label_img']:
            seg_image = PILtoTorch(v, resolution)
        elif k in ['sapiens', 'sapiens_cano', 'sapiens_ori']:
            seg_image = torch.from_numpy(v)
        elif k in ['mask_body_cano']:
            seg_image = PILtoTorch(v, resolution)
        seg_image_dict[k] = seg_image
    if cam_info.mask_image is not None:
        mask_image = PILtoTorch(cam_info.mask_image, resolution)
        loaded_mask = mask_image
    else:
        loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, seg_image=seg_image_dict, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    # sort the camera by image id
    cam_infos = sorted(cam_infos, key=lambda c : c[8])

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def create_rotate_cameras(num_views=18):
    camera_list = []
    cam_range = {
        'h_angle_range': [0, 10, 20],
        'v_angle_range': [0, 365, 360//num_views],
    }
    fovX = np.deg2rad(60)
    fovY = np.deg2rad(60)
    for h_angle in range(cam_range['h_angle_range'][0], cam_range['h_angle_range'][1], cam_range['h_angle_range'][2]):
        for v_angle in range(cam_range['v_angle_range'][0], cam_range['v_angle_range'][1], cam_range['v_angle_range'][2]):

            R, T = look_at_view_transform(2.0, h_angle, v_angle)
            R, T = R[0], T[0]
            R, T = R.permute(1, 0), T.T         # convert row-major matrix (used in pytorch3d) to column-major matrix (colmap / opencv)
            w2c = torch.eye(4)                  # this 'w2c' is the world-to-camera matrix in torch3d (but column-major)
            w2c[:3, :3] = R
            w2c[:3, 3] = T
            new_c2w = torch.linalg.inv(w2c)     # this is camera-to-world matrix in torch3d

            # from pytorch3d's left-up-forward to colmap's right-down-forward
            # get colmap rotation matrix
            R = new_c2w[:3, :3]                 # this is rotation matrix for camera-to-world in torch3d
            T = new_c2w[:3, 3]
            R = torch.stack([-R[:, 0], -R[:, 1], R[:, 2]], 1)   # from LUF to RDF for Rotation (this is rotation matrix for camera-to-world in colmap)
            c2w = torch.eye(4)                  # this is camera-to-world matrix in colmap
            c2w[:3, :3] = R
            c2w[:3, 3] = T
            colmap_w2c = torch.inverse(c2w)     # this is world-to-camera matrix in colmap
            R = colmap_w2c[:3, :3].numpy()
            T = colmap_w2c[:3, 3].numpy()
            img = torch.zeros((3, 1024, 1024))
            cam = Camera(
                colmap_id=0,
                R=R,
                T=T,
                FoVx=fovX,
                FoVy=fovY,
                image=img,
                seg_image=None, gt_alpha_mask=None, image_name='', uid=0
            )
            camera_list.append(cam)
    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry

def create_cam(old_cam: Camera, R, T, image):
    return Camera(colmap_id=0, R=R, T=T, 
                  FoVx=old_cam.FoVx, FoVy=old_cam.FoVy, 
                  image=image, seg_image=None, gt_alpha_mask=None,
                  image_name='', uid=id, data_device=old_cam.data_device)
