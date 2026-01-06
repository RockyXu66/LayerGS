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

"""
Training utilities for LayerGS.
"""

import os
import uuid
import getpass
import re
from typing import Dict, Tuple
from pathlib import Path
from datetime import datetime
from argparse import Namespace
from typing import Optional
import shutil
from tqdm import tqdm

import pytorch3d
import torchvision
import torch
import cv2
import imageio
from loguru import logger

from layergs.scene.gaussian_model import GaussianModel
from layergs.scene.cameras import HumanGaussianCamera
from gala_utils import helpers


# =============================================================================
# Training Setup Utilities
# =============================================================================

def set_sds(iteration: int, sds_cfg_list: list):
    """
    Parse SDS configuration at specific iterations.
    
    Args:
        iteration: Current training iteration
        sds_cfg_list: List of [iteration, batch_size, weight] triplets
        
    Returns:
        Configuration tuple (iteration, batch_size, weight) or None
    """
    chunk_size = 3
    sds_cfg = [sds_cfg_list[i:i + chunk_size] for i in range(0, len(sds_cfg_list), chunk_size)]
    for cfg in sds_cfg:
        if iteration == int(cfg[0]):
            return cfg
    return None


def prepare_output_and_logger(args):
    """
    Set up output folder and logging.
    
    Args:
        args: Argument namespace with model_path
    """
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    logger.info("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))


def log_system_info():
    """Log username and GPU information."""
    username = getpass.getuser()
    logger.info(f"Username: {username}")
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        logger.info(f"Detected {gpu_count} GPU(s):")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            logger.info(f"  GPU {i}: {gpu_name}")


def setup_wandb(dataset, scene, result_folder: str):
    """
    Initialize Weights & Biases logging.
    
    Args:
        dataset: Dataset configuration
        scene: Scene object with data_cfg
        result_folder: Name of the result folder
        
    Returns:
        wandb run object or None if disabled
    """
    wandb_config = {}
    for k in scene.data_cfg.keys():
        wandb_config[k] = scene.data_cfg[k]
    
    import wandb
    init_kwargs = dict(
        project=f"MLHG-{dataset.model_path.split('/')[-1]}",
        name=f'{result_folder}',
        resume="allow",
        config=wandb_config,
    )
    run = wandb.init(**init_kwargs)

    columns = list(scene.data_cfg.keys())
    values = [str(scene.data_cfg[k]) for k in columns]

    config_table = wandb.Table(columns=columns)
    config_table.add_data(*values)
    wandb.log({'experiment_config': config_table})
    
    return run


def training_report_seg(iteration: int, other_info: dict, dataset):
    """
    Save training images at specified intervals.
    
    Args:
        iteration: Current iteration
        other_info: Dict with 'image', 'gt_image', 'render_path', 'gts_path'
        dataset: Dataset with img_log_interval
    """
    if iteration % dataset.img_log_interval == 0:
        image = other_info['image']
        gt_image = other_info['gt_image']
        render_path = other_info['render_path']
        gts_path = other_info['gts_path']
        
        image_folder = os.path.join(render_path, 'images')
        os.makedirs(image_folder, exist_ok=True)
        
        torchvision.utils.save_image(
            image, 
            os.path.join(image_folder, '{0:05d}'.format(iteration) + ".png")
        )
        if gt_image is not None:
            torchvision.utils.save_image(
                gt_image, 
                os.path.join(gts_path, '{0:05d}'.format(iteration) + ".png")
            )


def setup_training_folders(dataset, cur_version: str, script_name: str):
    """
    Set up training output folders.
    
    Args:
        dataset: Dataset configuration
        cur_version: Version string (empty for overwrite mode)
        script_name: Name of the training script
        
    Returns:
        Tuple of (out_folder, result_folder, render_path, gts_path, overwrite)
    """
    overwrite = True if cur_version == "" else False
    
    if overwrite:
        out_folder = 'point_cloud'
        result_folder = 'train'
        os.makedirs(f'{dataset.model_path}/results/{result_folder}', exist_ok=True)
    else:
        # result_folder = f"{cur_version}_{script_name}"
        result_folder = f"{cur_version}"
        out_folder = f'results/{result_folder}'
        
    if not os.path.exists(f'{dataset.model_path}/results/{result_folder}'):
        logger.error(f'{dataset.model_path}/results/{result_folder} does not exist.')
        import sys
        sys.exit(0)
    
    if overwrite:
        render_path = os.path.join(dataset.model_path, "train", "renders")
    else:
        render_path = os.path.join(dataset.model_path, "results", f"{result_folder}", "renders")
    gts_path = os.path.join(dataset.model_path, "train", "gt")
    
    return out_folder, result_folder, render_path, gts_path, overwrite


def setup_logging(dataset, result_folder: str, cur_version: str, cmd_line: str):
    """
    Set up file logging.
    
    Args:
        dataset: Dataset configuration
        result_folder: Name of the result folder
        cur_version: Version string
        cmd_line: Command line string to log
    """
    log_filename = f'log_{cur_version}.log'
    logger.add(
        f'{dataset.model_path}/results/{result_folder}/{log_filename}', 
        level='INFO', 
        enqueue=True
    )
    logger.info(f'{cmd_line}')
    log_system_info()


def get_smplx_path(dataset) -> str:
    """
    Get the SMPLX file path based on dataset type.
    
    Args:
        dataset: Dataset configuration
        
    Returns:
        Path to SMPLX file
    """
    if dataset.data_type == 'pytorch3d':
        return f'{dataset.source_path}/{dataset.scan_name.replace("_norm", "-smplx")}.pkl'
    elif dataset.data_type == 'colmap':
        return f'{dataset.source_path}/estimated_smplx_no_rotation.pkl'
    else:
        raise ValueError(f"Unknown data_type: {dataset.data_type}")

def create_folder(folder_path: str, clean_before_create: bool = False):
    """Clear and recreate the folder."""
    if clean_before_create and os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path, exist_ok=True)

def format_cmd_line(argv: list) -> str:
    """
    Format command line arguments for logging.
    
    Args:
        argv: sys.argv list
        
    Returns:
        Formatted command line string
    """
    cmd_line = argv[0] + '\n'
    for i in range(1, len(argv[:-1])):
        arg = argv[i]
        if arg.startswith('--') or arg.startswith('-'):
            cmd_line += f'{arg.ljust(30)}'
        else:
            cmd_line += f'{arg}'
        if argv[i+1].startswith('--') or argv[i+1].startswith('-'):
            cmd_line += '\n'
        else:
            cmd_line += '  '
    cmd_line += f'{argv[-1]}\n'
    return cmd_line


# Segmentation labels from Sapiens
# https://github.com/facebookresearch/sapiens/blob/main/docs/SEG_README.md
SAPIENS_SEG_LABELS = {
    'background': 0,
    'lower_clothing': 12,
    'upper_clothing': 22,
}

# 4D-Dress segmentation labels (RGB values)
DRESS4D_SEG_LABELS = {
    'skin': [128, 128, 128],
    'hair': [255, 128, 0],
    'shoe': [128, 0, 255],
    'upper': [180, 50, 50],
    'lower': [50, 180, 50],
    'outer': [0, 128, 255],
}


# =============================================================================
# Gaussian Saving and Rendering Utilities
# =============================================================================

def save_multi_layer_gs(folder_path, gaussians : GaussianModel, gaussians_garment : GaussianModel, sh_degree, iteration, apply_2dgs=False, out_folder='point_cloud'):
    point_cloud_path = os.path.join(folder_path, f"{out_folder}/iteration_{iteration}", "point_cloud_body_cano.ply")
    gaussians.save_ply(point_cloud_path)
    print(f'Save the canonical body 2dgs file in {point_cloud_path}')

    posed_gaussians_body = GaussianModel(sh_degree, apply_2dgs)
    verts = gaussians.get_xyz
    offsets = torch.nn.functional.grid_sample(gaussians.shape_pose_offsets_grid , verts[None, None, None, :, :]).reshape(1, 3, -1).permute(0, 2, 1)
    lbs_weights = torch.nn.functional.grid_sample(gaussians.lbs_weights_grid, verts[None, None, None, :, :]).reshape(1, 55, -1).permute(0, 2, 1)
    verts = verts + offsets[0]  # shape pose blend shapes
    posed_xyz_body = helpers.skinning(verts, lbs_weights, gaussians.gt_smplx_tfs)[0]  # lbs

    w_tf = torch.einsum("bpn,bnij->bpij", lbs_weights, gaussians.gt_smplx_tfs)
    rotations = gaussians.get_rotation#[:, [1,2,3,0]]      # 3dgs use wxyz for quaternion and scipy use xyzw for quaternion
    rot_mat = pytorch3d.transforms.quaternion_to_matrix(rotations)
    new_rot_mat = w_tf[0, :, :3, :3] @ rot_mat
    new_quat = pytorch3d.transforms.matrix_to_quaternion(new_rot_mat)
    posed_rotation_body = new_quat#[:, [3,0,1,2]]
    posed_gaussians_body.create_from_data(posed_xyz_body, gaussians._features_dc, gaussians._features_rest, gaussians._opacity, gaussians._scaling, posed_rotation_body)

    posed_point_cloud_path = os.path.join(folder_path, f"{out_folder}/iteration_{iteration}", "point_cloud_body_posed.ply")
    posed_gaussians_body.save_ply(posed_point_cloud_path)
    print(f'Save the posed body 2dgs file in {posed_point_cloud_path}')




    point_cloud_path = os.path.join(folder_path, f"{out_folder}/iteration_{iteration}", "point_cloud_garment_cano.ply")
    gaussians_garment.save_ply(point_cloud_path)
    print(f'Save the canonical garment 2dgs file in {point_cloud_path}')

    posed_gaussians_garment = GaussianModel(sh_degree, apply_2dgs)
    verts = gaussians_garment.get_xyz
    offsets = torch.nn.functional.grid_sample(gaussians_garment.shape_pose_offsets_grid , verts[None, None, None, :, :]).reshape(1, 3, -1).permute(0, 2, 1)
    lbs_weights = torch.nn.functional.grid_sample(gaussians_garment.lbs_weights_grid, verts[None, None, None, :, :]).reshape(1, 55, -1).permute(0, 2, 1)
    verts = verts + offsets[0]  # shape pose blend shapes
    posed_xyz_garment = helpers.skinning(verts, lbs_weights, gaussians_garment.gt_smplx_tfs)[0]  # lbs

    w_tf = torch.einsum("bpn,bnij->bpij", lbs_weights, gaussians_garment.gt_smplx_tfs)
    rotations = gaussians_garment.get_rotation#[:, [1,2,3,0]]      # 3dgs use wxyz for quaternion and scipy use xyzw for quaternion
    rot_mat = pytorch3d.transforms.quaternion_to_matrix(rotations)
    new_rot_mat = w_tf[0, :, :3, :3] @ rot_mat
    new_quat = pytorch3d.transforms.matrix_to_quaternion(new_rot_mat)
    posed_rotation_garment = new_quat#[:, [3,0,1,2]]
    posed_gaussians_garment.create_from_data(posed_xyz_garment, gaussians_garment._features_dc, gaussians_garment._features_rest, gaussians_garment._opacity, gaussians_garment._scaling, posed_rotation_garment)

    posed_point_cloud_path = os.path.join(folder_path, f"{out_folder}/iteration_{iteration}", "point_cloud_garment_posed.ply")
    posed_gaussians_garment.save_ply(posed_point_cloud_path)
    print(f'Save the posed garment 2dgs file in {posed_point_cloud_path}')


    posed_gaussians_comp = GaussianModel(sh_degree)
    xyz_comp = torch.cat([posed_xyz_body, posed_xyz_garment], dim=0)
    feat_dc_comp = torch.cat([gaussians._features_dc, gaussians_garment._features_dc], dim=0)
    feat_rest_comp = torch.cat([gaussians._features_rest, gaussians_garment._features_rest], dim=0)
    opacity_comp = torch.cat([gaussians._opacity, gaussians_garment._opacity], dim=0)
    scaling_comp = torch.cat([gaussians._scaling, gaussians_garment._scaling], dim=0)
    rotation_comp = torch.cat([posed_rotation_body, posed_rotation_garment], dim=0)
    posed_gaussians_comp.create_from_data(xyz_comp, feat_dc_comp, feat_rest_comp, opacity_comp, scaling_comp, rotation_comp)
    posed_point_cloud_path = os.path.join(folder_path, f"{out_folder}/iteration_{iteration}", "point_cloud_comp_posed.ply")
    posed_gaussians_comp.save_ply(posed_point_cloud_path)
    print(f'Save the posed composed 2dgs file in {posed_point_cloud_path}')


def save_img_sequence(
    filename,
    img_dir,
    matcher,
    save_format="mp4",
    fps=30,
    keep_imgs=False,
) -> str:
    assert save_format in ["gif", "mp4"]
    if not filename.endswith(save_format):
        filename += f".{save_format}"
    save_path = os.path.join(Path(img_dir).parent, filename)
    matcher = re.compile(matcher)
    imgs = []
    for f in os.listdir(img_dir):
        if matcher.search(f):
            imgs.append(f)
    imgs = sorted(imgs, key=lambda f: int(matcher.search(f).groups()[0]))
    imgs = [cv2.imread(os.path.join(img_dir, f)) for f in imgs]

    if save_format == "gif":
        imgs = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in imgs]
        imageio.mimsave(save_path, imgs, fps=fps, palettesize=256)
    elif save_format == "mp4":
        imgs = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in imgs]
        imageio.mimsave(save_path, imgs, fps=fps)
    if not keep_imgs:
        shutil.rmtree(img_dir)
    return save_path

# def render_val_cam(iteration, other_infor, data_iter, gaussians, pipe, bg, cfg, isTest=False, freezed_gaussians=None, inpaint=False, postfix='test'):
def render_val_cam(iteration, other_infor, data_iter, gaussian_list, pipe, bg, cfg, isTest=False, postfix='test', apply_shape=False, apply_pose=False, aux=None):
    # Lazy import to avoid circular dependency
    from layergs.rendering import render_gaussian_list
    
    with torch.no_grad():
        if isTest:
            n_view = cfg.n_test_views
            render_path = os.path.join(other_infor['render_path'], f'it{iteration:05d}',f'it{iteration:05d}-test')
            os.makedirs(render_path, exist_ok=True)
            # progress = tqdm(n_view, 'Rendering test views', bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
        else:
            n_view = cfg.n_val_views
            render_path = other_infor['render_path']
            image_folder = os.path.join(render_path, 'images')
            os.makedirs(image_folder, exist_ok=True)
        for _ in range(n_view):
            batch = next(data_iter)
            id = 0
            c2w = batch['c2w'][id]
            fovy = batch['fovy']
            index = batch['index'][id]
            viewpoint_cam = HumanGaussianCamera(c2w, fovy, height=1024, width=1024)

            render_pkg = render_gaussian_list(viewpoint_cam, gaussian_list, pipe, bg, scaling_modifier = 1.0, override_color = None, \
                                              apply_shape=apply_shape, apply_pose=apply_pose, apply_2dgs=gaussian_list[0].apply_2dgs)

            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            if isTest:
                img_name = f"{index}.png"
                # progress.update(1)
                torchvision.utils.save_image(image, os.path.join(render_path, img_name))
            else:
                img_name = '{0:05d}'.format(iteration) + f"-{postfix}-{index}.png"
                torchvision.utils.save_image(image, os.path.join(image_folder, img_name))
        
        if isTest:
            # Save the video
            vid_path = save_img_sequence(
                f"it{iteration:05d}-{postfix}",
                render_path,
                "(\d+)\.png",
                save_format="mp4",
                fps=30,
            )
            return vid_path

def concat_gaussians(gaussian_list: list[GaussianModel]):
    import torch.nn as nn
    new_gaussian = GaussianModel(sh_degree=gaussian_list[0].max_sh_degree, apply_2dgs=gaussian_list[0].apply_2dgs)

    new_xyz = torch.concat([pc._xyz for pc in gaussian_list])
    new_feat_dc = torch.concat([pc._features_dc for pc in gaussian_list])
    if gaussian_list[0]._features_rest.shape[1] != 0:
        new_feat_rest = torch.concat([pc._features_rest for pc in gaussian_list])
    else:
        new_feat_rest = torch.concat([pc._features_rest for pc in gaussian_list])
    new_opacity = torch.concat([pc._opacity for pc in gaussian_list])
    new_scaling = torch.concat([pc._scaling for pc in gaussian_list])
    new_rotation = torch.concat([pc._rotation for pc in gaussian_list])

    new_gaussian._xyz = nn.Parameter(new_xyz)
    new_gaussian._features_dc = nn.Parameter(new_feat_dc)
    new_gaussian._features_rest = nn.Parameter(new_feat_rest)
    new_gaussian._opacity = nn.Parameter(new_opacity)
    new_gaussian._scaling = nn.Parameter(new_scaling)
    new_gaussian._rotation = nn.Parameter(new_rotation)

    size = new_gaussian.get_xyz.shape[0]
    new_gaussian.max_radii2D = torch.zeros((size), device="cuda")
    new_gaussian.xyz_gradient_accum = torch.zeros((size, 1), device="cuda")
    new_gaussian.denom = torch.zeros((size, 1), device="cuda")
    return new_gaussian

def parse_reward_table(block: str) -> Dict[str, Tuple[float, float]]:
    """
    Given a multi‐line string containing the ImageReward table rows, 
    returns a dict:
      {
        # "GALA w/ scan":       (30.50, 0.313),
        # "GALA w/o scan":      (30.08, 0.175),
        "Ours w/o scan":      (20.88, -1.977),
      }
    """
    pattern = re.compile(
        r"\|\s*(?P<name>.+?)\s*\|\s*"
        r"(?P<clip>[-+]?[0-9]*\.?[0-9]+)\s*\|\s*"
        r"(?P<reward>[-+]?[0-9]*\.?[0-9]+)\s*\|"
    )
    results = {}
    for m in pattern.finditer(block):
        name   = m.group("name")
        clip   = float(m.group("clip"))
        reward = float(m.group("reward"))
        results[name] = (clip, reward)
    return results