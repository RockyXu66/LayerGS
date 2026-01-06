#
# LayerGS Rendering Utilities
#
# Utilities for rendering Gaussian models during training and evaluation.
#

import os
import shutil
from os import makedirs
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from tqdm import tqdm
from loguru import logger

from layergs.rendering import render_gaussian_list
from layergs.utils.train_utils import save_img_sequence, create_folder


def render_set_list(
    model_path: str,
    name: str,
    res_type: str,
    views: list,
    gaussians_list: list,
    pipeline,
    background: torch.Tensor,
    save_video: bool = False,
    aux=None,
    apply_shape: bool = False,
    apply_pose: bool = False,
    save_mask: bool = True,
    save_vid: bool = False,
    vid_fps: int = 30,
    save_transparent_bg: bool = False,
):
    """
    Render a set of views with given Gaussian models.
    
    Args:
        model_path: Base path for saving outputs
        name: Name prefix for the output folder
        res_type: Result type suffix for folder naming
        views: List of camera views to render
        gaussians_list: List of GaussianModel instances to render
        pipeline: Pipeline parameters
        background: Background color tensor
        save_video: Whether to save video (alias for save_vid)
        aux: Auxiliary data for visibility masking (optional)
        apply_shape: Whether to apply shape transformations
        apply_pose: Whether to apply pose transformations
        save_mask: Whether to save depth masks
        save_vid: Whether to save video
        vid_fps: Video frames per second
        save_transparent_bg: Whether to save with transparent background
    """
    if save_mask:
        mask_name = f'{name}-{res_type}-mask'
        mask_render_path = os.path.join(model_path, mask_name)
        create_folder(mask_render_path, clean_before_create=True)

    name = f'{name}-{res_type}'
    render_path = os.path.join(model_path, name)
    create_folder(render_path, clean_before_create=True)
    logger.info(f'Save the results in {render_path}')

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # Optional visibility masking
        # if aux is not None:
        #     from utils.postprocess_utils import mask_gaussians_by_visibility
        #     mask_gaussians_by_visibility(view.camera_center, aux)
        #     gaussians_list = [item['gaussians'] for item in aux]

        rendering = render_gaussian_list(
            view, gaussians_list, pipeline, background,
            scaling_modifier=1.0, override_color=None,
            apply_shape=apply_shape, apply_pose=apply_pose,
            apply_2dgs=gaussians_list[0].apply_2dgs
        )
        rgb_img = rendering['render']

        if save_mask:
            depth_img = rendering['depth_median']
            depth_img = depth_img.squeeze(0)
            depth_np = depth_img.cpu().numpy()
            mask_img = np.zeros_like(depth_np, dtype=np.uint8)
            mask_img[np.where(depth_np != 0)] = 255
            img = Image.fromarray(mask_img, mode='L')
            img.save(os.path.join(mask_render_path, f'{idx:08d}.png'))

        if save_transparent_bg:
            depth_img = rendering['depth_median'].squeeze(0)
            fg_mask = (depth_img != 0).float()
            rgba = torch.ones(
                (4, rgb_img.shape[1], rgb_img.shape[2]),
                dtype=rgb_img.dtype, device=rgb_img.device
            )
            rgba[:3] = rgb_img * fg_mask.unsqueeze(0)
            rgba[3] = fg_mask
            torchvision.utils.save_image(rgba, os.path.join(render_path, f'{idx:08d}.png'))
        else:
            torchvision.utils.save_image(rgb_img, os.path.join(render_path, f'{idx:08d}.png'))

    if save_vid or save_video:
        vid_path = save_img_sequence(
            Path(name).stem,
            render_path,
            r"(\d+)\.png",
            save_format="mp4",
            fps=vid_fps,
            keep_imgs=True,
        )
        logger.info(f'Save the video in {vid_path}')

    return render_path

def focus_point_fn(poses: np.ndarray) -> np.ndarray:
  """Calculate nearest point to all focal axes in poses."""
  directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
  m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
  mt_m = np.transpose(m, [0, 2, 1]) @ m
  focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
  return focus_pt