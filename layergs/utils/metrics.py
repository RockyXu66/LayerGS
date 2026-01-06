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

from pathlib import Path
import os
import json
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from tqdm import tqdm
from argparse import ArgumentParser
from loguru import logger

from layergs.utils.loss_utils import ssim
from layergs.utils.image_utils import psnr
from lpipsPyTorch import lpips

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def evaluate(gt_dir, renders_dir):

    gt_dir = Path(gt_dir)
    renders_dir = Path(renders_dir)

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    logger.info("")

    scene_dir = gt_dir.parent
    # try:
    logger.info("Scene:", scene_dir)
    full_dict[scene_dir] = {}
    per_view_dict[scene_dir] = {}
    full_dict_polytopeonly[scene_dir] = {}
    per_view_dict_polytopeonly[scene_dir] = {}



    full_dict[scene_dir] = {}
    per_view_dict[scene_dir] = {}
    full_dict_polytopeonly[scene_dir] = {}
    per_view_dict_polytopeonly[scene_dir] = {}

    renders, gts, image_names = readImages(renders_dir, gt_dir)

    ssims = []
    psnrs = []
    lpipss = []

    for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
        ssims.append(ssim(renders[idx], gts[idx]))
        psnrs.append(psnr(renders[idx], gts[idx]))
        lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))

    logger.info("  SSIM : {:>12.4f}".format(torch.tensor(ssims).mean()))
    logger.info("  PSNR : {:>12.2f}".format(torch.tensor(psnrs).mean()))
    logger.info("  LPIPS: {:>12.4f}".format(torch.tensor(lpipss).mean()))
    logger.info("")

    full_dict[scene_dir].update({"SSIM": torch.tensor(ssims).mean().item(),
                                            "PSNR": torch.tensor(psnrs).mean().item(),
                                            "LPIPS": torch.tensor(lpipss).mean().item()})
    per_view_dict[scene_dir].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

    with open(scene_dir / "results.json", 'w') as fp:
        json.dump(full_dict[scene_dir], fp, indent=True)
    with open(scene_dir / "per_view.json", 'w') as fp:
        json.dump(per_view_dict[scene_dir], fp, indent=True)
    return {
        'ssim': torch.tensor(ssims).mean(),
        'psnr': torch.tensor(psnrs).mean(),
        'lpips': torch.tensor(lpipss).mean(),
    }

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser()
    parser.add_argument('--gt_dir', required=True, type=str, default='')
    parser.add_argument('--renders_dir', required=True, type=str, default='')
    args = parser.parse_args()
    evaluate(args.gt_dir, args.renders_dir)
