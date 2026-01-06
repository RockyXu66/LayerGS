#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#

import os
import sys
import torch
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from loguru import logger
from random import randint
import copy

from layergs.rendering import network_gui
from layergs.utils.general_utils import safe_state
from layergs.utils.train_utils import (
    setup_training_folders, setup_logging, setup_wandb, 
    prepare_output_and_logger, training_report_seg, 
    create_folder, save_multi_layer_gs,
    set_sds
)
from layergs.utils.metrics import evaluate

class BaseTrainer:
    def __init__(self, dataset, opt, pipe, args, script_name):
        self.dataset = dataset
        self.opt = opt
        self.pipe = pipe
        self.args = args
        self.script_name = script_name
        
        self.saving_iterations = args.save_iterations
        self.checkpoint_iterations = args.checkpoint_iterations
        self.checkpoint = args.start_checkpoint
        self.debug_from = args.debug_from
        self.config_file = args.config_file
        self.cmd_line = getattr(args, 'cmd_line', " ".join(sys.argv))
        
        self.iteration = 0
        self.start_time = None
        self.viewpoint_stack = None
        self.ema_loss_for_log = 0.0
        self.ema_dist_for_log = 0.0
        self.ema_normal_for_log = 0.0
        self.progress_bar = None
        self.run = None # WandB run
        
        # Scene related
        self.gaussians = None # Main gaussian model
        self.scene = None
        self.scene_eval = None
        self.background = None
        
        self.out_folder = None
        self.result_folder = None
        self.render_path = None
        self.gts_path = None
        self.overwrite = False
        
        # Data loaders
        self.human_gaussian_dataloader = None
        self.human_gaussian_data_iter = None
        self.human_gaussian_val_dataloader = None
        self.human_gaussian_test_dataloader = None
        
        # SDS loss weight (updated in pre_iteration_hook)
        self.w_sds_loss = 0.0

    def setup(self):
        prepare_output_and_logger(self.dataset)
        cur_version = self.dataset.cur_version
        
        # Setup folders
        self.out_folder, self.result_folder, self.render_path, self.gts_path, self.overwrite = setup_training_folders(
            self.dataset, cur_version, self.script_name
        )
        
        # Setup logging
        setup_logging(self.dataset, self.result_folder, cur_version, self.cmd_line)
        
        self.start_time = datetime.now()
        
        # Initialize Scene & Gaussians
        self.init_scene()
        
        # Setup WandB
        self.run = None if self.overwrite else setup_wandb(self.dataset, self.scene, self.result_folder)
        
        # Clear render folders
        create_folder(self.render_path, clean_before_create=True)
        create_folder(self.gts_path, clean_before_create=True)
        
        # Training setup for gaussians
        self.setup_optimization()

        # Background color
        bg_color = [1, 1, 1] if self.dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        # Setup DataLoaders
        self.setup_dataloaders()

    def init_scene(self):
        raise NotImplementedError("Subclasses must implement init_scene")

    def setup_optimization(self):
        self.gaussians.training_setup(self.opt)

    def setup_dataloaders(self):
        self.human_gaussian_val_dataloader = self.scene.getHumanGaussianCamerasValTest('val')
        self.human_gaussian_test_dataloader = self.scene.getHumanGaussianCamerasValTest('test')
        
        self.human_gaussian_dataloader = self.scene.getHumanGaussianCameras()
        self.human_gaussian_data_iter = iter(self.human_gaussian_dataloader)

    def train(self):
        self.setup()
        
        first_iter = self.iteration + 1
        self.progress_bar = tqdm(range(first_iter, self.opt.iterations + 1), desc="Training progress")
        
        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)

        logger.info(f'Overwrite = {self.overwrite}. Save the point cloud in {self.out_folder}.')

        for iteration in range(first_iter, self.opt.iterations + 1):
            self.iteration = iteration
            
            self.pre_iteration_hook(iteration)

            # Network GUI
            if network_gui.conn is None:
                network_gui.try_connect()
            while network_gui.conn is not None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, self.pipe.convert_SHs_python, self.pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                    if custom_cam is not None:
                        net_image = self.render_for_gui(custom_cam, scaling_modifer)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    network_gui.send(net_image_bytes, self.dataset.source_path)
                    if do_training and ((iteration < int(self.opt.iterations)) or not keep_alive):
                        break
                except Exception:
                    network_gui.conn = None

            iter_start.record()
            
            # Update LR
            self.update_learning_rate(iteration)

            # Get Camera
            if not self.viewpoint_stack:
                self.viewpoint_stack = self.scene.getTrainCameras().copy()
            viewpoint_cam = self.viewpoint_stack.pop(randint(0, len(self.viewpoint_stack) - 1))

            if (iteration - 1) == self.debug_from:
                self.pipe.debug = True
            
            # Get random background
            bg = torch.rand((3), device="cuda") if self.opt.random_background else self.background

            # Train Step (Subclass)
            loss, log_dict = self.train_step(iteration, viewpoint_cam, bg)
            
            iter_end.record()

            with torch.no_grad():
                # Logging & Progress
                self.log_progress(loss, log_dict)
                
                # Save/Eval
                self.save_and_evaluate(iteration, log_dict.get('other_info', {}))
                
                # Densification (if handled in subclass, this might be empty or call subclass method)
                self.densification(iteration)
                
            if iteration % 300 == 0:
                elapsed_time = datetime.now() - self.start_time
                logger.info(f"(Iter {iteration}) Elapsed time: {elapsed_time}.")

        if self.run is not None:
            self.run.finish()
            
    def pre_iteration_hook(self, iteration):
        # Update SDS configuration and dataloader if needed
        sds_cfg = set_sds(iteration, self.scene.data_cfg.sds_cfg_list)
        if sds_cfg is not None:
            sds_bs = int(sds_cfg[1])
            self.w_sds_loss = sds_cfg[2]
            logger.info(f'(Iter {iteration}) Set SDS batch_size to {sds_bs}. Set w_sds_loss to {self.w_sds_loss}')
            self.human_gaussian_dataloader = self.scene.getHumanGaussianCameras_with_bs(bs=sds_bs)
            self.human_gaussian_data_iter = iter(self.human_gaussian_dataloader)

    def render_for_gui(self, custom_cam, scaling_modifer):
        # Default implementation
        from layergs.rendering import render
        return render(custom_cam, self.gaussians, self.pipe, self.background, scaling_modifer)["render"]

    def update_learning_rate(self, iteration):
        self.gaussians.update_learning_rate(iteration)

    def train_step(self, iteration, viewpoint_cam, bg):
        raise NotImplementedError

    def densification(self, iteration):
        pass

    def log_progress(self, loss, log_dict):
        # EMA updates
        self.ema_loss_for_log = 0.4 * loss.item() + 0.6 * self.ema_loss_for_log
        
        if 'dist_loss' in log_dict:
            self.ema_dist_for_log = 0.4 * log_dict['dist_loss'].item() + 0.6 * self.ema_dist_for_log
        if 'normal_loss' in log_dict:
            self.ema_normal_for_log = 0.4 * log_dict['normal_loss'].item() + 0.6 * self.ema_normal_for_log

        if self.iteration % 10 == 0:
            postfix = {
                "Loss": f"{self.ema_loss_for_log:.5f}",
                "distort": f"{self.ema_dist_for_log:.5f}",
                "normal": f"{self.ema_normal_for_log:.5f}",
                "#gau": f"{self.gaussians.get_xyz.shape[0]}"
            }
            # Allow subclasses to add extra info
            if 'extra_log' in log_dict:
                postfix.update(log_dict['extra_log'])
                
            self.progress_bar.set_postfix(postfix)
            self.progress_bar.update(10)
            
            # WandB log
            if self.run is not None:
                wandb_log = {
                    'loss': self.ema_loss_for_log,
                    'distort': self.ema_dist_for_log,
                    'normal': self.ema_normal_for_log,
                }
                if 'extra_wandb' in log_dict:
                    wandb_log.update(log_dict['extra_wandb'])
                else:
                    wandb_log['#gau-comp'] = self.gaussians.get_xyz.shape[0]
                
                import wandb
                wandb.log(wandb_log, step=self.iteration)
        
        if self.iteration == self.opt.iterations:
            self.progress_bar.close()

    def save_and_evaluate(self, iteration, other_info):
        # Log images
        training_report_seg(iteration, other_info, self.dataset)
        
        # Checkpointing
        if (iteration in self.checkpoint_iterations):
            logger.info(f"\n[ITER {iteration}] Saving Checkpoint")
            torch.save((self.gaussians.capture(), iteration), self.scene.model_path + "/chkpnt" + str(iteration) + ".pth")
        
        # Helper to save
        if iteration == 1 or (iteration in self.saving_iterations):
             self.save_model(iteration)

        # Render validation videos
        self.rendering_results(iteration, other_info)

        # Evaluation
        self.evaluation(iteration)

    def save_model(self, iteration):
        save_multi_layer_gs(
            self.scene.model_path, self.gaussians, self.gaussians, self.dataset.sh_degree, 
            iteration, apply_2dgs=True, out_folder=self.out_folder
        )

    def rendering_results(self, iteration, other_info):
        pass

    def evaluation(self, iteration):
        pass
