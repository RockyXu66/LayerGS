#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#

import sys
import copy
import torch
import trimesh
import numpy as np
from argparse import ArgumentParser
from loguru import logger

from layergs.scene import Scene, GaussianModel
from layergs.scene.cameras import HumanGaussianCamera
from layergs.rendering import render_multi_layer_2dgs, render_gaussian_list, render, network_gui
from layergs.utils.loss_utils import l1_loss, ssim
from layergs.utils.general_utils import safe_state
from layergs.utils.train_utils import (
    render_val_cam,
    save_multi_layer_gs,
    set_sds,
    format_cmd_line,
    get_smplx_path,
    SAPIENS_SEG_LABELS,
    DRESS4D_SEG_LABELS
)
from layergs.rendering.render_utils import render_set_list
from layergs.utils.metrics import evaluate
from layergs.utils.gaussian_factory import (
    create_gaussians_from_mesh,
    create_gaussians_from_ply,
    load_pretrained_inner_gaussians,
)
from arguments import ModelParams, PipelineParams, OptimizationParams
from layergs.train.base import BaseTrainer
from layergs.guidance import sds_controlnet

class OuterLayerTrainer(BaseTrainer):
    def __init__(self, dataset, opt, pipe, args):
        super().__init__(dataset, opt, pipe, args, f"train_outer")
        self.gaussians_inner = None
        self.gaussians_outer = None
        self.gaussians_garment_dum = None
        self.sds_controlnet = None
        self.pretrained_garment_mesh = None
        
        self.viewspace_point_tensor_list_cano_body = []
        self.sds_loss = torch.tensor(0.0)
        self.ema_dist_garment_dum_for_log = 0.0
        
        self.dilate_seg = None
        self.garment_type = None
        self.seg_garment_name = None

    def init_scene(self):
        # Initialize Gaussians
        self.gaussians_inner = GaussianModel(self.dataset.sh_degree, apply_2dgs=True)
        self.gaussians_outer = GaussianModel(self.dataset.sh_degree, apply_2dgs=True)
        self.gaussians = self.gaussians_outer # Main optimization target
        
        self.scene = Scene(self.dataset, self.gaussians_outer, cfg_file=self.config_file)
        
        # Evaluation scene
        if self.scene.data_cfg.apply_evaluate and self.dataset.data_type != 'colmap':
            eval_dataset = copy.deepcopy(self.dataset)
            eval_dataset.eval = True
            self.scene_eval = Scene(eval_dataset, self.gaussians_inner, cfg_file=self.config_file)

        # Initialize GALA SDS ControlNet
        smplx_file_path = get_smplx_path(self.dataset)
        self.sds_controlnet = sds_controlnet(
            self.scene.use_opengl_camera, 
            smplx_file_path, 
            dataset_type=self.dataset.data_type, 
            prompt_type="outer", 
            data_cfg=self.scene.data_cfg
        )

        # Load pretrained inner body (frozen)
        pretrained_inner_folder = f'{self.dataset.model_path}/{self.dataset.pretrained_inner_name}'
        self.gaussians_inner = load_pretrained_inner_gaussians(
            pretrained_inner_folder, self.dataset.sh_degree, self.sds_controlnet, object_idx=6
        )

        # Initialize outer garment from mesh
        pretrained_name = self.dataset.pretrained_name
        mesh_name = self.dataset.mesh_name
        mesh_path = f'{self.dataset.model_path}/{pretrained_name}/mesh/{mesh_name}'
        ply_path = f'{self.dataset.model_path}/{pretrained_name}/point_cloud_body_cano.ply'
        
        if self.scene.data_cfg.from_ply == 1:
            self.gaussians_outer = create_gaussians_from_ply(mesh_path, ply_path, self.dataset.sh_degree, method=1, object_idx=3)
        elif self.scene.data_cfg.from_ply == 2:
            self.gaussians_outer = create_gaussians_from_ply(mesh_path, ply_path, self.dataset.sh_degree, method=2, object_idx=3)
        elif self.scene.data_cfg.from_ply == 3:
            # Initialize from mesh surface
            self.pretrained_garment_mesh = trimesh.load(mesh_path)
            sampled_points, _ = trimesh.sample.sample_surface(self.pretrained_garment_mesh, self.scene.data_cfg.inner_init_sample_num)
            pretrained_garment_vertices = torch.from_numpy(sampled_points).float().cuda()
            self.gaussians_outer.create_from_xyz(pretrained_garment_vertices, object_idx=3)

        self.gaussians_outer.shape_pose_offsets_grid = self.sds_controlnet.shape_pose_offsets_grid
        self.gaussians_outer.lbs_weights_grid = self.sds_controlnet.lbs_weights_grid
        self.gaussians_outer.gt_smplx_tfs = self.sds_controlnet.gt_smplx_tfs

        # Create dummy garment for distance regularization
        if self.scene.data_cfg.apply_dum_dist:
            self.gaussians_garment_dum = create_gaussians_from_mesh(
                mesh_path, self.scene.data_cfg.inner_dum_sample_num, self.dataset.sh_degree, object_idx=3
            )
            self.gaussians_garment_dum.shape_pose_offsets_grid = self.sds_controlnet.shape_pose_offsets_grid
            self.gaussians_garment_dum.lbs_weights_grid = self.sds_controlnet.lbs_weights_grid
            self.gaussians_garment_dum.gt_smplx_tfs = self.sds_controlnet.gt_smplx_tfs

        # Keep reference to original mesh for distance thresholding
        if self.pretrained_garment_mesh is None:
            self.pretrained_garment_mesh = trimesh.load(mesh_path)

        # Segmentation helpers
        dilation_size = self.scene.data_cfg.seg_dilation_size
        self.dilate_seg = torch.nn.MaxPool2d(kernel_size=dilation_size, stride=1, padding=dilation_size // 2)
        self.garment_type = self.scene.data_cfg.inpaint_garment_type
        self.seg_garment_name = f'{self.garment_type}_clothing'

    def setup_optimization(self):
        # Gaussians inner is frozen in this stage
        self.gaussians_outer.training_setup(self.opt)


    def update_learning_rate(self, iteration):
        self.gaussians_outer.update_learning_rate(iteration)

    def render_for_gui(self, custom_cam, scaling_modifer):
        # Use inner (which is frozen) for GUI? Or outer?
        # Original code: render(custom_cam, gaussians_inner, ...)
        # This implies the GUI visualizes the inner body? That's weird for Outer training.
        # But I will follow the original code:
        # net_image = render(custom_cam, gaussians_inner, pipe, background, scaling_modifer)["render"]
        return render(custom_cam, self.gaussians_inner, self.pipe, self.background, scaling_modifer)["render"]

    def train_step(self, iteration, viewpoint_cam, bg):
        use_op_control_body = True
        apply_op_control_start_iteration = self.scene.data_cfg.apply_op_control_start_iteration
        apply_op_control_end_iteration = self.scene.data_cfg.apply_op_control_end_iteration
        
        self.viewspace_point_tensor_list_cano_body = []
        self.sds_loss = torch.tensor(0.0)

        # Pose handling
        if torch.rand(1).item() < self.scene.data_cfg.apose_prob:
            self.sds_controlnet.points3D = self.sds_controlnet.apose_points3D.copy()
            self.gaussians_inner.apply_sds_pose = False
            self.gaussians_outer.apply_sds_pose = False
        else:
            self.gaussians_inner.apply_sds_pose = True
            self.gaussians_outer.apply_sds_pose = True
            self.gaussians_inner.sds_smplx_tfs = self.sds_controlnet.sds_smplx_tfs.clone()
            self.gaussians_outer.sds_smplx_tfs = self.sds_controlnet.sds_smplx_tfs.clone()
            self.sds_controlnet.points3D = self.sds_controlnet.sds_points3D.copy()

        # SDS loss computation (both layers)
        if use_op_control_body and apply_op_control_start_iteration < iteration < apply_op_control_end_iteration:
            images = []
            batch = next(self.human_gaussian_data_iter)
            if iteration in [1, 3000]:
                logger.info(f"sds batch size: {batch['c2w'].shape[0]}")
            for id in range(batch['c2w'].shape[0]):
                c2w = batch['c2w'][id]
                fovy = batch['fovy'][id]
                sds_cam = HumanGaussianCamera(c2w, fovy, height=1024, width=1024)

                random_bg = torch.rand((3), device="cuda")
                render_pkg = render_gaussian_list(sds_cam, [self.gaussians_inner, self.gaussians_outer], self.pipe, random_bg, apply_shape=True, apply_2dgs=True)
                image = render_pkg["render"]
                viewspace_point_tensor = render_pkg["viewspace_points"]

                images.append(image)
                self.viewspace_point_tensor_list_cano_body.append(viewspace_point_tensor)

            images = torch.stack(images, dim=0)
            mvp = batch['mvp_mtx'].detach().cpu().numpy()
            azimuth = batch['azimuth']
            self.sds_loss = self.sds_controlnet.compute_sds_loss(images.shape[2], images, mvp, azimuth)

        self.gaussians_inner.apply_sds_pose = False
        self.gaussians_outer.apply_sds_pose = False

        # Main render (multi-layer)
        render_pkg = render_multi_layer_2dgs(viewpoint_cam, self.gaussians_inner, self.gaussians_outer, self.pipe, bg, apply_pose=True)
        image = render_pkg["render"]
        self.viewspace_point_tensor = render_pkg["viewspace_points"]
        self.visibility_filter = render_pkg["visibility_filter"]
        self.radii = render_pkg["radii"]
        rendered_image_obj = render_pkg['rendered_image_obj']

        # Regularization renders
        render_pkg_multilayer = None
        if iteration > self.scene.data_cfg.lambda_normal_iter or iteration > self.scene.data_cfg.lambda_dist_iter:
            render_pkg_multilayer = render_multi_layer_2dgs(viewpoint_cam, self.gaussians_inner, self.gaussians_outer, self.pipe, bg, apply_pose=False)

        render_pkg_garment_dum = None
        apply_dum_dist_start_iter = self.scene.data_cfg.apply_dum_dist_start_iter
        apply_dum_dist_end_iter = self.scene.data_cfg.apply_dum_dist_end_iter
        if self.scene.data_cfg.apply_dum_dist and apply_dum_dist_start_iter < iteration < apply_dum_dist_end_iter:
            render_pkg_garment_dum = render_multi_layer_2dgs(viewpoint_cam, self.gaussians_outer, self.gaussians_garment_dum, self.pipe, bg, apply_pose=False)

        # Loss computation inputs
        gt_image = viewpoint_cam.original_image.cuda()
        gt_seg_sapiens = viewpoint_cam.seg_image['sapiens']
        gt_foreground_mask = viewpoint_cam.gt_alpha_mask[0].cuda().long()
        gt_seg_4dDress = viewpoint_cam.seg_image['seg_label_img']

        apply_4dDress_seg = self.scene.data_cfg.apply_4dDress_seg
        seg_4dDress_outer_label = self.scene.data_cfg.seg_4dDress_outer_label
        apply_sapiens_bg = self.scene.data_cfg.apply_sapiens_bg
        if self.dataset.data_type == 'colmap':
            apply_4dDress_seg = False

        # Segmentation Masks
        if apply_4dDress_seg:
            mask_body = gt_foreground_mask.clone()
            label_tensor = torch.tensor(DRESS4D_SEG_LABELS[seg_4dDress_outer_label])[:, None, None] / 255.0
            mask_body[(gt_seg_4dDress == label_tensor).all(dim=0)] = 0
            mask_garment = torch.zeros((image.shape[1], image.shape[2]), dtype=torch.float, device='cuda')
            mask_garment[(gt_seg_4dDress == label_tensor).all(dim=0)] = 1
            mask_foreground = gt_foreground_mask.clone()
            if apply_sapiens_bg:
                mask_foreground = torch.zeros((image.shape[1], image.shape[2]), dtype=torch.float, device='cuda')
                mask_foreground[gt_seg_sapiens != SAPIENS_SEG_LABELS['background']] = 1.0
        else:
            mask_body = torch.zeros((image.shape[1], image.shape[2]), dtype=torch.float, device='cuda')
            mask_body[(gt_seg_sapiens != SAPIENS_SEG_LABELS['background']) & (gt_seg_sapiens != SAPIENS_SEG_LABELS[self.seg_garment_name])] = 1.0
            mask_garment = torch.zeros((image.shape[1], image.shape[2]), dtype=torch.float, device='cuda')
            mask_garment[gt_seg_sapiens == SAPIENS_SEG_LABELS[self.seg_garment_name]] = 1.0
            mask_foreground = torch.zeros((image.shape[1], image.shape[2]), dtype=torch.float, device='cuda')
            mask_foreground[gt_seg_sapiens != SAPIENS_SEG_LABELS['background']] = 1.0
        
        mask_body = mask_body.unsqueeze(0)
        mask_garment = mask_garment.unsqueeze(0)

        if self.scene.data_cfg.apply_body_shrink:
            mask_garment_shrinked = (1 - self.dilate_seg(mask_body.unsqueeze(0).float())).squeeze() * mask_foreground
        else:
            mask_garment_shrinked = mask_garment

        # Color labels
        red = torch.zeros_like(rendered_image_obj, dtype=torch.float, device='cuda')
        red[0, ...] = 1.0
        green = torch.zeros_like(rendered_image_obj, dtype=torch.float, device='cuda')
        green[1, ...] = 1.0
        bg_color_tensor = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float, device='cuda').unsqueeze(-1)

        # Losses
        seg_loss_body = l1_loss(rendered_image_obj * mask_foreground * mask_body, red * mask_foreground * mask_body)
        seg_loss_garment = l1_loss(rendered_image_obj * mask_foreground * mask_garment_shrinked, green * mask_foreground * mask_garment_shrinked)
        seg_loss_bg = l1_loss(rendered_image_obj[:, ~((mask_body.bool() | mask_garment.bool()))[0]], bg_color_tensor)

        Ll1_outer = l1_loss(image * mask_foreground * mask_garment_shrinked, gt_image * mask_foreground * mask_garment_shrinked)
        ssim_outer = ssim(image * mask_foreground * mask_garment_shrinked.float(), gt_image * mask_foreground * mask_garment_shrinked.float())
        img_loss = (1.0 - self.opt.lambda_dssim) * Ll1_outer + self.opt.lambda_dssim * (1.0 - ssim_outer)

        # Regularization
        lambda_normal = self.scene.data_cfg.lambda_normal if iteration > self.scene.data_cfg.lambda_normal_iter else 0.0
        lambda_dist = self.scene.data_cfg.lambda_dist if iteration > self.scene.data_cfg.lambda_dist_iter else 0.0
        
        if render_pkg_multilayer is not None:
            rend_dist = render_pkg_multilayer["rend_dist"]
            rend_normal = render_pkg_multilayer['rend_normal']
            surf_normal = render_pkg_multilayer['surf_normal']
            normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
            normal_loss = lambda_normal * (normal_error * mask_garment[None]).mean()
            dist_loss = lambda_dist * (rend_dist * mask_garment[None]).mean()
        else:
            normal_loss = torch.tensor(0.0)
            dist_loss = torch.tensor(0.0)

        # Dummy garment regularization
        dist_garment_dum_loss = torch.tensor(0.0)
        normal_dum_loss = torch.tensor(0.0)
        if self.scene.data_cfg.apply_dum_dist and render_pkg_garment_dum is not None and apply_dum_dist_start_iter < iteration < apply_dum_dist_end_iter:
            lambda_dist_garment_dum = self.scene.data_cfg.lambda_dist_body_dum
            dist_garment_dum_loss = lambda_dist_garment_dum * render_pkg_garment_dum['rend_dist'].mean()
            if iteration > 2500:
                rend_normal_dum = render_pkg_garment_dum['rend_normal']
                surf_normal_dum = render_pkg_garment_dum['surf_normal']
                normal_error_dum = (1 - (rend_normal_dum * surf_normal_dum).sum(dim=0))[None]
                normal_dum_loss = 0.008 * normal_error_dum.mean()

        # Combine
        w_seg_loss_body = self.scene.data_cfg.w_seg_loss_body
        w_seg_loss_garment = self.scene.data_cfg.w_seg_loss_garment
        w_seg_loss_bg = self.scene.data_cfg.w_seg_loss_bg
        w_img_loss = self.scene.data_cfg.w_img_loss
        w_dist_loss = self.scene.data_cfg.w_dist_loss
        w_normal_loss = self.scene.data_cfg.w_normal_loss

        loss = (
            img_loss * w_img_loss +
            seg_loss_bg * w_seg_loss_bg +
            seg_loss_body * w_seg_loss_body +
            seg_loss_garment * w_seg_loss_garment +
            normal_loss * w_normal_loss +
            dist_loss * w_dist_loss
        )

        if apply_op_control_start_iteration < iteration < apply_op_control_end_iteration:
            loss += torch.sum(self.sds_loss) * self.w_sds_loss

        if self.scene.data_cfg.apply_dum_dist and apply_dum_dist_start_iter < iteration < apply_dum_dist_end_iter:
            loss += dist_garment_dum_loss * w_dist_loss
            if iteration > 2500:
                loss += normal_dum_loss * w_normal_loss

        loss.backward()

        other_info = {
            'image': image,
            'gt_image': gt_image,
            'model_type': self.dataset.model_type,
            'render_path': self.render_path,
            'gts_path': self.gts_path,
        }
        
        self.ema_dist_garment_dum_for_log = 0.4 * dist_garment_dum_loss.item() + 0.6 * self.ema_dist_garment_dum_for_log
        
        log_dict = {
            'dist_loss': dist_loss,
            'normal_loss': normal_dum_loss, # Using dummy normal loss for log in original code? 
            # Original code: ema_normal_for_log = 0.4 * normal_dum_loss.item() + ...
            'other_info': other_info,
            'extra_log': {
                'distort_garment_dum': f"{self.ema_dist_garment_dum_for_log:.5f}",
                '#gau_garment': f"{self.gaussians_outer.get_xyz.shape[0]}"
            },
            'extra_wandb': {
                'distort_garment_dum': self.ema_dist_garment_dum_for_log,
                '#gau-body': self.gaussians_inner.get_xyz.shape[0],
                '#gau-garment': self.gaussians_outer.get_xyz.shape[0],
                '#gau-comp': self.gaussians_inner.get_xyz.shape[0] + self.gaussians_outer.get_xyz.shape[0],
            }
        }
        return loss, log_dict

    def densification(self, iteration):
        apply_op_control_start_iteration = self.scene.data_cfg.apply_op_control_start_iteration
        apply_op_control_end_iteration = self.scene.data_cfg.apply_op_control_end_iteration
        reset_list = self.scene.data_cfg.opacity_reset_list
        use_op_control_body = True 
        
        if self.dataset.model_type == 'multi-layer-2dgs' and iteration < self.scene.data_cfg.end_densify_iter and use_op_control_body:
            size_threshold_fix_step = 500
            inner_size = self.gaussians_inner.get_xyz.shape[0]
            outer_size = self.gaussians_outer.get_xyz.shape[0]

            grad = self.viewspace_point_tensor[inner_size:inner_size + outer_size]
            if apply_op_control_start_iteration < iteration < apply_op_control_end_iteration:
                viewspace_point_tensor_grad = torch.zeros_like(self.viewspace_point_tensor_list_cano_body[0])
                for idx in range(len(self.viewspace_point_tensor_list_cano_body)):
                    viewspace_point_tensor_grad += self.viewspace_point_tensor_list_cano_body[idx].grad
                grad += viewspace_point_tensor_grad[inner_size:inner_size + outer_size]

            self.gaussians_outer.max_radii2D[self.visibility_filter[inner_size:inner_size + outer_size]] = torch.max(
                self.gaussians_outer.max_radii2D[self.visibility_filter[inner_size:inner_size + outer_size]],
                self.radii[inner_size:inner_size + outer_size][self.visibility_filter[inner_size:inner_size + outer_size]]
            )
            self.gaussians_outer.add_densification_stats_grad(grad, self.visibility_filter[inner_size:inner_size + outer_size])

            if iteration > self.scene.system_cfg.densify_prune_start_step and iteration % self.scene.system_cfg.densify_prune_interval == 0:
                size_threshold = self.scene.system_cfg.size_threshold if iteration > size_threshold_fix_step else None
                old_gau_size = self.gaussians_outer.get_xyz.shape[0]
                cameras_extent = self.scene.data_cfg.max_scale * 100
                
                self.gaussians_outer.densify_and_prune(
                    self.scene.system_cfg.max_grad,
                    self.scene.data_cfg.prune_min_opacity,
                    cameras_extent,
                    size_threshold
                )
                logger.info(f'Iteration {iteration}. After densify_and_prune. Gaussian outer size: {old_gau_size} -> {self.gaussians_outer.get_xyz.shape[0]}. Gaussian inner size: {self.gaussians_inner.get_xyz.shape[0]}')

            if iteration in reset_list:
                self.gaussians_outer.reset_opacity(reset_val=self.scene.data_cfg.opacity_reset_val)
                logger.info(f'Reset gaussian opacity to {self.scene.data_cfg.opacity_reset_val}.')

        # Optimizer step (only outer garment)
        self.gaussians_outer.optimizer.step()
        self.gaussians_outer.optimizer.zero_grad(set_to_none=True)

        # Scale clipping
        if self.scene.data_cfg.apply_scale_clipping:
            current_scaling = self.gaussians_outer.get_scaling
            max_scale = self.scene.data_cfg.max_scale
            clipped_scaling = torch.min(current_scaling, torch.full_like(current_scaling, max_scale))
            new_scaling_params = self.gaussians_outer.scaling_inverse_activation(clipped_scaling)
            self.gaussians_outer._scaling.data.copy_(new_scaling_params)

        # Distance thresholding
        apply_distance_thresh = self.scene.data_cfg.apply_distance_thresh
        apply_distance_thresh_start_iter = self.scene.data_cfg.apply_distance_thresh_start_iter
        apply_distance_thresh_end_iter = self.scene.data_cfg.apply_distance_thresh_end_iter
        
        if apply_distance_thresh and apply_distance_thresh_start_iter < iteration and (iteration + 1) % (self.dataset.img_log_interval * 4) == 0:
            if iteration < apply_distance_thresh_end_iter:
                distance_thresh = self.scene.data_cfg.distance_thresh
            else:
                distance_thresh = self.scene.data_cfg.distance_thresh + 0.01
                
            pq = trimesh.proximity.ProximityQuery(self.pretrained_garment_mesh)
            garment_pts = self.gaussians_outer.get_xyz.detach().cpu().numpy()
            old_num = self.gaussians_outer.get_xyz.shape[0]
            _, distances, _ = pq.on_surface(garment_pts)
            masked = torch.from_numpy(distances > distance_thresh)
            self.gaussians_outer.prune_points(masked)
            new_num = self.gaussians_outer.get_xyz.shape[0]
            logger.info(f"(Iter {iteration}) Mask outlier 2D Gaussians by distance. Masked {old_num - new_num}. Old: {old_num} -> New: {new_num}")
            
            # Render check
            human_gaussian_test_data_iter = iter(self.human_gaussian_test_dataloader)
            # Need 'other_info' here? It's available via saved log_dict in loop but here we are in densification hook.
            # Actually `densification` hook in BaseTrainer is called with just iteration. 
            # I might need `other_info` or just pass a dummy or cached one.
            # For now, I'll skip rendering here or cache other_info. 
            # The original code rendered here using 'other_info' from the loop scope.
            # I will skip this render for now to avoid complexity, as it is just a check.

    def save_model(self, iteration):
        save_multi_layer_gs(
            self.scene.model_path, self.gaussians_inner, self.gaussians_outer, self.dataset.sh_degree,
            iteration, apply_2dgs=True, out_folder=self.out_folder
        )

    def rendering_results(self, iteration, other_info):
        # Render validation videos
        if iteration == 1 or iteration % (self.dataset.img_log_interval * 2) == 0 or (iteration in self.saving_iterations):
            if iteration == 1:
                human_gaussian_test_data_iter = iter(self.human_gaussian_test_dataloader)
                vid_path_body_cano = render_val_cam(
                    iteration, other_info, human_gaussian_test_data_iter, [self.gaussians_inner], self.pipe, self.background,
                    self.scene.data_cfg, isTest=True, postfix='body-cano', apply_shape=True
                )
                human_gaussian_test_data_iter = iter(self.human_gaussian_test_dataloader)
                vid_path_body_pose = render_val_cam(
                    iteration, other_info, human_gaussian_test_data_iter, [self.gaussians_inner], self.pipe, self.background,
                    self.scene.data_cfg, isTest=True, postfix='body-pose', apply_shape=True, apply_pose=True
                )
                
            human_gaussian_test_data_iter = iter(self.human_gaussian_test_dataloader)
            vid_path_garment_cano = render_val_cam(
                iteration, other_info, human_gaussian_test_data_iter, [self.gaussians_outer], self.pipe, self.background,
                self.scene.data_cfg, isTest=True, postfix='garment-cano', apply_shape=True
            )
            human_gaussian_test_data_iter = iter(self.human_gaussian_test_dataloader)
            vid_path_zcomp_cano = render_val_cam(
                iteration, other_info, human_gaussian_test_data_iter, [self.gaussians_inner, self.gaussians_outer], self.pipe, self.background,
                self.scene.data_cfg, isTest=True, postfix='zcomp-cano', apply_shape=True
            )
            human_gaussian_test_data_iter = iter(self.human_gaussian_test_dataloader)
            vid_path_zcomp_posed = render_val_cam(
                iteration, other_info, human_gaussian_test_data_iter, [self.gaussians_inner, self.gaussians_outer], self.pipe, self.background,
                self.scene.data_cfg, isTest=True, postfix='zcomp-posed', apply_shape=True, apply_pose=True
            )

            if self.run is not None:
                import wandb
                vids = {
                    'garment-cano': wandb.Video(vid_path_garment_cano, format='mp4'),
                    'zcomp-cano': wandb.Video(vid_path_zcomp_cano, format='mp4'),
                    'zcomp-posed': wandb.Video(vid_path_zcomp_posed, format='mp4'),
                }
                if iteration == 1:
                    vids['body-cano'] = wandb.Video(vid_path_body_cano, format='mp4')
                    vids['body-pose'] = wandb.Video(vid_path_body_pose, format='mp4')
                wandb.log(vids, step=iteration)
        
    def evaluation(self, iteration):
        # Evaluation
        if self.scene.data_cfg.apply_evaluate and self.dataset.data_type != 'colmap' and self.scene_eval is not None:
            if iteration == 1 or iteration % (self.dataset.img_log_interval * 4) == 0 or (iteration in self.saving_iterations):
                logger.info(f'Evaluation at iteration {iteration}')
                render_set_list(
                    model_path=f'{self.dataset.model_path}/results/{self.result_folder}',
                    name='eval',
                    res_type='comp',
                    views=self.scene_eval.getTestCameras(),
                    gaussians_list=[self.gaussians_inner, self.gaussians_outer],
                    pipeline=self.pipe,
                    background=self.background,
                    apply_shape=True,
                    apply_pose=True,
                    save_mask=False,
                )

                img_metrics = evaluate(
                    gt_dir=f'{self.dataset.source_path}/torch3d_imgs_test_gt',
                    renders_dir=f'{self.dataset.model_path}/results/{self.result_folder}/eval-comp',
                )
                
                if self.run is not None:
                    import wandb
                    try:
                        log_dict = {
                            "ssim":                         round(img_metrics["ssim"].item(), 4),
                            "psnr":                         round(img_metrics["psnr"].item(), 2),
                            "lpips":                        round(img_metrics["lpips"].item(), 4),
                        }
                    except Exception as e:
                        log_dict = {
                            "ssim":                         0.0,
                            "psnr":                         0.0,
                            "lpips":                        0.0,
                        }
                    wandb.log(log_dict, step=iteration)

def main():
    cmd_line = format_cmd_line(sys.argv)

    parser = ArgumentParser(description="Multi-layer 2DGS Outer Layer Training")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--config_file", type=str, default="config/seg_train.json")
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args = op.set_lr(args)
    args.cmd_line = cmd_line

    logger.info("Optimizing " + args.model_path)
    safe_state(args.quiet)

    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    trainer = OuterLayerTrainer(lp.extract(args), op.extract(args), pp.extract(args), args)
    trainer.train()

    logger.info("\nTraining complete.")

if __name__ == "__main__":
    main()
