#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#

import sys
import copy
import torch
import trimesh
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
    format_cmd_line,
    get_smplx_path,
    parse_reward_table,
    SAPIENS_SEG_LABELS,
    DRESS4D_SEG_LABELS
)
from layergs.rendering.render_utils import render_set_list
from layergs.utils.metrics import evaluate
from layergs.utils.gaussian_factory import create_gaussians_from_ply
from arguments import ModelParams, PipelineParams, OptimizationParams
from layergs.train.base import BaseTrainer
from layergs.guidance import sds_controlnet

class InnerLayerTrainer(BaseTrainer):
    def __init__(self, dataset, opt, pipe, args):
        super().__init__(dataset, opt, pipe, args, f"train_inner")
        self.gaussians_inner = None
        self.gaussians_outer = None
        self.gaussians_body_dum = None
        self.sds_controlnet = None
        
        self.viewspace_point_tensor_list_cano_body = []
        self.sds_loss_body = torch.tensor(0.0)
        self.dilate_seg = None
        self.garment_type = None
        self.seg_garment_name = None

    def init_scene(self):
        # Initialize Inner Gaussians (Main optimization target)
        self.gaussians_inner = GaussianModel(self.dataset.sh_degree, apply_2dgs=True)
        self.gaussians = self.gaussians_inner # Alias for base class usage
        self.scene = Scene(self.dataset, self.gaussians_inner, cfg_file=self.config_file)
        
        # Evaluation scene
        if self.scene.data_cfg.apply_evaluate and self.dataset.data_type != 'colmap':
            eval_dataset = copy.deepcopy(self.dataset)
            eval_dataset.eval = True
            self.scene_eval = Scene(eval_dataset, self.gaussians_inner)

        # Initialize GALA SDS ControlNet
        smplx_file_path = get_smplx_path(self.dataset)
        self.sds_controlnet = sds_controlnet(
            self.scene.use_opengl_camera, 
            smplx_file_path, 
            dataset_type=self.dataset.data_type, 
            data_cfg=self.scene.data_cfg
        )

        # Sample points from SMPLX mesh for inner body
        mesh = trimesh.Trimesh(
            self.sds_controlnet.smplx_verts_apose.detach().cpu().numpy(), 
            self.sds_controlnet.faces
        )
        smplx_points, _ = trimesh.sample.sample_surface(mesh, self.scene.data_cfg.inner_init_sample_num)
        smplx_points = torch.from_numpy(smplx_points).float().cuda()

        self.gaussians_inner.create_from_xyz(smplx_points, object_idx=6)
        self.gaussians_inner.shape_pose_offsets_grid = self.sds_controlnet.shape_pose_offsets_grid
        self.gaussians_inner.lbs_weights_grid = self.sds_controlnet.lbs_weights_grid
        self.gaussians_inner.gt_smplx_tfs = self.sds_controlnet.gt_smplx_tfs

        # Create dummy body for distance regularization
        if self.scene.data_cfg.apply_dum_dist:
            self.gaussians_body_dum = GaussianModel(self.dataset.sh_degree, apply_2dgs=True)
            dum_points, _ = trimesh.sample.sample_surface(mesh, self.scene.data_cfg.inner_dum_sample_num)
            dum_points = torch.from_numpy(dum_points).float().cuda()
            self.gaussians_body_dum.create_from_xyz(dum_points, object_idx=6)
            self.gaussians_body_dum.shape_pose_offsets_grid = self.sds_controlnet.shape_pose_offsets_grid
            self.gaussians_body_dum.lbs_weights_grid = self.sds_controlnet.lbs_weights_grid
            self.gaussians_body_dum.gt_smplx_tfs = self.sds_controlnet.gt_smplx_tfs

        # Load outer garment from pretrained (frozen)
        pretrained_name = self.dataset.pretrained_name
        mesh_name = self.dataset.mesh_name
        mesh_path = f'{self.dataset.model_path}/{pretrained_name}/mesh/{mesh_name}'
        ply_path = f'{self.dataset.model_path}/{pretrained_name}/{self.scene.data_cfg.ply_name}.ply'
        
        self.gaussians_outer = create_gaussians_from_ply(
            mesh_path, ply_path, self.dataset.sh_degree, 
            method=self.scene.data_cfg.from_ply, 
            object_idx=3
        )
        self.gaussians_outer.shape_pose_offsets_grid = self.sds_controlnet.shape_pose_offsets_grid
        self.gaussians_outer.lbs_weights_grid = self.sds_controlnet.lbs_weights_grid
        self.gaussians_outer.gt_smplx_tfs = self.sds_controlnet.gt_smplx_tfs

        # Segmentation helpers
        dilation_size = self.scene.data_cfg.seg_dilation_size
        self.dilate_seg = torch.nn.MaxPool2d(kernel_size=dilation_size, stride=1, padding=dilation_size // 2)
        self.garment_type = self.scene.data_cfg.inpaint_garment_type
        self.seg_garment_name = f'{self.garment_type}_clothing'
        
    def setup_optimization(self):
        # Setup both inner and outer, though outer might be frozen
        self.gaussians_inner.training_setup(self.opt)
        self.gaussians_outer.training_setup(self.opt)


    def update_learning_rate(self, iteration):
        self.gaussians_inner.update_learning_rate(iteration)
        self.gaussians_outer.update_learning_rate(iteration)

    def render_for_gui(self, custom_cam, scaling_modifer):
        # Override to use gaussians_inner for GUI
        return render(custom_cam, self.gaussians_inner, self.pipe, self.background, scaling_modifer)["render"]

    def train_step(self, iteration, viewpoint_cam, bg):
        use_op_control_body = True
        apply_op_control_start_iteration = self.scene.data_cfg.apply_op_control_start_iteration
        apply_op_control_end_iteration = self.scene.data_cfg.apply_op_control_end_iteration
        
        self.viewspace_point_tensor_list_cano_body = []
        self.sds_loss_body = torch.tensor(0.0)

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

        # SDS loss computation (inner body only)
        if use_op_control_body and apply_op_control_start_iteration < iteration < apply_op_control_end_iteration:
            images = []
            batch = next(self.human_gaussian_data_iter)
            for id in range(batch['c2w'].shape[0]):
                c2w = batch['c2w'][id]
                fovy = batch['fovy'][id]
                sds_cam = HumanGaussianCamera(c2w, fovy, height=1024, width=1024)

                random_bg = torch.rand((3), device="cuda")
                render_pkg = render_gaussian_list(sds_cam, [self.gaussians_inner], self.pipe, random_bg, apply_shape=True, apply_2dgs=True)
                image = render_pkg["render"]
                viewspace_point_tensor = render_pkg["viewspace_points"]

                images.append(image)
                self.viewspace_point_tensor_list_cano_body.append(viewspace_point_tensor)

            images = torch.stack(images, dim=0)
            mvp = batch['mvp_mtx'].detach().cpu().numpy()
            azimuth = batch['azimuth']
            self.sds_loss_body = self.sds_controlnet.compute_sds_loss(images.shape[2], images, mvp, azimuth)

        self.gaussians_inner.apply_sds_pose = False
        self.gaussians_outer.apply_sds_pose = False

        # Main render (multi-layer)
        render_pkg = render_multi_layer_2dgs(viewpoint_cam, self.gaussians_inner, self.gaussians_outer, self.pipe, bg, apply_pose=True)
        image = render_pkg["render"]
        self.viewspace_point_tensor = render_pkg["viewspace_points"]
        self.visibility_filter = render_pkg["visibility_filter"]
        self.radii = render_pkg["radii"]
        rendered_image_obj = render_pkg['rendered_image_obj']

        # Dummy body for distance regularization
        render_pkg_body_dum = None
        if self.scene.data_cfg.apply_dum_dist:
            render_pkg_body_dum = render_multi_layer_2dgs(viewpoint_cam, self.gaussians_inner, self.gaussians_body_dum, self.pipe, bg, apply_pose=True)

        # Loss computation inputs
        gt_image = viewpoint_cam.original_image.cuda()
        gt_seg_sapiens = viewpoint_cam.seg_image['sapiens']
        gt_seg_4dDress = viewpoint_cam.seg_image['seg_label_img']
        gt_foreground_mask = viewpoint_cam.gt_alpha_mask[0].cuda().long()
        
        apply_sam_seg = self.scene.data_cfg.apply_sam_seg
        apply_4dDress_seg = self.scene.data_cfg.apply_4dDress_seg
        seg_4dDress_outer_label = self.scene.data_cfg.seg_4dDress_outer_label
        apply_sapiens_bg = self.scene.data_cfg.apply_sapiens_bg
        if self.dataset.data_type == 'colmap':
            apply_4dDress_seg = False

        gt_seg_image = None
        if apply_sam_seg:
            gt_seg_image = viewpoint_cam.seg_image[f'sam_{self.garment_type}'][0].cuda().long()

        # Segmentation Masks logic
        if apply_sam_seg:
            mask_body = gt_foreground_mask & (1 - gt_seg_image)
            mask_garment = gt_seg_image
            mask_foreground = gt_foreground_mask
        elif apply_4dDress_seg:
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

        # Handle original image masking
        if self.scene.data_cfg.apply_ori_img:
            gt_seg_sapiens_ori = viewpoint_cam.seg_image['sapiens_ori']
            ori_img = viewpoint_cam.seg_image['ori_img'].cuda()
            mask_garment_ori = torch.zeros((image.shape[1], image.shape[2]), dtype=torch.float, device='cuda')
            mask_garment_ori[(gt_seg_sapiens_ori == SAPIENS_SEG_LABELS['upper_clothing']) | (gt_seg_sapiens_ori == SAPIENS_SEG_LABELS['lower_clothing'])] = 1.0
            mask_body_ori = (1 - self.dilate_seg(mask_garment_ori.unsqueeze(0).float())).squeeze() * mask_foreground
            mask_body_ori = torch.where(mask_body_ori != 0)
            gt_image[:, mask_body_ori[0], mask_body_ori[1]] = ori_img[:, mask_body_ori[0], mask_body_ori[1]]
            mask_foreground = torch.zeros((image.shape[1], image.shape[2]), dtype=torch.float, device='cuda')
            mask_foreground[(gt_seg_sapiens != SAPIENS_SEG_LABELS['background']) & (gt_seg_sapiens_ori != SAPIENS_SEG_LABELS['background'])] = 1.0

        # Color labels
        red = torch.zeros_like(rendered_image_obj, dtype=torch.float, device='cuda')
        red[0, ...] = 1.0
        green = torch.zeros_like(rendered_image_obj, dtype=torch.float, device='cuda')
        green[1, ...] = 1.0
        bg_color_tensor = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float, device='cuda').unsqueeze(-1)

        if self.scene.data_cfg.apply_body_shrink:
            mask_body = (1 - self.dilate_seg(mask_garment.unsqueeze(0).float())).squeeze() * mask_foreground

        # Losses
        seg_loss_body = l1_loss(rendered_image_obj * mask_foreground * mask_body, red * mask_foreground * mask_body)
        seg_loss_garment = l1_loss(rendered_image_obj * mask_foreground * mask_garment, green * mask_foreground * mask_garment)
        seg_loss_bg = l1_loss(rendered_image_obj[:, ~((mask_body.bool() | mask_garment.bool()))[0]], bg_color_tensor)

        Ll1_body = l1_loss(image * mask_foreground * mask_body, gt_image * mask_foreground * mask_body)
        ssim_body = ssim(image * mask_body.float(), gt_image * mask_body.float())
        img_loss = (1.0 - self.opt.lambda_dssim) * Ll1_body + self.opt.lambda_dssim * (1.0 - ssim_body)

        # Regularization
        lambda_normal = self.scene.data_cfg.lambda_normal if iteration > self.scene.data_cfg.lambda_normal_iter else 0.0
        lambda_dist = self.scene.data_cfg.lambda_dist if iteration > self.scene.data_cfg.lambda_dist_iter else 0.0
        rend_dist = render_pkg["rend_dist"]
        rend_normal = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error * mask_body[None]).mean()
        dist_loss = lambda_dist * (rend_dist * mask_body[None]).mean()

        # Combine
        w_seg_loss_body = self.scene.data_cfg.w_seg_loss_body
        w_seg_loss_garment = self.scene.data_cfg.w_seg_loss_garment
        w_seg_loss_bg = self.scene.data_cfg.w_seg_loss_bg
        w_img_loss = self.scene.data_cfg.w_img_loss
        w_dist_loss = self.scene.data_cfg.w_dist_loss
        w_normal_loss = self.scene.data_cfg.w_normal_loss
        lambda_dist_body_dum = self.scene.data_cfg.lambda_dist_body_dum

        loss = (
            img_loss * w_img_loss +
            seg_loss_bg * w_seg_loss_bg +
            normal_loss * w_normal_loss +
            dist_loss * w_dist_loss +
            seg_loss_body * w_seg_loss_body +
            seg_loss_garment * w_seg_loss_garment
        )
        
        if apply_op_control_start_iteration < iteration < apply_op_control_end_iteration:
            loss += torch.sum(self.sds_loss_body) * self.w_sds_loss
            
        if self.scene.data_cfg.apply_dum_dist and render_pkg_body_dum is not None:
            dist_body_dum_loss = lambda_dist_body_dum * (render_pkg_body_dum['rend_dist'] * mask_garment).mean()
            loss += dist_body_dum_loss * w_dist_loss

        loss.backward()

        other_info = {
            'image': image,
            'gt_image': gt_image,
            'model_type': self.dataset.model_type,
            'render_path': self.render_path,
            'gts_path': self.gts_path,
        }
        
        log_dict = {
            'dist_loss': dist_loss,
            'normal_loss': normal_loss,
            'other_info': other_info,
            'extra_log': {'#gau_garment': f"{self.gaussians_outer.get_xyz.shape[0]}"},
            'extra_wandb': {
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
        use_op_control_body = True # Copied from original logic
        
        if self.dataset.model_type == 'multi-layer-2dgs' and iteration < self.scene.data_cfg.end_densify_iter and use_op_control_body:
            size_threshold_fix_step = 1500
            inpaint_size = self.gaussians_inner.get_xyz.shape[0]

            grad = self.viewspace_point_tensor[0:inpaint_size]
            if apply_op_control_start_iteration < iteration < apply_op_control_end_iteration:
                viewspace_point_tensor_grad = torch.zeros_like(self.viewspace_point_tensor_list_cano_body[0])
                for idx in range(len(self.viewspace_point_tensor_list_cano_body)):
                    viewspace_point_tensor_grad += self.viewspace_point_tensor_list_cano_body[idx].grad
                grad += viewspace_point_tensor_grad

            self.gaussians_inner.max_radii2D[self.visibility_filter[:inpaint_size]] = torch.max(
                self.gaussians_inner.max_radii2D[self.visibility_filter[:inpaint_size]],
                self.radii[:inpaint_size][self.visibility_filter[:inpaint_size]]
            )
            self.gaussians_inner.add_densification_stats_grad(grad, self.visibility_filter, apply_range=[0, inpaint_size])

            if iteration > self.scene.system_cfg.densify_prune_start_step and iteration % self.scene.system_cfg.densify_prune_interval == 0:
                size_threshold = self.scene.system_cfg.size_threshold if iteration > size_threshold_fix_step else None
                old_gau_size = self.gaussians_inner.get_xyz.shape[0]
                cameras_extent = self.scene.data_cfg.max_scale * 100 
                
                self.gaussians_inner.densify_and_prune(
                    self.scene.system_cfg.max_grad,
                    self.scene.data_cfg.prune_min_opacity,
                    cameras_extent,
                    size_threshold
                )
                logger.info(f'Iteration {iteration}. After densify_and_prune. Gaussian size: {old_gau_size} -> {self.gaussians_inner.get_xyz.shape[0]}. Gaussian garment size: {self.gaussians_outer.get_xyz.shape[0]}')

            if iteration in reset_list:
                self.gaussians_inner.reset_opacity(reset_val=self.scene.data_cfg.opacity_reset_val)
                logger.info(f'Reset gaussian opacity to {self.scene.data_cfg.opacity_reset_val}.')

        # Optimizer step (only inner body)
        self.gaussians_inner.optimizer.step()
        self.gaussians_inner.optimizer.zero_grad(set_to_none=True)

        # Scale clipping
        if self.scene.data_cfg.apply_scale_clipping:
            current_scaling = self.gaussians_inner.get_scaling
            max_scale = self.scene.data_cfg.max_scale
            clipped_scaling = torch.min(current_scaling, torch.full_like(current_scaling, max_scale))
            new_scaling_params = self.gaussians_inner.scaling_inverse_activation(clipped_scaling)
            self.gaussians_inner._scaling.data.copy_(new_scaling_params)

    def save_model(self, iteration):
        save_multi_layer_gs(
            self.scene.model_path, self.gaussians_inner, self.gaussians_outer, self.dataset.sh_degree,
            iteration, apply_2dgs=True, out_folder=self.out_folder
        )

    def rendering_results(self, iteration, other_info):
        # Render validation videos
        if iteration == 1 or iteration % (self.dataset.img_log_interval * 2) == 0 or (iteration in self.saving_iterations):
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
            
            if iteration == 1:
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
                    'body-cano': wandb.Video(vid_path_body_cano, format='mp4'),
                    'body-pose': wandb.Video(vid_path_body_pose, format='mp4'),
                    'zcomp-cano': wandb.Video(vid_path_zcomp_cano, format='mp4'),
                    'zcomp-posed': wandb.Video(vid_path_zcomp_posed, format='mp4'),
                }
                if iteration == 1:
                    vids['garment-cano'] = wandb.Video(vid_path_garment_cano, format='mp4')
                wandb.log(vids, step=iteration)
        
    def evaluation(self, iteration):
        # Evaluation
        if self.scene.data_cfg.apply_evaluate and self.dataset.data_type != 'colmap' and self.scene_eval is not None:
            if iteration == 1 or iteration % (self.dataset.img_log_interval * 4) == 0 or (iteration in self.saving_iterations):
                logger.info(f'Evaluation at iteration {iteration}')
                # First render: body only (canonical)
                render_set_list(
                    model_path=f'{self.dataset.model_path}/results/{self.result_folder}',
                    name='eval',
                    res_type='rotation-body_cano',
                    views=self.scene_eval.getTestCameras(),
                    gaussians_list=[self.gaussians_inner],
                    pipeline=self.pipe,
                    background=self.background,
                    apply_shape=True,
                    apply_pose=False,
                    save_mask=False,
                )

                result = self.run_IR()

                # Second render: composition (both layers)
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
                    log_dict = {
                        "ssim":                         round(img_metrics["ssim"].item(), 4),
                        "psnr":                         round(img_metrics["psnr"].item(), 2),
                        "lpips":                        round(img_metrics["lpips"].item(), 4),
                    }
                    try:
                        img_reward = parse_reward_table(result.stdout)
                        log_dict.update({
                            "ours_clip":                    img_reward["Ours w/o scan"][0],
                            "ours_image_reward":            img_reward["Ours w/o scan"][1],
                            # "gala_w_scan_clip":             img_reward['GALA w/ scan'][0],
                            # "gala_w_scan_image_reward":     img_reward['GALA w/ scan'][1],
                            # "gala_wo_scan_clip":            img_reward['GALA w/o scan'][0],
                            # "gala_wo_scan_image_reward":    img_reward['GALA w/o scan'][1],
                        })
                    except Exception as e:
                        log_dict.update({
                            "ours_clip":                    0.0,
                            "ours_image_reward":            0.0,
                            # "gala_w_scan_clip":             0.0,
                            # "gala_w_scan_image_reward":     0.0,
                            # "gala_wo_scan_clip":            0.0,
                            # "gala_wo_scan_image_reward":    0.0,
                        })
                    wandb.log(log_dict, step=iteration)
    
    def run_IR(self):
        import subprocess
        IR_ENV = '/home/yixu/Tools/anaconda3/envs/IR/bin/python' # replace this with your own ImageReward environment
        import os
        PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        command = f"{IR_ENV} tools/inpainting_body_comparison.py --subject_name {self.dataset.model_path.split('/')[-1]} --gs_folder_path {PROJECT_ROOT}/{self.dataset.model_path}/results/{self.result_folder}/eval-rotation-body_cano"
        print(command)
        result = subprocess.run(command, shell=True, capture_output=True, text=True)

        logger.info(result.stdout)
        return result

def main():
    cmd_line = format_cmd_line(sys.argv)

    parser = ArgumentParser(description="Multi-layer 2DGS Inner Layer Training")
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
    
    trainer = InnerLayerTrainer(lp.extract(args), op.extract(args), pp.extract(args), args)
    trainer.train()

    logger.info("\nTraining complete.")

if __name__ == "__main__":
    main()
