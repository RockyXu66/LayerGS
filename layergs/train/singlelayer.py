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
from layergs.rendering import render_single_layer_2dgs, render_gaussian_list, network_gui
from layergs.utils.loss_utils import l1_loss, ssim
from layergs.utils.general_utils import safe_state
from layergs.utils.train_utils import (
    render_val_cam,
    set_sds,
    format_cmd_line,
    get_smplx_path,
)
from layergs.rendering.render_utils import render_set_list
from layergs.utils.metrics import evaluate
from arguments import ModelParams, PipelineParams, OptimizationParams
from layergs.train.base import BaseTrainer
from layergs.guidance import sds_controlnet

class SingleLayerTrainer(BaseTrainer):
    def __init__(self, dataset, opt, pipe, args):
        super().__init__(dataset, opt, pipe, args, "singlelayer")
        self.sds_controlnet = None
        self.viewspace_point_tensor_list_cano_comp = []
        self.sds_loss = torch.tensor(0.0)

    def init_scene(self):
        # Initialize Gaussians
        self.gaussians = GaussianModel(self.dataset.sh_degree, apply_2dgs=True)
        self.scene = Scene(self.dataset, self.gaussians, cfg_file=self.config_file)
        
        # Setup evaluation scene if needed (skip the evaluation for custom video)
        if self.scene.data_cfg.apply_evaluate and self.dataset.data_type != 'colmap':
            eval_dataset = copy.deepcopy(self.dataset)
            eval_dataset.eval = True
            self.scene_eval = Scene(eval_dataset, self.gaussians)

        # Initialize GALA SDS ControlNet
        smplx_file_path = get_smplx_path(self.dataset)
        self.sds_controlnet = sds_controlnet(
            self.scene.use_opengl_camera, 
            smplx_file_path, 
            dataset_type=self.dataset.data_type, 
            prompt_type='outer', 
            data_cfg=self.scene.data_cfg
        )

        # Sample points from SMPLX mesh
        mesh = trimesh.Trimesh(
            self.sds_controlnet.smplx_verts_apose.detach().cpu().numpy(), 
            self.sds_controlnet.faces
        )
        smplx_points, _ = trimesh.sample.sample_surface(mesh, self.scene.data_cfg.inner_init_sample_num)
        smplx_points = torch.from_numpy(smplx_points).float().cuda()

        # Initialize Gaussians from sampled points
        self.gaussians.create_from_xyz(smplx_points, object_idx=6)
        self.gaussians.shape_pose_offsets_grid = self.sds_controlnet.shape_pose_offsets_grid
        self.gaussians.lbs_weights_grid = self.sds_controlnet.lbs_weights_grid
        self.gaussians.gt_smplx_tfs = self.sds_controlnet.gt_smplx_tfs

    def train_step(self, iteration, viewpoint_cam, bg):
        # Pose handling
        if torch.rand(1).item() < self.scene.data_cfg.apose_prob:
            self.sds_controlnet.points3D = self.sds_controlnet.apose_points3D.copy()
            self.gaussians.apply_sds_pose = False
        else:
            self.gaussians.apply_sds_pose = True
            self.gaussians.sds_smplx_tfs = self.sds_controlnet.sds_smplx_tfs.clone()
            self.sds_controlnet.points3D = self.sds_controlnet.sds_points3D.copy()

        self.viewspace_point_tensor_list_cano_comp = []
        self.sds_loss = torch.tensor(0.0)
        
        # Get config values
        apply_op_control_start_iteration = self.scene.data_cfg.apply_op_control_start_iteration
        apply_op_control_end_iteration = self.scene.data_cfg.apply_op_control_end_iteration
        apply_foreground_mask = self.scene.data_cfg.apply_foreground_mask
        if self.dataset.data_type == 'colmap':
            apply_foreground_mask = True

        # SDS loss computation
        if apply_op_control_start_iteration < iteration < apply_op_control_end_iteration:
            images = []
            batch = next(self.human_gaussian_data_iter)
            for id in range(batch['c2w'].shape[0]):
                c2w = batch['c2w'][id]
                fovy = batch['fovy'][id]
                sds_cam = HumanGaussianCamera(c2w, fovy, height=1024, width=1024)

                random_bg = torch.rand((3), device="cuda")
                render_pkg = render_gaussian_list(sds_cam, [self.gaussians], self.pipe, random_bg, apply_shape=True, apply_2dgs=True)
                image = render_pkg["render"]
                viewspace_point_tensor = render_pkg["viewspace_points"]

                images.append(image)
                self.viewspace_point_tensor_list_cano_comp.append(viewspace_point_tensor)

            images = torch.stack(images, dim=0)
            mvp = batch['mvp_mtx'].detach().cpu().numpy()
            azimuth = batch['azimuth']
            self.sds_loss = self.sds_controlnet.compute_sds_loss(images.shape[2], images, mvp, azimuth)

        self.gaussians.apply_sds_pose = False

        # Main render
        render_pkg = render_single_layer_2dgs(viewpoint_cam, self.gaussians, self.pipe, bg, apply_pose=True)
        image = render_pkg["render"]
        self.viewspace_point_tensor = render_pkg["viewspace_points"]
        self.visibility_filter = render_pkg["visibility_filter"]
        self.radii = render_pkg["radii"]
        
        # Loss computation
        gt_image = viewpoint_cam.original_image.cuda()
        if apply_foreground_mask:
            gt_foreground_mask = viewpoint_cam.gt_alpha_mask.cuda().long()
            gt_image = gt_image * gt_foreground_mask

        Ll1 = l1_loss(image, gt_image)
        img_loss = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        # 2DGS regularization
        lambda_normal = self.scene.data_cfg.lambda_normal if iteration > self.scene.data_cfg.lambda_normal_iter else 0.0
        lambda_dist = self.scene.data_cfg.lambda_dist if iteration > self.scene.data_cfg.lambda_dist_iter else 0.0
        rend_dist = render_pkg["rend_dist"]
        rend_normal = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error * viewpoint_cam.gt_alpha_mask).mean()
        dist_loss = lambda_dist * (rend_dist * viewpoint_cam.gt_alpha_mask).mean()

        # Combine losses
        w_img_loss = self.scene.data_cfg.w_img_loss
        w_dist_loss = self.scene.data_cfg.w_dist_loss
        w_normal_loss = self.scene.data_cfg.w_normal_loss

        loss = img_loss * w_img_loss + normal_loss * w_normal_loss + dist_loss * w_dist_loss
        
        if apply_op_control_start_iteration < iteration < apply_op_control_end_iteration:
            loss += torch.sum(self.sds_loss) * self.w_sds_loss

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
            'other_info': other_info
        }
        return loss, log_dict

    def densification(self, iteration):
        apply_op_control_start_iteration = self.scene.data_cfg.apply_op_control_start_iteration
        apply_op_control_end_iteration = self.scene.data_cfg.apply_op_control_end_iteration
        reset_list = self.scene.data_cfg.opacity_reset_list
        gau_size_body = self.gaussians.get_xyz.shape[0]

        if self.dataset.model_type == 'multi-layer-2dgs' and iteration < self.scene.data_cfg.end_densify_iter:
            grad = self.viewspace_point_tensor
            if apply_op_control_start_iteration < iteration < apply_op_control_end_iteration:
                viewspace_point_tensor_grad = torch.zeros_like(self.viewspace_point_tensor_list_cano_comp[0])
                for idx in range(len(self.viewspace_point_tensor_list_cano_comp)):
                    viewspace_point_tensor_grad += self.viewspace_point_tensor_list_cano_comp[idx].grad
                viewspace_point_tensor_grad /= len(self.viewspace_point_tensor_list_cano_comp)
                grad += viewspace_point_tensor_grad
                
            self.gaussians.max_radii2D[self.visibility_filter[:gau_size_body]] = torch.max(
                self.gaussians.max_radii2D[self.visibility_filter[:gau_size_body]], 
                self.radii[:gau_size_body][self.visibility_filter[:gau_size_body]]
            )
            self.gaussians.add_densification_stats(grad, self.visibility_filter, apply_range=[0, gau_size_body])

            if iteration > self.opt.densify_from_iter and iteration % self.opt.densification_interval == 0:
                size_threshold = 20 if iteration > self.opt.opacity_reset_interval else None
                old_gau_size = self.gaussians.get_xyz.shape[0]
                
                cameras_extent = 10 # scaling threshold
                self.gaussians.densify_and_prune(
                    self.opt.densify_grad_threshold, 
                    self.scene.data_cfg.prune_min_opacity, 
                    cameras_extent, 
                    size_threshold
                )
                logger.info(f'Iteration {iteration}. After densify_and_prune. Gaussian size: {old_gau_size} -> {self.gaussians.get_xyz.shape[0]}')

            if iteration in reset_list:
                self.gaussians.reset_opacity(reset_val=self.scene.data_cfg.opacity_reset_val)
                logger.info(f'Reset gaussian opacity to {self.scene.data_cfg.opacity_reset_val}.')

        # Optimizer step
        self.gaussians.optimizer.step()
        self.gaussians.optimizer.zero_grad(set_to_none=True)

    def rendering_results(self, iteration, other_info):
        if iteration == 1 or iteration % (self.dataset.img_log_interval * 2) == 0 or (iteration in self.saving_iterations):
            human_gaussian_test_data_iter = iter(self.human_gaussian_test_dataloader)
            vid_path_zcomp_cano = render_val_cam(
                iteration, other_info, human_gaussian_test_data_iter, [self.gaussians], self.pipe, self.background,
                self.scene.data_cfg, isTest=True, postfix='zcomp-cano', apply_shape=True
            )
            human_gaussian_test_data_iter = iter(self.human_gaussian_test_dataloader)
            vid_path_zcomp_posed = render_val_cam(
                iteration, other_info, human_gaussian_test_data_iter, [self.gaussians], self.pipe, self.background,
                self.scene.data_cfg, isTest=True, postfix='zcomp-posed', apply_shape=True, apply_pose=True
            )

            if self.run is not None:
                import wandb
                wandb.log({
                    'zcomp-cano': wandb.Video(vid_path_zcomp_cano, format='mp4'),
                    'zcomp-posed': wandb.Video(vid_path_zcomp_posed, format='mp4'),
                }, step=iteration)
                
    def evaluation(self, iteration):
        if iteration == 1 or iteration % (self.dataset.img_log_interval * 2) == 0 or (iteration in self.saving_iterations):
            if self.scene.data_cfg.apply_evaluate and self.dataset.data_type != 'colmap' and self.scene_eval is not None:
                if iteration == 1 or iteration % (self.dataset.img_log_interval * 4) == 0 or (iteration in self.saving_iterations):
                    logger.info(f'Evaluation at iteration {iteration}')
                    render_set_list(
                        model_path='debug',
                        name='eval',
                        res_type='comp',
                        views=self.scene_eval.getTestCameras(),
                        gaussians_list=[self.gaussians],
                        pipeline=self.pipe,
                        background=self.background,
                        apply_shape=True,
                        apply_pose=True,
                    )

                    img_metrics = evaluate(
                        gt_dir=f'{self.dataset.source_path}/torch3d_imgs_test_gt',
                        renders_dir='debug/eval-comp',
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

    parser = ArgumentParser(description="Single-layer 2DGS Training")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--config_file", type=str, default="config/inpaint.yaml")
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args = op.set_lr(args)
    args.cmd_line = cmd_line

    logger.info("Optimizing " + args.model_path)
    safe_state(args.quiet)

    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    trainer = SingleLayerTrainer(lp.extract(args), op.extract(args), pp.extract(args), args)
    trainer.train()

    logger.info("\nTraining complete.")

if __name__ == "__main__":
    main()
