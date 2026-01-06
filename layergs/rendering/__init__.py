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
import math
import pytorch3d
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
# from diff_gaussian_rasterization_cls import GaussianRasterizationSettings as GaussianRasterizationSettingsCls, GaussianRasterizer as GaussianRasterizerCls
# from diff_gaussian_rasterization_proj import GaussianRasterizationSettings as GaussianRasterizationSettingsProj, GaussianRasterizer as GaussianRasterizerProj
from diff_surfel_rasterization import GaussianRasterizationSettings as GaussianRasterizationSettingsSurfel, GaussianRasterizer as GaussianRasterizerSurfel

from layergs.scene.gaussian_model import GaussianModel
from layergs.utils.sh_utils import eval_sh, RGB2SH
from layergs.utils.point_utils import depth_to_normal
from gala_utils import helpers

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}

def render_seg(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene with object class on each gaussian. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettingsCls(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizerCls(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    sh_objs = pc.get_object

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
    
    # verts = pc.get_xyz
    # offsets = torch.nn.functional.grid_sample(pc.shape_pose_offsets_grid , verts[None, None, None, :, :]).reshape(1, 3, -1).permute(0, 2, 1)
    # lbs_weights = torch.nn.functional.grid_sample(pc.lbs_weights_grid, verts[None, None, None, :, :]).reshape(1, 55, -1).permute(0, 2, 1)
    # verts = verts + offsets[0]  # shape pose blend shapes
    # means3D = helpers.skinning(verts, lbs_weights, pc.gt_smplx_tfs)[0]  # lbs
    # yoffset = torch.zeros((means3D.shape[0], 3), dtype=torch.float, device='cuda')
    # means3D = means3D + yoffset
    # w_tf = torch.einsum("bpn,bnij->bpij", lbs_weights, pc.gt_smplx_tfs)
    # rotations = pc._rotation#[:, [1,2,3,0]]      # 3dgs use wxyz for quaternion and scipy use xyzw for quaternion
    # rot_mat = pytorch3d.transforms.quaternion_to_matrix(rotations)
    # new_rot_mat = w_tf[0, :, :3, :3] @ rot_mat
    # new_quat = pytorch3d.transforms.matrix_to_quaternion(new_rot_mat)
    # rotations = new_quat#[:, [3,0,1,2]]

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, rendered_objects = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        sh_objs = sh_objs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "render_object": rendered_objects}

def render_naive_inpaint(viewpoint_camera, pc : GaussianModel, freezed_pc: GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(torch.concat([pc.get_xyz, freezed_pc.get_xyz]), dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = torch.concat([pc.get_xyz, freezed_pc.get_xyz])
    means2D = screenspace_points
    opacity = torch.concat([pc.get_opacity, freezed_pc.get_opacity])

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = torch.concat([pc.get_scaling, freezed_pc.get_scaling])
        rotations = torch.concat([pc.get_rotation, freezed_pc.get_rotation])

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = torch.concat([pc.get_features, freezed_pc.get_features])
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # rendered_image, radii = rasterizer(
    rendered_image, radii, rendered_depth, rendered_alpha  = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "depth": rendered_depth,
            "alpha": rendered_alpha,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}

def render_gaussian_list(viewpoint_camera, pc_list : list[GaussianModel], pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, \
                         apply_shape = False, apply_pose = False, apply_2dgs=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(torch.concat([pc.get_xyz for pc in pc_list]), dtype=pc_list[0].get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    if apply_2dgs:
        GRSFunc = GaussianRasterizationSettingsSurfel
        RasterizerFunc = GaussianRasterizerSurfel
    else:
        GRSFunc = GaussianRasterizationSettings
        RasterizerFunc = GaussianRasterizer

    raster_settings = GRSFunc(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc_list[0].active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = RasterizerFunc(raster_settings=raster_settings)

    if apply_shape: 
        # means3D_list = []
        # lbs_weights_list = []
        # for pc in pc_list:
        #     verts = pc.get_xyz
        #     offsets = torch.nn.functional.grid_sample(pc.shape_pose_offsets_grid , verts[None, None, None, :, :]).reshape(1, 3, -1).permute(0, 2, 1)
        #     verts = verts + offsets[0]  # shape pose blend shapes
        #     if apply_pose:
        #         lbs_weights = torch.nn.functional.grid_sample(pc.lbs_weights_grid, verts[None, None, None, :, :]).reshape(1, 55, -1).permute(0, 2, 1)
        #         means3D = helpers.skinning(verts, lbs_weights, pc.gt_smplx_tfs)[0]  # lbs
        #         lbs_weights_list.append(lbs_weights)
        #     elif pc.apply_sds_pose:
        #         lbs_weights = torch.nn.functional.grid_sample(pc.lbs_weights_grid, verts[None, None, None, :, :]).reshape(1, 55, -1).permute(0, 2, 1)
        #         means3D = helpers.skinning(verts, lbs_weights, pc.sds_smplx_tfs)[0]  # lbs
        #         lbs_weights_list.append(lbs_weights)
        #     else:
        #         means3D = verts
        #     means3D_list.append(means3D)
        # means3D = torch.concat(means3D_list)
        # if len(lbs_weights_list) != 0:
        #     lbs_weights = torch.concat(lbs_weights_list, dim=1)

        verts = torch.concat([pc.get_xyz for pc in pc_list])
        offsets = torch.nn.functional.grid_sample(pc_list[0].shape_pose_offsets_grid , verts[None, None, None, :, :]).reshape(1, 3, -1).permute(0, 2, 1)
        lbs_weights = torch.nn.functional.grid_sample(pc_list[0].lbs_weights_grid, verts[None, None, None, :, :]).reshape(1, 55, -1).permute(0, 2, 1)
        verts = verts + offsets[0]  # shape pose blend shapes
        if apply_pose:
            means3D = helpers.skinning(verts, lbs_weights, pc_list[0].gt_smplx_tfs)[0]  # lbs
            yoffset = torch.zeros((means3D.shape[0], 3), dtype=torch.float, device='cuda')
            means3D = means3D + yoffset
        elif pc_list[0].apply_sds_pose:
            means3D = helpers.skinning(verts, lbs_weights, pc_list[0].sds_smplx_tfs)[0]  # lbs
        else:
            means3D = verts
    else:
        means3D = torch.concat([pc.get_xyz for pc in pc_list])

    means2D = screenspace_points
    opacity = torch.concat([pc.get_opacity for pc in pc_list])

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = torch.concat([pc.get_covariance(scaling_modifier) for pc in pc_list])
    else:
        scales = torch.concat([pc.get_scaling for pc in pc_list])
        rotations = torch.concat([pc.get_rotation for pc in pc_list])
        if apply_pose:
            w_tf = torch.einsum("bpn,bnij->bpij", lbs_weights, pc_list[0].gt_smplx_tfs)
            rot_mat = pytorch3d.transforms.quaternion_to_matrix(rotations)
            new_rot_mat = w_tf[0, :, :3, :3] @ rot_mat
            new_quat = pytorch3d.transforms.matrix_to_quaternion(new_rot_mat)
            rotations = new_quat#[:, [3,0,1,2]]1
        elif pc_list[0].apply_sds_pose:
            w_tf = torch.einsum("bpn,bnij->bpij", lbs_weights, pc_list[0].sds_smplx_tfs)
            rot_mat = pytorch3d.transforms.quaternion_to_matrix(rotations)
            new_rot_mat = w_tf[0, :, :3, :3] @ rot_mat
            new_quat = pytorch3d.transforms.matrix_to_quaternion(new_rot_mat)
            rotations = new_quat#[:, [3,0,1,2]]1

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = torch.concat([pc.get_features for pc in pc_list])
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # rendered_image, radii = rasterizer(
    out  = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    if apply_2dgs:
        rendered_image, radii, allmap = out
        # additional regularizations for 2dgs
        render_alpha = allmap[1:2]

        # get normal map
        # transform normal from view space to world space
        render_normal = allmap[2:5]
        render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
        
        # get median depth map
        render_depth_median = allmap[5:6]
        render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

        # get expected depth map
        render_depth_expected = allmap[0:1]
        render_depth_expected = (render_depth_expected / render_alpha)
        render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
        
        # get depth distortion map
        render_dist = allmap[6:7]
        return {
            "render": rendered_image,
            'rend_dist': render_dist,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "depth_median": render_depth_median,
        }
    else:
        rendered_image, radii, rendered_depth, rendered_alpha = out
        return {
            "render": rendered_image,
            "depth": rendered_depth,
            "alpha": rendered_alpha,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
        }

def render_gala_controlent_inpaint(viewpoint_camera, pc : GaussianModel, freezed_pc: GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, inpaint=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    if inpaint:
        screenspace_points = torch.zeros_like(torch.concat([pc.get_xyz, freezed_pc.get_xyz]), dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    else:
        screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    if inpaint:
        means3D = torch.concat([pc.get_xyz, freezed_pc.get_xyz])
    else:
        means3D = pc.get_xyz

    # # means3D[:, 1] += 0.4    # add the yoffset
    # y_offset = torch.zeros_like(means3D, dtype=means3D.dtype, device=means3D.device).requires_grad_(False)
    # y_offset[:, 1] = 0.4
    # means3D = means3D + y_offset

    means2D = screenspace_points
    if inpaint:
        opacity = torch.concat([pc.get_opacity, freezed_pc.get_opacity])
    else:
        opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        if inpaint:
            scales = torch.concat([pc.get_scaling, freezed_pc.get_scaling])
            rotations = torch.concat([pc.get_rotation, freezed_pc.get_rotation])
        else:
            scales = pc.get_scaling
            rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if inpaint:
                shs = torch.concat([pc.get_features, freezed_pc.get_features])
            else:
                shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # rendered_image, radii = rasterizer(
    rendered_image, radii, rendered_depth, rendered_alpha  = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": rendered_image,
        "depth": rendered_depth,
        "alpha": rendered_alpha,
        "viewspace_points": screenspace_points,
        "visibility_filter" : radii > 0,
        "radii": radii,
    }


def render_multi_layer(viewpoint_camera, pc : GaussianModel, pc_garment : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, apply_pose = False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros((pc.get_xyz.shape[0] + pc_garment.get_xyz.shape[0], 3), dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # means3D = pc.get_xyz

    verts = torch.concat([pc.get_xyz, pc_garment.get_xyz], dim=0)
    offsets = torch.nn.functional.grid_sample(pc.shape_pose_offsets_grid , verts[None, None, None, :, :]).reshape(1, 3, -1).permute(0, 2, 1)
    lbs_weights = torch.nn.functional.grid_sample(pc.lbs_weights_grid, verts[None, None, None, :, :]).reshape(1, 55, -1).permute(0, 2, 1)
    verts = verts + offsets[0]  # shape pose blend shapes

    if apply_pose:
        means3D = helpers.skinning(verts, lbs_weights, pc.gt_smplx_tfs)[0]  # lbs
        yoffset = torch.zeros((means3D.shape[0], 3), dtype=torch.float, device='cuda')
        means3D = means3D + yoffset
    else:
        means3D = verts

    means2D = screenspace_points
    opacity = torch.concat([pc.get_opacity, pc_garment.get_opacity], dim=0)

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = torch.concat([pc.get_scaling, pc_garment.get_scaling], dim=0)
        rotations = torch.concat([pc.get_rotation, pc_garment.get_rotation], dim=0)
    

    rotations = torch.concatenate([pc._rotation, pc_garment._rotation], dim=0)#[:, [1,2,3,0]]      # 3dgs use wxyz for quaternion and scipy use xyzw for quaternion
    if apply_pose:
        w_tf = torch.einsum("bpn,bnij->bpij", lbs_weights, pc.gt_smplx_tfs)
        rot_mat = pytorch3d.transforms.quaternion_to_matrix(rotations)
        new_rot_mat = w_tf[0, :, :3, :3] @ rot_mat
        new_quat = pytorch3d.transforms.matrix_to_quaternion(new_rot_mat)
        rotations = new_quat#[:, [3,0,1,2]]


    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = torch.concat([pc.get_features, pc_garment.get_features], dim=0)
    else:
        colors_precomp = override_color

    
    try:
        # Rasterize visible Gaussians to image, obtain their radii (on screen). 
        rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        # import cv2; cv2.imwrite('debug/img_pred.png', rendered_image.permute(1,2,0).detach().cpu().numpy()[:,:,::-1]*255)

        body_shs = torch.zeros((pc.get_xyz.shape[0], 3), dtype=torch.float)
        body_shs[:, 0] = 1.0
        fused_color = RGB2SH(body_shs.cuda())
        body_features = torch.zeros((fused_color.shape[0], 3, (3 + 1) ** 2)).float().cuda()
        body_features[:, :3, 0 ] = fused_color
        body_features[:, 3:, 1:] = 0.0
        body_features = body_features.transpose(1, 2)
        
        garment_shs = torch.zeros((pc_garment.get_xyz.shape[0], 3), dtype=torch.float)
        garment_shs[:, 1] = 1.0
        fused_color = RGB2SH(garment_shs.cuda())
        garment_features = torch.zeros((fused_color.shape[0], 3, (3 + 1) ** 2)).float().cuda()
        garment_features[:, :3, 0 ] = fused_color
        garment_features[:, 3:, 1:] = 0.0
        garment_features = garment_features.transpose(1, 2)
        shs = torch.concat([body_features, garment_features], dim=0)

        rendered_image_obj, _, _, _ = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    except:
        raise Exception()

    # # Render body only
    # means3D = pc.get_xyz
    # sds_screenspace_points = torch.zeros((pc.get_xyz.shape[0], 3), dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    # try:
    #     sds_screenspace_points.retain_grad()
    # except:
    #     pass
    # means2D = sds_screenspace_points
    # opacity = pc.get_opacity

    # # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # # scaling / rotation by the rasterizer.
    # scales = None
    # rotations = None
    # cov3D_precomp = None
    # if pipe.compute_cov3D_python:
    #     cov3D_precomp = pc.get_covariance(scaling_modifier)
    # else:
    #     scales = pc.get_scaling
    #     rotations = pc.get_rotation
    
    # # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    # shs = None
    # colors_precomp = None
    # if override_color is None:
    #     if pipe.convert_SHs_python:
    #         shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
    #         dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
    #         dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
    #         sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
    #         colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    #     else:
    #         shs = pc.get_features
    # else:
    #     colors_precomp = override_color
    # image_tpose_body, tpose_radii, _, _ = rasterizer(
    #     means3D = means3D,
    #     means2D = means2D,
    #     shs = shs,
    #     colors_precomp = colors_precomp,
    #     opacities = opacity,
    #     scales = scales,
    #     rotations = rotations,
    #     cov3D_precomp = cov3D_precomp)
    
    # if render_garment:
    #     image_tpose_garment = render_garment_gs(rasterizer, viewpoint_camera, pc_garment, pipe, bg_color, scaling_modifier, override_color)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": rendered_image,
        "depth": rendered_depth,
        "viewspace_points": screenspace_points,
        "visibility_filter" : radii > 0,
        "radii": radii,
        "gau_size_body": pc.get_xyz.shape[0],
        "gau_size_garment": pc_garment.get_xyz.shape[0],
        "rendered_image_obj": rendered_image_obj,
        # "image_tpose_body": image_tpose_body,
        # "image_tpose_garment": image_tpose_garment if render_garment else None,
    }

def render_garment_gs(rasterizer, viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    means3D = pc.get_xyz
    sds_screenspace_points = torch.zeros((pc.get_xyz.shape[0], 3), dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        sds_screenspace_points.retain_grad()
    except:
        pass
    means2D = sds_screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
    
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color
    image, radii, _, _ = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    return image

def render_2dgs(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettingsSurfel(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
        # pipe.debug
    )

    rasterizer = GaussianRasterizerSurfel(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        # currently don't support normal consistency loss if use precomputed covariance
        splat2world = pc.get_covariance(scaling_modifier)
        W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
        near, far = viewpoint_camera.znear, viewpoint_camera.zfar
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W-1) / 2],
            [0, H / 2, 0, (H-1) / 2],
            [0, 0, far-near, near],
            [0, 0, 0, 1]]).float().cuda().T
        world2pix =  viewpoint_camera.full_proj_transform @ ndc2pix
        cov3D_precomp = (splat2world[:, [0,1,3]] @ world2pix[:,[0,1,3]]).permute(0,2,1).reshape(-1, 9) # column major
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
    
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    pipe.convert_SHs_python = False
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color
    
    rendered_image, radii, allmap = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp
    )
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rets =  {"render": rendered_image,
            "viewspace_points": means2D,
            "visibility_filter" : radii > 0,
            "radii": radii,
    }


    # additional regularizations
    render_alpha = allmap[1:2]

    # get normal map
    # transform normal from view space to world space
    render_normal = allmap[2:5]
    render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
    
    # get median depth map
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    # get expected depth map
    render_depth_expected = allmap[0:1]
    render_depth_expected = (render_depth_expected / render_alpha)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
    
    # get depth distortion map
    render_dist = allmap[6:7]

    # psedo surface attributes
    # surf depth is either median or expected by setting depth_ratio to 1 or 0
    # for bounded scene, use median depth, i.e., depth_ratio = 1; 
    # for unbounded scene, use expected depth, i.e., depth_ration = 0, to reduce disk anliasing.
    surf_depth = render_depth_expected * (1-pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median
    
    # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    surf_normal = surf_normal.permute(2,0,1)
    # remember to multiply with accum_alpha since render_normal is unnormalized.
    surf_normal = surf_normal * (render_alpha).detach()


    rets.update({
            'rend_alpha': render_alpha,
            'rend_normal': render_normal,
            'rend_dist': render_dist,
            'surf_depth': surf_depth,
            'surf_normal': surf_normal,
    })

    return rets

def render_multi_layer_2dgs(viewpoint_camera, pc : GaussianModel, pc_garment : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, apply_pose = False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros((pc.get_xyz.shape[0] + pc_garment.get_xyz.shape[0], 3), dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettingsSurfel(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizerSurfel(raster_settings=raster_settings)

    # means3D = pc.get_xyz

    verts = torch.concat([pc.get_xyz, pc_garment.get_xyz], dim=0)
    offsets = torch.nn.functional.grid_sample(pc.shape_pose_offsets_grid , verts[None, None, None, :, :]).reshape(1, 3, -1).permute(0, 2, 1)
    lbs_weights = torch.nn.functional.grid_sample(pc.lbs_weights_grid, verts[None, None, None, :, :]).reshape(1, 55, -1).permute(0, 2, 1)
    verts = verts + offsets[0]  # shape pose blend shapes

    if apply_pose:
        means3D = helpers.skinning(verts, lbs_weights, pc.gt_smplx_tfs)[0]  # lbs
        yoffset = torch.zeros((means3D.shape[0], 3), dtype=torch.float, device='cuda')
        means3D = means3D + yoffset
    else:
        means3D = verts

    means2D = screenspace_points
    opacity = torch.concat([pc.get_opacity, pc_garment.get_opacity], dim=0)

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = torch.concat([pc.get_scaling, pc_garment.get_scaling], dim=0)
        rotations = torch.concat([pc.get_rotation, pc_garment.get_rotation], dim=0)
    

    rotations = torch.concatenate([pc._rotation, pc_garment._rotation], dim=0)#[:, [1,2,3,0]]      # 3dgs use wxyz for quaternion and scipy use xyzw for quaternion
    if apply_pose:
        w_tf = torch.einsum("bpn,bnij->bpij", lbs_weights, pc.gt_smplx_tfs)
        rot_mat = pytorch3d.transforms.quaternion_to_matrix(rotations)
        new_rot_mat = w_tf[0, :, :3, :3] @ rot_mat
        new_quat = pytorch3d.transforms.matrix_to_quaternion(new_rot_mat)
        rotations = new_quat#[:, [3,0,1,2]]


    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = torch.concat([pc.get_features, pc_garment.get_features], dim=0)
    else:
        colors_precomp = override_color

    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, allmap = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    # import cv2; cv2.imwrite('debug/img_pred.png', rendered_image.permute(1,2,0).detach().cpu().numpy()[:,:,::-1]*255)
    results = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : radii > 0,
        "radii": radii,
    }


    # additional regularizations for 2dgs
    render_alpha = allmap[1:2]

    # get normal map
    # transform normal from view space to world space
    render_normal = allmap[2:5]
    render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
    
    # get median depth map
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    # get expected depth map
    render_depth_expected = allmap[0:1]
    render_depth_expected = (render_depth_expected / render_alpha)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
    
    # get depth distortion map
    render_dist = allmap[6:7]

    # psedo surface attributes
    # surf depth is either median or expected by setting depth_ratio to 1 or 0
    # for bounded scene, use median depth, i.e., depth_ratio = 1; 
    # for unbounded scene, use expected depth, i.e., depth_ration = 0, to reduce disk anliasing.
    surf_depth = render_depth_expected * (1-pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median
    
    # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    surf_normal = surf_normal.permute(2,0,1)
    # remember to multiply with accum_alpha since render_normal is unnormalized.
    surf_normal = surf_normal * (render_alpha).detach()
    results.update({
            'rend_alpha': render_alpha,
            'rend_normal': render_normal,
            'rend_dist': render_dist,
            'surf_depth': surf_depth,
            'surf_normal': surf_normal,
    })


    body_shs = torch.zeros((pc.get_xyz.shape[0], 3), dtype=torch.float)
    body_shs[:, 0] = 1.0
    fused_color = RGB2SH(body_shs.cuda())
    body_features = torch.zeros((fused_color.shape[0], 3, (3 + 1) ** 2)).float().cuda()
    body_features[:, :3, 0 ] = fused_color
    body_features[:, 3:, 1:] = 0.0
    body_features = body_features.transpose(1, 2)
    
    garment_shs = torch.zeros((pc_garment.get_xyz.shape[0], 3), dtype=torch.float)
    garment_shs[:, 1] = 1.0
    fused_color = RGB2SH(garment_shs.cuda())
    garment_features = torch.zeros((fused_color.shape[0], 3, (3 + 1) ** 2)).float().cuda()
    garment_features[:, :3, 0 ] = fused_color
    garment_features[:, 3:, 1:] = 0.0
    garment_features = garment_features.transpose(1, 2)
    shs = torch.concat([body_features, garment_features], dim=0)

    rendered_image_obj, _, _= rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    results.update({
        "gau_size_body": pc.get_xyz.shape[0],
        "gau_size_garment": pc_garment.get_xyz.shape[0],
        "rendered_image_obj": rendered_image_obj,
    })

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return results

def render_single_layer_2dgs(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, apply_pose = False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros((pc.get_xyz.shape[0], 3), dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettingsSurfel(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizerSurfel(raster_settings=raster_settings)

    # means3D = pc.get_xyz

    verts = torch.concat([pc.get_xyz], dim=0)
    offsets = torch.nn.functional.grid_sample(pc.shape_pose_offsets_grid , verts[None, None, None, :, :]).reshape(1, 3, -1).permute(0, 2, 1)
    lbs_weights = torch.nn.functional.grid_sample(pc.lbs_weights_grid, verts[None, None, None, :, :]).reshape(1, 55, -1).permute(0, 2, 1)
    verts = verts + offsets[0]  # shape pose blend shapes

    if apply_pose:
        means3D = helpers.skinning(verts, lbs_weights, pc.gt_smplx_tfs)[0]  # lbs
        yoffset = torch.zeros((means3D.shape[0], 3), dtype=torch.float, device='cuda')
        means3D = means3D + yoffset
    else:
        means3D = verts

    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
    

    rotations = pc._rotation#[:, [1,2,3,0]]      # 3dgs use wxyz for quaternion and scipy use xyzw for quaternion
    if apply_pose:
        w_tf = torch.einsum("bpn,bnij->bpij", lbs_weights, pc.gt_smplx_tfs)
        rot_mat = pytorch3d.transforms.quaternion_to_matrix(rotations)
        new_rot_mat = w_tf[0, :, :3, :3] @ rot_mat
        new_quat = pytorch3d.transforms.matrix_to_quaternion(new_rot_mat)
        rotations = new_quat#[:, [3,0,1,2]]


    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, allmap = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    # import cv2; cv2.imwrite('debug/img_pred.png', rendered_image.permute(1,2,0).detach().cpu().numpy()[:,:,::-1]*255)
    results = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : radii > 0,
        "radii": radii,
    }


    # additional regularizations for 2dgs
    render_alpha = allmap[1:2]

    # get normal map
    # transform normal from view space to world space
    render_normal = allmap[2:5]
    render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
    
    # get median depth map
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    # get expected depth map
    render_depth_expected = allmap[0:1]
    render_depth_expected = (render_depth_expected / render_alpha)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
    
    # get depth distortion map
    render_dist = allmap[6:7]

    # psedo surface attributes
    # surf depth is either median or expected by setting depth_ratio to 1 or 0
    # for bounded scene, use median depth, i.e., depth_ratio = 1; 
    # for unbounded scene, use expected depth, i.e., depth_ration = 0, to reduce disk anliasing.
    surf_depth = render_depth_expected * (1-pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median
    
    # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    surf_normal = surf_normal.permute(2,0,1)
    # remember to multiply with accum_alpha since render_normal is unnormalized.
    surf_normal = surf_normal * (render_alpha).detach()
    results.update({
            'rend_alpha': render_alpha,
            'rend_normal': render_normal,
            'rend_dist': render_dist,
            'surf_depth': surf_depth,
            'surf_normal': surf_normal,
    })


    results.update({
        "gau_size_body": pc.get_xyz.shape[0],
        # "gau_size_garment": pc_garment.get_xyz.shape[0],
        # "rendered_image_obj": rendered_image_obj,
    })

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return results



def render_three_layer_2dgs(viewpoint_camera, pc_garment : GaussianModel, pc : GaussianModel, pc_upper : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, apply_pose = False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros((pc_upper.get_xyz.shape[0] + pc.get_xyz.shape[0] + pc_garment.get_xyz.shape[0], 3), dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettingsSurfel(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizerSurfel(raster_settings=raster_settings)

    # means3D = pc.get_xyz

    verts = torch.concat([pc_garment.get_xyz, pc.get_xyz, pc_upper.get_xyz], dim=0)
    offsets = torch.nn.functional.grid_sample(pc.shape_pose_offsets_grid , verts[None, None, None, :, :]).reshape(1, 3, -1).permute(0, 2, 1)
    lbs_weights = torch.nn.functional.grid_sample(pc.lbs_weights_grid, verts[None, None, None, :, :]).reshape(1, 55, -1).permute(0, 2, 1)
    verts = verts + offsets[0]  # shape pose blend shapes

    if apply_pose:
        means3D = helpers.skinning(verts, lbs_weights, pc.gt_smplx_tfs)[0]  # lbs
        yoffset = torch.zeros((means3D.shape[0], 3), dtype=torch.float, device='cuda')
        means3D = means3D + yoffset
    else:
        means3D = verts

    means2D = screenspace_points
    opacity = torch.concat([pc_garment.get_opacity, pc.get_opacity, pc_upper.get_opacity], dim=0)

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = torch.concat([pc_garment.get_scaling, pc.get_scaling, pc_upper.get_scaling], dim=0)
        rotations = torch.concat([pc_garment.get_rotation, pc.get_rotation, pc_upper.get_rotation], dim=0)
    

    rotations = torch.concatenate([pc_garment._rotation, pc._rotation, pc_upper._rotation], dim=0)#[:, [1,2,3,0]]      # 3dgs use wxyz for quaternion and scipy use xyzw for quaternion
    if apply_pose:
        w_tf = torch.einsum("bpn,bnij->bpij", lbs_weights, pc.gt_smplx_tfs)
        rot_mat = pytorch3d.transforms.quaternion_to_matrix(rotations)
        new_rot_mat = w_tf[0, :, :3, :3] @ rot_mat
        new_quat = pytorch3d.transforms.matrix_to_quaternion(new_rot_mat)
        rotations = new_quat#[:, [3,0,1,2]]


    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = torch.concat([pc_garment.get_features, pc.get_features, pc_upper.get_features], dim=0)
    else:
        colors_precomp = override_color

    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, allmap = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    # import cv2; cv2.imwrite('debug/img_pred.png', rendered_image.permute(1,2,0).detach().cpu().numpy()[:,:,::-1]*255)
    results = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : radii > 0,
        "radii": radii,
    }


    # additional regularizations for 2dgs
    render_alpha = allmap[1:2]

    # get normal map
    # transform normal from view space to world space
    render_normal = allmap[2:5]
    render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
    
    # get median depth map
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    # get expected depth map
    render_depth_expected = allmap[0:1]
    render_depth_expected = (render_depth_expected / render_alpha)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
    
    # get depth distortion map
    render_dist = allmap[6:7]

    # psedo surface attributes
    # surf depth is either median or expected by setting depth_ratio to 1 or 0
    # for bounded scene, use median depth, i.e., depth_ratio = 1; 
    # for unbounded scene, use expected depth, i.e., depth_ration = 0, to reduce disk anliasing.
    surf_depth = render_depth_expected * (1-pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median
    
    # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    surf_normal = surf_normal.permute(2,0,1)
    # remember to multiply with accum_alpha since render_normal is unnormalized.
    surf_normal = surf_normal * (render_alpha).detach()
    results.update({
            'rend_alpha': render_alpha,
            'rend_normal': render_normal,
            'rend_dist': render_dist,
            'surf_depth': surf_depth,
            'surf_normal': surf_normal,
    })
    
    garment_shs = torch.zeros((pc_garment.get_xyz.shape[0], 3), dtype=torch.float)
    garment_shs[:, 1] = 1.0
    fused_color = RGB2SH(garment_shs.cuda())
    garment_features = torch.zeros((fused_color.shape[0], 3, (3 + 1) ** 2)).float().cuda()
    garment_features[:, :3, 0 ] = fused_color
    garment_features[:, 3:, 1:] = 0.0
    garment_features = garment_features.transpose(1, 2)

    body_shs = torch.zeros((pc.get_xyz.shape[0], 3), dtype=torch.float)
    body_shs[:, 0] = 1.0
    fused_color = RGB2SH(body_shs.cuda())
    body_features = torch.zeros((fused_color.shape[0], 3, (3 + 1) ** 2)).float().cuda()
    body_features[:, :3, 0 ] = fused_color
    body_features[:, 3:, 1:] = 0.0
    body_features = body_features.transpose(1, 2)

    upper_shs = torch.zeros((pc_upper.get_xyz.shape[0], 3), dtype=torch.float)
    upper_shs[:, 1] = 1.0
    fused_color = RGB2SH(upper_shs.cuda())
    upper_features = torch.zeros((fused_color.shape[0], 3, (3 + 1) ** 2)).float().cuda()
    upper_features[:, :3, 0 ] = fused_color
    upper_features[:, 3:, 1:] = 0.0
    upper_features = upper_features.transpose(1, 2)

    shs = torch.concat([garment_features, body_features, upper_features], dim=0)

    rendered_image_obj, _, _= rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    results.update({
        "gau_size_upper": pc_upper.get_xyz.shape[0],
        "gau_size_body": pc.get_xyz.shape[0],
        "gau_size_garment": pc_garment.get_xyz.shape[0],
        "rendered_image_obj": rendered_image_obj,
    })

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return results

def render_gaussian_list_on_mesh(viewpoint_camera, pc_list : list[GaussianModel], pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(torch.concat([pc.get_xyz for pc in pc_list]), dtype=pc_list[0].get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    GRSFunc = GaussianRasterizationSettingsSurfel
    RasterizerFunc = GaussianRasterizerSurfel

    raster_settings = GRSFunc(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc_list[0].active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = RasterizerFunc(raster_settings=raster_settings)

    means3D = torch.concat([pc.get_xyz for pc in pc_list])

    means2D = screenspace_points
    opacity = torch.concat([pc.get_opacity for pc in pc_list])

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = torch.concat([pc.get_covariance(scaling_modifier) for pc in pc_list])
    else:
        scales = torch.concat([pc.get_scaling for pc in pc_list])
        rotations = torch.concat([pc.get_rotation for pc in pc_list])

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = torch.concat([pc.get_features for pc in pc_list])
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # rendered_image, radii = rasterizer(
    out  = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image, radii, allmap = out
    # additional regularizations for 2dgs
    render_alpha = allmap[1:2]

    # get normal map
    # transform normal from view space to world space
    render_normal = allmap[2:5]
    render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
    
    # get median depth map
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    # get expected depth map
    render_depth_expected = allmap[0:1]
    render_depth_expected = (render_depth_expected / render_alpha)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
    
    # get depth distortion map
    render_dist = allmap[6:7]
    return {
        "render": rendered_image,
        'rend_dist': render_dist,
        "viewspace_points": screenspace_points,
        "visibility_filter" : radii > 0,
        "radii": radii,
        "depth_median": render_depth_median,
    }