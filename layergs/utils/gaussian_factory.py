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
Factory functions for creating Gaussian models from various sources.
"""

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import trimesh
import open3d as o3d

from layergs.scene.gaussian_model import GaussianModel


def create_gaussians_from_mesh(
    mesh_path: str, 
    sample_num: int, 
    sh_degree: int,
    object_idx: int = 3
) -> GaussianModel:
    """
    Create Gaussian model by sampling points from a mesh.
    
    Args:
        mesh_path: Path to the mesh file
        sample_num: Number of points to sample
        sh_degree: Spherical harmonics degree
        object_idx: Object index for labeling
        
    Returns:
        Initialized GaussianModel
    """
    mesh = trimesh.load(mesh_path)
    sampled_points, _ = trimesh.sample.sample_surface(mesh, sample_num)
    vertices = torch.from_numpy(sampled_points).float().cuda()

    gaussians = GaussianModel(sh_degree, apply_2dgs=True)
    gaussians.create_from_xyz(vertices, object_idx=object_idx)
    return gaussians


def create_gaussians_from_ply(
    mesh_path: str, 
    ply_path: str, 
    sh_degree: int,
    method: int = 1,
    object_idx: int = 3
) -> GaussianModel:
    """
    Create Gaussian model from a PLY file, filtered by proximity to a mesh.
    
    Args:
        mesh_path: Path to the reference mesh for filtering
        ply_path: Path to the PLY file containing Gaussians
        sh_degree: Spherical harmonics degree
        method: Filtering method (1 = KDTree only, 2 = KDTree + barycentric)
        object_idx: Object index for labeling
        
    Returns:
        Filtered GaussianModel
    """
    if method == 1:
        return _create_from_ply_kdtree(mesh_path, ply_path, sh_degree, object_idx)
    elif method == 2:
        return _create_from_ply_barycentric(mesh_path, ply_path, sh_degree, object_idx)
    else:
        raise ValueError(f"Unknown method: {method}. Use 1 or 2.")


def _create_from_ply_kdtree(
    mesh_path: str, 
    ply_path: str, 
    sh_degree: int,
    object_idx: int,
    threshold: float = 0.015
) -> GaussianModel:
    """Filter Gaussians using KDTree distance threshold."""
    gaussians = GaussianModel(sh_degree=sh_degree, apply_2dgs=True)
    gaussians.load_ply(ply_path)

    points = gaussians._xyz.detach().cpu().numpy()
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    
    # Create KDTree from mesh samples (use seed for reproducibility)
    o3d.utility.random.seed(0)
    mesh_tree = o3d.geometry.KDTreeFlann(
        mesh.sample_points_uniformly(number_of_points=500000)
    )

    indices_within_threshold = []
    for i, point in enumerate(points):
        [_, idx, dists] = mesh_tree.search_knn_vector_3d(point, 1)
        if np.sqrt(dists[0]) < threshold:
            indices_within_threshold.append(i)

    indices = np.array(indices_within_threshold)
    _filter_gaussians_by_indices(gaussians, indices)
    
    gaussians.init_obj(object_idx=object_idx)
    gaussians.init_other()
    return gaussians


def _create_from_ply_barycentric(
    mesh_path: str, 
    ply_path: str, 
    sh_degree: int,
    object_idx: int,
    threshold: float = 0.03
) -> GaussianModel:
    """Filter Gaussians using KDTree + barycentric coordinate check."""
    gaussians = GaussianModel(sh_degree=sh_degree, apply_2dgs=True)
    gaussians.load_ply(ply_path)

    points = gaussians._xyz.detach().cpu().numpy()
    mesh_o3d = o3d.io.read_triangle_mesh(mesh_path)
    
    # First pass: KDTree filtering (use seed for reproducibility)
    o3d.utility.random.seed(0)
    mesh_tree = o3d.geometry.KDTreeFlann(
        mesh_o3d.sample_points_uniformly(number_of_points=500000)
    )

    indices_pass1 = []
    for i, point in enumerate(points):
        [_, idx, dists] = mesh_tree.search_knn_vector_3d(point, 1)
        if np.sqrt(dists[0]) < threshold:
            indices_pass1.append(i)

    points = points[indices_pass1]
    indices_pass1 = np.array(indices_pass1)
    _filter_gaussians_by_indices(gaussians, indices_pass1)

    # Second pass: Barycentric filtering
    mesh = trimesh.load(mesh_path)
    from trimesh.proximity import ProximityQuery
    
    pq = ProximityQuery(mesh)
    
    # Process in chunks
    chunk_size = 10000
    closest_list, distance_list, triangle_list = [], [], []
    
    for i in tqdm(range(0, points.shape[0], chunk_size), desc="Computing proximity"):
        pts_chunk = points[i:i+chunk_size]
        c_chunk, d_chunk, tri_chunk = pq.on_surface(pts_chunk)
        closest_list.append(c_chunk)
        distance_list.append(d_chunk)
        triangle_list.append(tri_chunk)
    
    distance = np.concatenate(distance_list, axis=0)
    triangle_id = np.concatenate(triangle_list, axis=0)
    
    indices_pass2 = []
    for i, (pt, d, tri_id) in enumerate(tqdm(
        zip(points, distance, triangle_id), 
        total=len(points),
        desc="Barycentric filtering"
    )):
        if d < threshold:
            tri = mesh.triangles[tri_id]
            v0, v1, v2 = tri
            
            # Compute plane normal
            normal = np.cross(v1 - v0, v2 - v0)
            normal = normal / np.linalg.norm(normal)
            
            # Project point onto plane
            pt_proj = pt - np.dot(pt - v0, normal) * normal

            # Compute barycentric coordinates
            v0v1 = v1 - v0
            v0v2 = v2 - v0
            v0pt = pt_proj - v0

            dot00 = np.dot(v0v2, v0v2)
            dot01 = np.dot(v0v2, v0v1)
            dot02 = np.dot(v0v2, v0pt)
            dot11 = np.dot(v0v1, v0v1)
            dot12 = np.dot(v0v1, v0pt)
            invDenom = 1.0 / (dot00 * dot11 - dot01 * dot01)
            u = (dot11 * dot02 - dot01 * dot12) * invDenom
            v = (dot00 * dot12 - dot01 * dot02) * invDenom

            # Check if inside triangle
            if (u >= 0) and (v >= 0) and (u + v <= 1):
                indices_pass2.append(i)

    indices_pass2 = np.array(indices_pass2)
    _filter_gaussians_by_indices(gaussians, indices_pass2)

    gaussians.init_obj(object_idx=object_idx)
    gaussians.init_other()
    return gaussians


def _filter_gaussians_by_indices(gaussians: GaussianModel, indices: np.ndarray):
    """Filter Gaussian parameters by indices."""
    gaussians._xyz = nn.Parameter(
        gaussians._xyz[indices].clone().detach().to(dtype=torch.float, device="cuda").requires_grad_(True)
    )
    gaussians._features_dc = nn.Parameter(
        gaussians._features_dc[indices].clone().detach().to(dtype=torch.float, device="cuda").requires_grad_(True)
    )
    gaussians._features_rest = nn.Parameter(
        gaussians._features_rest[indices].clone().detach().to(dtype=torch.float, device="cuda").requires_grad_(True)
    )
    gaussians._opacity = nn.Parameter(
        gaussians._opacity[indices].clone().detach().to(dtype=torch.float, device="cuda").requires_grad_(True)
    )
    gaussians._scaling = nn.Parameter(
        gaussians._scaling[indices].clone().detach().to(dtype=torch.float, device="cuda").requires_grad_(True)
    )
    gaussians._rotation = nn.Parameter(
        gaussians._rotation[indices].clone().detach().to(dtype=torch.float, device="cuda").requires_grad_(True)
    )


def load_pretrained_inner_gaussians(
    pretrained_folder: str,
    sh_degree: int,
    sds_controlnet,
    object_idx: int = 6
) -> GaussianModel:
    """
    Load pretrained inner body Gaussians.
    
    Args:
        pretrained_folder: Path to pretrained checkpoint folder
        sh_degree: Spherical harmonics degree
        sds_controlnet: GALA controller for skinning weights
        object_idx: Object index for labeling
        
    Returns:
        Loaded GaussianModel with skinning weights attached
    """
    gaussians = GaussianModel(sh_degree=sh_degree, apply_2dgs=True)
    gaussians.load_ply(f'{pretrained_folder}/point_cloud_body_cano.ply')
    gaussians.init_obj(object_idx=object_idx)
    gaussians.init_other()
    gaussians.shape_pose_offsets_grid = sds_controlnet.shape_pose_offsets_grid
    gaussians.lbs_weights_grid = sds_controlnet.lbs_weights_grid
    gaussians.gt_smplx_tfs = sds_controlnet.gt_smplx_tfs
    return gaussians

