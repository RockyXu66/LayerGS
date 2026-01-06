#!/usr/bin/env python
"""
Generate Colmap format sparse files (images.bin, cameras.bin) from torch3d_cam_list.pkl.

Usage:
    python -m scripts.generate_colmap_from_torch3d \
        --data_path /path/to/data \
        --output_folder sparse_from_torch3d \
        --scale 1.9  # Scale camera positions to match original Colmap scale

This will create:
    /path/to/data/sparse_from_torch3d/0/images.bin
    /path/to/data/sparse_from_torch3d/0/cameras.bin
"""

import os
import argparse
import pickle
import numpy as np
from pathlib import Path

from scipy.spatial.transform import Rotation
from data_preprocess.colmap_utils import (
    write_extrinsics_binary,
    write_intrinsics_binary,
    Image,
    Camera,
)


def convert_torch3d_cam_list_to_colmap(torch3d_cam_list, scale=1.0):
    """
    Convert torch3d_cam_list.pkl format to Colmap format.
    
    Args:
        torch3d_cam_list: dict loaded from torch3d_cam_list.pkl
        scale: Scale factor for camera positions (to match original Colmap scale)
        
    Returns:
        cam_extrinsics: dict of Image namedtuples (Colmap extrinsics)
        cam_intrinsics: dict of Camera namedtuples (Colmap intrinsics)
    """
    cam_extrinsics = {}
    cam_intrinsics = {}
    
    first_key = list(torch3d_cam_list.keys())[0]
    first_cam = torch3d_cam_list[first_key]
    
    # Get intrinsics from first camera (assuming all cameras have same intrinsics)
    intr = first_cam['intrinsics']
    fx, fy = intr[0, 0], intr[1, 1]
    cx, cy = intr[0, 2], intr[1, 2]
    height, width = int(cy * 2), int(cx * 2)
    
    cam_intrinsics[1] = Camera(
        id=1,
        model='PINHOLE',
        height=height,
        width=width,
        params=np.array([fx, fy, cx, cy])
    )
    
    for k, cam in torch3d_cam_list.items():
        idx = int(k)
        ext = cam['extrinsics']  # 3x4 matrix [R|T] in pytorch3d style (row-major w2c)
        R = ext[:3, :3]  # Row-major rotation (w2c)
        T = ext[:3, 3]
        
        # pytorch3d extrinsics is w2c matrix in row-major format
        # Convert from row-major to column-major
        R_col = R.T
        w2c = np.eye(4)
        w2c[:3, :3] = R_col
        w2c[:3, 3] = T
        c2w = np.linalg.inv(w2c)
        
        # Apply scale to camera position
        c2w[:3, 3] *= scale
        
        # Convert from pytorch3d's LUF to Colmap's RDF coordinate system
        R_c2w = c2w[:3, :3]
        T_c2w = c2w[:3, 3]
        R_c2w_rdf = np.stack([-R_c2w[:, 0], -R_c2w[:, 1], R_c2w[:, 2]], axis=1)
        c2w_rdf = np.eye(4)
        c2w_rdf[:3, :3] = R_c2w_rdf
        c2w_rdf[:3, 3] = T_c2w
        w2c_rdf = np.linalg.inv(c2w_rdf)
        
        # Convert to quaternion (Colmap uses wxyz format)
        qvec_scipy = Rotation.from_matrix(w2c_rdf[:3, :3]).as_quat()  # xyzw
        qvec_colmap = np.array([qvec_scipy[3], qvec_scipy[0], qvec_scipy[1], qvec_scipy[2]])  # wxyz
        tvec = w2c_rdf[:3, 3]
        
        image_name = cam.get('name', f'{idx:08d}.png')
        cam_extrinsics[idx + 1] = Image(
            id=idx + 1,
            qvec=qvec_colmap,
            tvec=tvec,
            camera_id=1,
            name=image_name,
            xys=np.array([[0, 0]]),
            point3D_ids=np.array([0])
        )
    
    return cam_extrinsics, cam_intrinsics


def main():
    parser = argparse.ArgumentParser(description="Generate Colmap format files from torch3d_cam_list.pkl")
    parser.add_argument("--data_path", type=str, required=True, help="Path to data folder containing torch3d_cam_list.pkl")
    parser.add_argument("--output_folder", type=str, default="sparse_from_torch3d", help="Output folder name under data_path")
    parser.add_argument("--scale", type=float, default=1.9, help="Scale factor for camera positions (default: 1.9 to match typical Colmap scale)")
    args = parser.parse_args()
    
    data_path = Path(args.data_path)
    torch3d_cam_path = data_path / "torch3d_cam_list.pkl"
    
    if not torch3d_cam_path.exists():
        raise FileNotFoundError(f"torch3d_cam_list.pkl not found at {torch3d_cam_path}")
    
    print(f"Loading {torch3d_cam_path}...")
    torch3d_cam_list = pickle.load(open(torch3d_cam_path, 'rb'))
    print(f"Found {len(torch3d_cam_list)} cameras")
    print(f"Using scale factor: {args.scale}")
    
    # Convert to Colmap format with scaling
    print("Converting to Colmap format...")
    cam_extrinsics, cam_intrinsics = convert_torch3d_cam_list_to_colmap(torch3d_cam_list, scale=args.scale)
    
    # Create output directory
    output_dir = data_path / args.output_folder / "0"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write binary files
    images_bin_path = output_dir / "images.bin"
    cameras_bin_path = output_dir / "cameras.bin"
    
    print(f"Writing {images_bin_path}...")
    write_extrinsics_binary(str(images_bin_path), cam_extrinsics)
    
    print(f"Writing {cameras_bin_path}...")
    write_intrinsics_binary(str(cameras_bin_path), cam_intrinsics)
    
    # Print camera statistics
    positions = []
    for k, img in cam_extrinsics.items():
        from layergs.data.colmap_loader import qvec2rotmat
        R = qvec2rotmat(img.qvec)
        T = np.array(img.tvec)
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = T
        c2w = np.linalg.inv(w2c)
        positions.append(c2w[:3, 3])
    
    positions = np.array(positions)
    avg_distance = np.linalg.norm(positions, axis=1).mean()
    
    print(f"\nCamera statistics:")
    print(f"  X range: [{positions[:,0].min():.4f}, {positions[:,0].max():.4f}]")
    print(f"  Y range: [{positions[:,1].min():.4f}, {positions[:,1].max():.4f}]")
    print(f"  Z range: [{positions[:,2].min():.4f}, {positions[:,2].max():.4f}]")
    print(f"  Distance to origin: [{np.linalg.norm(positions, axis=1).min():.4f}, {np.linalg.norm(positions, axis=1).max():.4f}]")
    print(f"  Average distance: {avg_distance:.4f}")
    print(f"  Expected bounding radius: ~{avg_distance:.2f}")
    
    print(f"\nDone! Colmap sparse files saved to: {output_dir}")

if __name__ == "__main__":
    main()
