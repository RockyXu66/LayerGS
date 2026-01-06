"""
Adapted from https://github.com/snuvclab/gala/blob/main/utils/normalize_mesh.py
"""

import open3d as o3d
import numpy as np
import argparse
import os
import pickle
from pathlib import Path

def find_startswith(file_content, target):
    # Find the line that starts with 'mtllib'
    for line in file_content.split('\n'):
        if line.startswith(target):
            return line

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--mesh_dir", default="data/rp/rp_christopher_posed_008", help="mesh obj")
parser.add_argument("--dataset_type", default='rp')
parser.add_argument("--norm_with_smplx", action='store_true')
args = parser.parse_args()

mesh_dir = Path(args.mesh_dir)
dataset_type = args.dataset_type

if dataset_type == 'rp':
    mesh_name = mesh_dir.stem + '_100k'
elif dataset_type == 'tada':
    mesh_name = 'mesh'
else:
    mesh_name = f'{mesh_dir.stem}'
    
import trimesh
label_ply_path = str(mesh_dir / f'{mesh_name}-in-origin-label.ply')
if os.path.exists(label_ply_path):
    apply_label = True
else:
    apply_label = False

if apply_label:
    mesh_label = trimesh.load(label_ply_path)
    vertices_label = np.asarray(mesh_label.vertices)

# if not yet normlized with smplx scale and translation
if args.norm_with_smplx:
    if 'thuman' in dataset_type:
        mesh = o3d.io.read_triangle_mesh(str(mesh_dir / f'{mesh_name}.obj'))
        vertices = np.asarray(mesh.vertices)
        if dataset_type == 'thuman20':  # dict keys are slightly different for given thuman smplx param files
            smplx_params = np.load(mesh_dir / 'thuman20_smplx_param.pkl', allow_pickle=True)
            transl = smplx_params['translation']
            scale = smplx_params['scale']
            vertices = (vertices - transl) / scale
        elif dataset_type == 'thuman21':
            smplx_params = np.load(mesh_dir / 'thuman21_smplx_param.pkl', allow_pickle=True)
            transl = smplx_params['transl']
            scale = smplx_params['scale']
            vertices = (vertices / scale) - transl
        # else:
        #     transl = smplx_params['global_body_translation']
        #     scale = smplx_params['body_scale']


        vertices[:, 1] += 0.4 # shift all ys by 0.4
    elif '4d-dress' in dataset_type:
        mesh = o3d.io.read_triangle_mesh(str(mesh_dir / f'{mesh_name}-in-origin.obj'))
        vertices = np.asarray(mesh.vertices)
        Y_OFFSET = 0.4
        smplx_params = np.load(mesh_dir / f'{mesh_name}-smplx.pkl', allow_pickle=True)
        # transl = smplx_params['transl']
        # vertices = vertices - transl # the translation offset has been removed when saving original obj file
        vertices[:, 1] += Y_OFFSET
        if apply_label:
            vertices_label[:, 1] += Y_OFFSET

        
        import torch
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from deformer.smplx import SMPLX
        class Struct:
            def __init__(self, **entries):
                self.__dict__.update(entries)
        smplx_config = {
            'topology_path': "deformer/data/SMPL_X_template_FLAME_uv.obj",
            # 'smplx_model_path': "deformer/data/SMPLX_NEUTRAL_2020.npz",
            'smplx_model_path': 'data/models/smplx/SMPLX_MALE.npz',
            'extra_joint_path': "deformer/data/smplx_extra_joints.yaml",
            'j14_regressor_path': "deformer/data/SMPLX_to_J14.pkl",
            'mano_ids_path': "deformer/data/MANO_SMPLX_vertex_ids.pkl",
            'flame_vertex_masks_path': "deformer/data/FLAME_masks.pkl",
            'flame_ids_path': "deformer/data/SMPL-X__FLAME_vertex_ids.npy",
            'n_shape': 10,
            'n_exp': 10
        }
        smplx_config = Struct(**smplx_config)

        smplx_params = {k: torch.from_numpy(v).float() for k, v in smplx_params.items()}
        smplx_model = SMPLX(smplx_config)
        beta = smplx_params['betas'].squeeze()[:10]
        body_pose = smplx_params['body_pose'] if smplx_params['body_pose'].shape[1] == 63 else smplx_params['body_pose'][:, 3:]
        body_pose = body_pose.to(torch.float32) # prevent type confliction
        full_pose = torch.cat([smplx_params['global_orient'], body_pose,
                            smplx_params['jaw_pose'], smplx_params['leye_pose'], smplx_params['reye_pose'],  
                            smplx_params['left_hand_pose'][:, :6], smplx_params['right_hand_pose'][:, :6]], dim=1)
        exp = smplx_params['expression'].squeeze()[:10]
        xyz, _, joints, A, T, shape_offsets, pose_offsets = smplx_model(full_pose=full_pose[None], shape_params=beta[None], return_T=True, 
                                                                  transl=torch.tensor([0, Y_OFFSET, 0],dtype=torch.float32)[None],
                                                                  expression_params=exp[None], axis_pca=True)
        from pytorch3d.structures import Meshes
        from pytorch3d.io import save_obj
        smplx_mesh = Meshes(verts=[xyz[0]], faces=[smplx_model.faces_tensor])
        save_obj(str(mesh_dir / f'{mesh_name}_smplx.obj'), xyz[0], smplx_model.faces_tensor)
        aa = 10

# NOTE: trimesh export does not allow the different number of vertices and uv coordinates, which leads to artifacts.
mesh.vertices = o3d.utility.Vector3dVector(vertices)
o3d.io.write_triangle_mesh(str(mesh_dir / f'{mesh_name}_norm.obj'), mesh)

if apply_label:
    mesh_label.vertices = vertices_label
    mesh_label.export(str(mesh_dir / f'{mesh_name}_norm-label.ply'))

if os.path.exists(f'{mesh_dir}/{mesh_name}_norm.mtl'):
    os.remove(f'{mesh_dir}/{mesh_name}_norm.mtl')
        
# update normalized obj file to refer to original mtl file
if os.path.exists(mesh_dir / f'{mesh_name}-in-origin.obj'):
    with open(mesh_dir / f'{mesh_name}-in-origin.obj', 'r') as obj_orig:
        mtllib_line_orig = find_startswith(obj_orig.read(), 'mtllib')
else:
    with open(mesh_dir / f'{mesh_name}.obj', 'r') as obj_orig:
        mtllib_line_orig = find_startswith(obj_orig.read(), 'mtllib')

mtl_file = mtllib_line_orig.split()[-1]  # ex. "mtllib material0.mtl"
with open(mesh_dir / mtl_file, 'r') as mtl:
    material_name = find_startswith(mtl.read(), 'newmtl').split()[-1]

mtl_lines = mtllib_line_orig + f'\nusemtl {material_name}'

with open(mesh_dir / f'{mesh_name}_norm.obj', 'r') as obj_norm:
    obj_norm_content = obj_norm.read()
    mtllib_line_norm = find_startswith(obj_norm_content, 'mtllib')

obj_norm_content_updated = obj_norm_content.replace(mtllib_line_norm, mtl_lines)

with open(mesh_dir / f'{mesh_name}_norm.obj', 'w') as obj_norm:
    obj_norm.write(obj_norm_content_updated)
