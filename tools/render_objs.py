# Adapted from https://github.com/snuvclab/gala/blob/main/segmentation/render.py
# Mesh rendering code
# Mesh is normalized such that its smplx scale is 1 and smplx transl is 0.

import os
import argparse
import cv2
import torch
import pickle
import nvdiffrast.torch as dr
import numpy as np
from pathlib import Path

from pytorch3d.io import load_obj
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--mesh_dir", default="data/rp/rp_christopher_posed_008", help="folder directory where mesh obj file exists")
parser.add_argument("-d", "--device", default="cuda:0")
parser.add_argument("--use_opengl", action="store_true")
parser.add_argument("--mesh_name", type=str, default="fuse_post_comp.obj", help="name of the mesh file to render")
args = parser.parse_args()

data_dir = args.mesh_dir
# out_dir = data_dir + '/render'
device = args.device
use_opengl = args.use_opengl

# res = 1024
res = 512

# mesh_names = sorted(os.listdir(data_dir))
# mesh_names = [m for m in mesh_names if "_norm.obj" in m]

mesh_name = args.mesh_name
mesh_names = [mesh_name]
# mesh_names = [f'fuse_post_comp_wshape.obj']
# mesh_names = [f'fuse_post_comp_posed.obj']

print(mesh_names)
for mesh_name in tqdm(mesh_names):

    out_dir = Path(f'{data_dir}/{mesh_name}').parent

    os.makedirs(f'{out_dir}/segm_images', exist_ok=True)

    verts, faces, aux = load_obj(f'{data_dir}/{mesh_name}', device=device)
    uv = aux.verts_uvs
    uv[:, 1] = 1 - uv[:, 1]  # flip v coordinate to match opengl convention
    uv_idx = faces.textures_idx.to(torch.int32)
    tex = list(aux.texture_images.values())[0].to(device)  # dealing with arbitrary key name for texture

    # pix_to_faces = {}
    # ipdb.set_trace()

    glctx = dr.RasterizeGLContext() if use_opengl else dr.RasterizeCudaContext()
    # body
    for x in range(150, 211, 20):
        for y in tqdm(range(0, 360, 20)):
    # for x in range(145, 206, 20):
    #     for y in tqdm(range(-10, 350, 20)):
    # for x in range(150, 211, 10):
    #     for y in tqdm(range(0, 360, 10)):
    # for x in range(180, 190, 10):
        # for y in tqdm(range(180, 190, 10)):
            r_xpi = R.from_euler('x', x, degrees=True).as_matrix()
            r_ypi = R.from_euler('y', y, degrees=True).as_matrix()
            r = r_xpi @ r_ypi

            transformed_vertices = (torch.tensor(r, device=device).float() @ (verts.T)).T
            transformed_vertices_h = torch.cat([transformed_vertices, torch.ones_like(transformed_vertices[:, 0:1])], axis=1)
            
            rast, rast_out_db = dr.rasterize(glctx, transformed_vertices_h[None], faces.verts_idx.int(), 
                                             resolution=np.array([res, res]))
            
            texc, _ = dr.interpolate(uv[None, ...], rast, uv_idx)
            color = dr.texture(tex[None, ...], texc, filter_mode='linear')
            color = color * torch.clamp(rast[..., -1:], 0, 1)  # Mask out background

            transforms.ToPILImage()(color[0].permute(2, 0, 1)).save(f'{out_dir}/segm_images/{x:03d}_{y:03d}.png')

