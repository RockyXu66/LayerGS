# Mesh rendering code
# Mesh is normalized such that its smplx scale is 1 and smplx transl is 0.

import os
import argparse
import pickle
import numpy as np
import shutil
import math
from pathlib import Path

import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.io import imread
from PIL import Image
import trimesh

import torch
import torch.nn as nn
from pytorch3d.io import load_obj
from scipy.spatial.transform import Rotation as R
from torchvision import transforms
# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj, load_ply, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
from pytorch3d.renderer.blending import hard_rgb_blend, BlendParams

import open3d as o3d

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)

import colmap_utils

# add path for demo utils functions 
import sys
import os
sys.path.append(os.path.abspath(''))

# Setup the device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

def pytorch3d_camera(cam_list, H, W, focal_length_y, focal_length_x, R, T, img_idx, image_name):
    intrinsics = np.eye(3)
    intrinsics[0, 0] = H / 2 * focal_length_x
    intrinsics[1, 1] = W / 2 * focal_length_y
    intrinsics[0, 2] = W / 2
    intrinsics[1, 2] = H / 2

    extrinsics = np.eye(4)
    extrinsics[:3, :3] = R
    extrinsics[:3, 3] = T

    cam_list[f'{img_idx:04d}'] = {
        'intrinsics': intrinsics,
        'extrinsics': extrinsics[:3],
    }


class SimpleShader(nn.Module):
    def __init__(self, device="cpu", blend_params=None):
        super().__init__()
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        blend_params = kwargs.get("blend_params", self.blend_params)
        texels = meshes.sample_textures(fragments)
        images = hard_rgb_blend(texels, fragments, blend_params)
        return images  # (N, H, W, 3) RGBA image

class MeshRendererWithDepth(nn.Module):
    def __init__(self, rasterizer, shader):
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader

    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(meshes_world, **kwargs)
        images = self.shader(fragments, meshes_world, **kwargs)
        return images, fragments.zbuf

def load_mesh_single_texture(obj_filename, normalize_verts=False):
    mesh = load_objs_as_meshes([obj_filename], device=device)

    verts = mesh.verts_packed()
    faces = mesh.faces_packed()
    textures = mesh.textures

    # if normalize_verts:
    #     verts = center_normalize_verts(verts)
    mesh = Meshes(verts=[verts], faces=[faces], textures=mesh.textures)
    return mesh

def load_mesh_from_ply(ply_filename):

    # Load the PLY file
    ply = o3d.io.read_triangle_mesh(ply_filename)
    vertices = torch.tensor(ply.vertices, dtype=torch.float32)
    colors = torch.tensor(ply.vertex_colors, dtype=torch.float32)
    if len(colors) == 0:
        colors = torch.ones_like(vertices)
        colors[:] = 0.6

    # You might need to convert face data from open3d to torch tensor if your PLY file has faces
    faces = torch.tensor(ply.triangles, dtype=torch.int64)

    vertices = vertices.to(device)
    colors = colors.to(device)
    faces = faces.to(device)

    # Create a PyTorch3D mesh
    mesh = Meshes(verts=[vertices], faces=[faces], textures=None)

    # Apply vertex colors as textures
    mesh.textures = TexturesVertex(verts_features=[colors])
    return mesh

# load data from pkl_dir
def load_pickle(pkl_dir):
    return pickle.load(open(pkl_dir, "rb"))

def load_mesh_from_pkl(scan_mesh_fn):
    """
    Load pkl file based on official 4d dress script
    """
    # load scan_mesh with vertex colors
    scan_mesh = load_pickle(scan_mesh_fn)
    scan_mesh['uv_path'] = scan_mesh_fn.replace('mesh-f', 'atlas-f')
    if 'colors' not in scan_mesh:
        # load atlas data
        atlas_data = load_pickle(scan_mesh['uv_path'])
        # load scan uv_coordinate and uv_image as TextureVisuals
        uv_image = Image.fromarray(atlas_data).transpose(method=Image.Transpose.FLIP_TOP_BOTTOM).convert("RGB")
        texture_visual = trimesh.visual.texture.TextureVisuals(uv=scan_mesh['uvs'], image=uv_image)
        # pack scan data as trimesh
        scan_trimesh = trimesh.Trimesh(
            vertices=scan_mesh['vertices'],
            faces=scan_mesh['faces'],
            vertex_normals=scan_mesh['normals'],
            visual=texture_visual,
            process=False,
        )
        scan_mesh['colors'] = scan_trimesh.visual.to_color().vertex_colors
    return scan_mesh

def render_from_pytorch3d():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--mesh_path", default="", help="mesh path")
    parser.add_argument("--test", action="store_true", help="test mode")
    parser.add_argument("--rotation", action="store_true", help="rotation camera mode")
    parser.add_argument("--model_type", default="obj")
    parser.add_argument("--bg_color", type=lambda s: tuple(map(float, s.split(','))), default="0.0,0.0,0.0", help="Background color as R,G,B tuple in range 0.0-1.0")
    args = parser.parse_args()

    if not hasattr(args, 'bg_color'):
        args.bg_color = (0.0/255.0, 0.0/255.0, 0.0/255.0)          # black
        # args.bg_color = (160.0/255.0, 160.0/255.0, 160.0/255.0)  # grey

    mesh_dir = Path(args.mesh_path).parent

    # Setup
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    
    model_type = args.model_type

    if model_type == 'ply':
        # Load ply file
        mesh = load_mesh_from_ply(args.mesh_path)
    elif model_type == 'obj':
        # Set paths
        obj_filename = args.mesh_path
        mesh = load_objs_as_meshes([obj_filename], device=device)

        verts = mesh._verts_list[0]
        faces = mesh._faces_list[0]
        # Initialize each vertex to be white in color.
        verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
        textures = TexturesVertex(verts_features=verts_rgb.to(device))

        # Create a Meshes object for the teapot. Here we have only one mesh in the batch.
        smplx_mesh = Meshes(
            verts=[verts.to(device)],   
            faces=[faces.to(device)], 
            textures=textures
        )

    elif model_type == 'pkl':
        scan_mesh_fn = '/mnt/raid5/yixu/Dataset/4D-Dress/00122/Inner/Take8/Meshes_pkl/mesh-f00011.pkl'
        scan_mesh = load_mesh_from_pkl(scan_mesh_fn)
        verts = torch.from_numpy(scan_mesh['vertices']).float()
        faces = torch.from_numpy(scan_mesh['faces'])
        verts_rgb = torch.from_numpy(np.array(scan_mesh['colors'][:, :3]) / 255.).float()[None]
        textures = TexturesVertex(verts_features=verts_rgb.to(device))
        mesh = Meshes(
            verts=[verts.to(device)],   
            faces=[faces.to(device)], 
            textures=textures
        )
    elif model_type == 'extract_ply':
        garment_ply_file = args.mesh_path
        garment_mesh = trimesh.load_mesh(garment_ply_file)
    
    width = 1024
    height = 1024
    raster_settings = RasterizationSettings(
        image_size=(height, width), 
        blur_radius=0.0, 
        faces_per_pixel=1,
    )

    img_idx = 0

    # train = True    # train or test
    # train = False    # train or test

    if args.rotation:
        cam_range = {
            'h_angle_range': [175, 190, 20],
            'v_angle_range': [180, 360 + 184, 4],
        }
        img_folder = os.path.join(mesh_dir, 'torch3d_imgs_rot_gt')
        mask_folder = os.path.join(mesh_dir, 'torch3d_masks_rot_gt')
        distance = 1.95
        at = ((0, 0.1, 0),)
    elif not args.test:
        # 72 views
        cam_range = {
            'h_angle_range': [150, 211, 20],
            'v_angle_range': [0, 360, 20],
        }

        # # 360 views
        # cam_range = {
        #     'h_angle_range': [150, 211, 10],
        #     'v_angle_range': [0, 360, 10],
        # }
        img_folder = os.path.join(mesh_dir, 'torch3d_imgs')
        mask_folder = os.path.join(mesh_dir, 'torch3d_masks')
        distance = 2.0
        at = ((0, 0, 0),)
    else:
        cam_range = {
            'h_angle_range': [145, 206, 20],
            'v_angle_range': [-10, 350, 20]
            # 'h_angle_range': [0, 10, 10],
            # 'v_angle_range': [0, 10, 10]
        }
        img_folder = os.path.join(mesh_dir, 'torch3d_imgs_test_gt')
        mask_folder = os.path.join(mesh_dir, 'torch3d_masks_test')
        distance = 2.0
        at = ((0, 0, 0),)

    if os.path.exists(img_folder):
        shutil.rmtree(img_folder)
    os.makedirs(img_folder, exist_ok=True)
    if os.path.exists(mask_folder):
        shutil.rmtree(mask_folder)
    os.makedirs(mask_folder, exist_ok=True)
    print(f'Save the image in {img_folder}')

    """
    The camera coordinate sysmte in COLMAP is right-down-forward
    Pytorch3D is left-up-forward
    """
    # convert torch3d-format camera data into colmap-format camera data
    if args.rotation:
        folder_path = os.path.join(mesh_dir, 'converted_cams_rot')
    elif not args.test:
        folder_path = os.path.join(mesh_dir, 'converted_cams')
    else:
        folder_path = os.path.join(mesh_dir, 'converted_cams_test')
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)

    focal_length = 1 / math.tan(np.deg2rad(60) * 0.5)

    torch3d_cam_list = {}

    for h_angle in range(cam_range['h_angle_range'][0], cam_range['h_angle_range'][1], cam_range['h_angle_range'][2]):
        for v_angle in tqdm(range(cam_range['v_angle_range'][0], cam_range['v_angle_range'][1], cam_range['v_angle_range'][2])):

    # for h_angle in [0]:
    #     for v_angle in [90]:

            render_depth = False

            image_name = f'{img_idx:08d}.png'

            R, T = look_at_view_transform(distance, h_angle, v_angle, at=at)

            # save the camera info in colmap format
            colmap_utils.save_torch3d_to_colmap_cam(folder=folder_path, H=height, W=width, 
                                                    focal_length_y=focal_length, focal_length_x=focal_length, 
                                                    R=R[0], T=T[0], img_idx=img_idx + 1, image_name=image_name)
            
            pytorch3d_camera(cam_list=torch3d_cam_list, H=height, W=width, 
                                  focal_length_y=focal_length, focal_length_x=focal_length,
                                  R=R[0], T=T[0], img_idx=img_idx, image_name=image_name)

            sfm_camera = PerspectiveCameras(device=device, R=R, T=T, focal_length=focal_length)
            rasterizer = MeshRasterizer(
                cameras=sfm_camera, 
                raster_settings=raster_settings
            )
            background_color = args.bg_color
            blend_params = BlendParams(background_color=background_color)
            shader = SimpleShader(blend_params=blend_params)
            if render_depth:
                sfm_renderer = MeshRendererWithDepth(
                    rasterizer=rasterizer,
                    shader=shader,
                )
            else:
                sfm_renderer = MeshRenderer(
                    rasterizer=rasterizer,
                    shader=shader,
                )

            cam_center = sfm_camera.get_camera_center()

            if render_depth:
                images, depth = sfm_renderer(mesh)
            else:
                images = sfm_renderer(mesh)
            masks = images[..., 3]
            np_img = images[0].detach().cpu().numpy()[:,:,:3][:,:,::-1] * 255
            mask_label = np.where(masks[0].detach().cpu().numpy() == 0)
            # np_img[mask_label] = 0
            cv2.imwrite(os.path.join(img_folder, image_name), np_img)
            np_mask = masks[0].detach().cpu().numpy() * 255
            cv2.imwrite(os.path.join(mask_folder, image_name), np_mask)
            img_idx += 1
    
    # Save pytorch3d cameras into the pickle file
    if args.rotation:
        torch3d_cam_file = f'{mesh_dir}/torch3d_cam_list_rot.pkl'
    elif not args.test:
        torch3d_cam_file = f'{mesh_dir}/torch3d_cam_list.pkl'
    else:
        torch3d_cam_file = f'{mesh_dir}/torch3d_cam_list_test.pkl'
    with open(torch3d_cam_file, 'wb') as file:
        pickle.dump(torch3d_cam_list, file)
    print(f'Save the pytorch3d cameras in {torch3d_cam_file}')


def render_from_nvdiffrast():
    import nvdiffrast.torch as dr
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--mesh_dir", default="data/rp/rp_christopher_posed_008", help="folder directory where mesh obj file exists")
    parser.add_argument("-d", "--device", default="cuda:0")
    parser.add_argument("--use_opengl", action="store_true")
    args = parser.parse_args()

    data_dir = args.mesh_dir
    out_dir = data_dir + '/render'
    device = args.device
    use_opengl = args.use_opengl

    res = 1024

    mesh_names = sorted(os.listdir(data_dir))
    mesh_names = [m for m in mesh_names if "_norm.obj" in m]

    print(mesh_names)
    for mesh_name in tqdm(mesh_names):

        os.makedirs(f'{out_dir}/images_{res}', exist_ok=True)
        os.makedirs(f'{out_dir}/masks_{res}', exist_ok=True)

        verts, faces, aux = load_obj(f'{data_dir}/{mesh_name}', device=device)
        uv = aux.verts_uvs
        uv[:, 1] = 1 - uv[:, 1]  # flip v coordinate to match opengl convention
        uv_idx = faces.textures_idx.to(torch.int32)
        tex = list(aux.texture_images.values())[0].to(device)  # dealing with arbitrary key name for texture

        # pix_to_faces = {}
        # ipdb.set_trace()

        glctx = dr.RasterizeGLContext() if use_opengl else dr.RasterizeCudaContext()
        # body
        # for x in range(150, 211, 30):
        #     for y in tqdm(range(0, 360, 30)):
        img_idx = 0
        for x in range(150, 211, 20):
            for y in tqdm(range(0, 360, 20)):
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

                mask = torch.ones_like(color) * torch.clamp(rast[..., -1:], 0, 1)  # Mask out background

                # transforms.ToPILImage()(color[0].permute(2, 0, 1)).save(f'{out_dir}/images_{res}/{x:03d}_{y:03d}.png')
                # transforms.ToPILImage()(mask[0].permute(2, 0, 1)).save(f'{out_dir}/masks_{res}/{x:03d}_{y:03d}.png')
                transforms.ToPILImage()(color[0].permute(2, 0, 1)).save(f'{out_dir}/images_{res}/{img_idx:08d}.png')
                transforms.ToPILImage()(mask[0].permute(2, 0, 1)).save(f'{out_dir}/masks_{res}/{img_idx:08d}.png')
                img_idx += 1



if __name__ == '__main__':
    # render_from_nvdiffrast()
    render_from_pytorch3d()