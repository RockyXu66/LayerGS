import pickle
import os
import cv2
import trimesh
import numpy as np
import argparse
import nvdiffrast.torch as dr
import torch
from scipy.spatial.transform import Rotation as R
from pytorch3d.io import load_obj
from tqdm import tqdm
from pathlib import Path

try:
    import open3d as o3d
    _HAS_O3D = True
except Exception:
    o3d = None
    _HAS_O3D = False
from typing import List, Tuple, Optional, Dict

# ====================== Tiny-triangle removal / remesh ======================

def _tri_areas(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Compute triangle areas given vertices (N,3) and faces (M,3) int."""
    v0 = V[F[:, 0]]
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)


def _equilateral_edge_from_area(A: float) -> float:
    """For equilateral tri: A = (sqrt(3)/4) * s^2  =>  s = sqrt(4A/sqrt(3))."""
    return float(np.sqrt(max(1e-18, 4.0 * A / np.sqrt(3.0))))


def _trimesh_to_o3d(mesh: trimesh.Trimesh):
    if not _HAS_O3D:
        raise ImportError("open3d is required for remeshing. Please `pip install open3d`. ")
    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(mesh.vertices.astype(np.float64))
    m.triangles = o3d.utility.Vector3iVector(mesh.faces.astype(np.int32))
    if mesh.vertex_normals is not None and len(mesh.vertex_normals) == len(mesh.vertices):
        try:
            m.vertex_normals = o3d.utility.Vector3dVector(mesh.vertex_normals.astype(np.float64))
        except Exception:
            pass
    return m


def _o3d_to_trimesh(m: 'o3d.geometry.TriangleMesh') -> trimesh.Trimesh:
    V = np.asarray(m.vertices)
    F = np.asarray(m.triangles)
    return trimesh.Trimesh(vertices=V, faces=F, process=False)


def remesh_avoid_tiny_triangles(mesh: trimesh.Trimesh,
                                min_area: float = 1e-6,
                                smooth_iters: int = 5,
                                taubin_mu: float = 0.53,
                                taubin_lambda: float = -0.53) -> Tuple[trimesh.Trimesh, Dict]:
    """
    Regenerate topology to avoid very small triangles:
      1) Convert to Open3D
      2) Cleanup degenerate/duplicated elements
      3) Vertex clustering with voxel ~= edge length implied by min_area
      4) Taubin smoothing (few iters) to regularize
      5) Final cleanup, convert back to trimesh
    Returns (new_mesh, info_dict)
    """
    if not _HAS_O3D:
        raise ImportError("Open3D not available; cannot remesh. Install open3d>=0.16.")

    info: Dict = {}
    V0 = mesh.vertices
    F0 = mesh.faces
    areas0 = _tri_areas(V0, F0)
    info["pre_faces"] = int(len(F0))
    info["pre_min_area"] = float(areas0.min()) if len(areas0) else 0.0
    info["pre_med_area"] = float(np.median(areas0)) if len(areas0) else 0.0

    m = _trimesh_to_o3d(mesh)

    # Basic cleanup
    try:
        m.remove_degenerate_triangles()
        m.remove_duplicated_triangles()
        m.remove_duplicated_vertices()
        m.remove_non_manifold_edges()
        m.remove_unreferenced_vertices()
    except Exception:
        pass

    # Choose clustering voxel size from desired minimum area
    s_min = _equilateral_edge_from_area(min_area)
    voxel = max(1e-9, s_min)  # meters

    try:
        m = m.simplify_vertex_clustering(
            voxel_size=voxel,
            contraction=o3d.geometry.SimplificationContraction.Average,
        )
    except Exception:
        # Fallback: quadric decimation to a rough target count
        curF = np.asarray(m.triangles).shape[0]
        target = max(100, int(0.75 * curF))
        m = m.simplify_quadric_decimation(target)

    # Optional gentle smoothing to reduce slivers further
    try:
        if smooth_iters > 0:
            m = m.filter_smooth_taubin(number_of_iterations=int(smooth_iters),
                                       lambda_filter=taubin_lambda, mu=taubin_mu)
    except Exception:
        pass

    # Final cleanup
    try:
        m.remove_degenerate_triangles()
        m.remove_duplicated_triangles()
        m.remove_unreferenced_vertices()
        m.remove_non_manifold_edges()
    except Exception:
        pass

    m.compute_vertex_normals()

    mesh_new = _o3d_to_trimesh(m)

    # Verify small-triangle condition; report if any remain
    areas1 = _tri_areas(mesh_new.vertices, mesh_new.faces) if len(mesh_new.faces) else np.array([])
    info["post_faces"] = int(len(mesh_new.faces))
    info["post_min_area"] = float(areas1.min()) if len(areas1) else 0.0
    info["post_med_area"] = float(np.median(areas1)) if len(areas1) else 0.0
    info["min_area_threshold"] = float(min_area)
    info["voxel_size"] = float(voxel)
    info["tiny_faces_remaining"] = int(np.sum(areas1 < min_area)) if len(areas1) else 0

    # If a tiny number of faces remain under threshold, try a light decimation pass
    if info["tiny_faces_remaining"] > 0:
        try:
            m2 = _trimesh_to_o3d(mesh_new)
            curF = np.asarray(m2.triangles).shape[0]
            target = max(100, int(0.95 * curF))
            m2 = m2.simplify_quadric_decimation(target)
            m2.remove_degenerate_triangles(); m2.remove_unreferenced_vertices()
            mesh_new = _o3d_to_trimesh(m2)
            areas2 = _tri_areas(mesh_new.vertices, mesh_new.faces) if len(mesh_new.faces) else np.array([])
            info["post2_faces"] = int(len(mesh_new.faces))
            info["post2_min_area"] = float(areas2.min()) if len(areas2) else 0.0
            info["post2_med_area"] = float(np.median(areas2)) if len(areas2) else 0.0
            info["tiny_faces_remaining2"] = int(np.sum(areas2 < min_area)) if len(areas2) else 0
        except Exception:
            pass

    return mesh_new, info


parser = argparse.ArgumentParser()
parser.add_argument("-f", "--mesh_dir", type=Path, default="data/rp/rp_george_posed_002", help="directory to the mesh")
parser.add_argument("--dataset_type", default='rp')
parser.add_argument("--use_opengl", action="store_true")
parser.add_argument("-d", "--device", default="cuda:0")
parser.add_argument("--mesh_name", type=str, default="fuse_post_comp.ply", help="name of the mesh file to render")
args = parser.parse_args()

data_dir = args.mesh_dir
dataset_type = args.dataset_type
device = args.device
use_opengl = args.use_opengl

res_upsample = 2048  # high resolution rendering for finding pixel-face correspondences
num_views = 72


ROOT_FOLDER = args.mesh_dir
mesh_name = args.mesh_name

ply_filename = f'{ROOT_FOLDER}/{mesh_name}'
# ply_filename = f'{ROOT_FOLDER}/fuse_post_comp_wshape.ply'
# ply_filename = f'{ROOT_FOLDER}/fuse_post_comp_posed.ply'
print(ply_filename)
mesh = trimesh.load(ply_filename)

# apply remesh to remove very very small triangles
MIN_TRI_AREA = 5e-5  # adjust if needed (area units follow your mesh units^2)
mesh, rinfo = remesh_avoid_tiny_triangles(mesh, min_area=MIN_TRI_AREA, smooth_iters=5)

verts = torch.from_numpy(mesh.vertices).float().cuda()
indices = torch.from_numpy(mesh.faces).int().cuda()

face_mask_visible_stack = np.zeros((len(indices), num_views))
face_mask_segm_stack = np.zeros((len(indices), num_views))

glctx = dr.RasterizeGLContext() if use_opengl else dr.RasterizeCudaContext()

########## Use sapiens' result ##########
use_sapiens = True
# use_sapiens = False
sapiens_seg_labels = {
    'background': 0,
    'lower_clothing': 12,
    'upper_clothing': 22,
}

# seg_garment_name = 'lower_clothing'
seg_garment_name = 'upper_clothing'
# seg_garment_name = ''

view_idx = 0
# for x in range(150, 211, 30):
#     for y in tqdm(range(0, 360, 30)):
for x in range(150, 211, 20):
    for y in tqdm(range(0, 360, 20)):
        if use_sapiens:
            segm_path = f'{ROOT_FOLDER}/sapiens_1b/{x:03d}_{y:03d}_seg.npy'
            segm_mask = np.load(segm_path)
            segm_mask = torch.from_numpy(segm_mask)
            mask_body = torch.zeros((segm_mask.shape[0], segm_mask.shape[1], 1), dtype=torch.float32, device=device)
            mask_body[torch.where(segm_mask==sapiens_seg_labels[seg_garment_name])] = 1.0
            segm_mask = mask_body
        else:
            segm_path = f'{ROOT_FOLDER}/segm_masks/{x:03d}_{y:03d}.png'
            if not os.path.exists(segm_path):  # SAM failed
                continue
            segm_mask = cv2.imread(segm_path) // 255
            segm_mask = torch.tensor(segm_mask[..., 0:1], dtype=torch.float32, device=device)

        r_xpi = R.from_euler('x', x, degrees=True).as_matrix()
        r_ypi = R.from_euler('y', y, degrees=True).as_matrix()
        r = r_xpi @ r_ypi

        transformed_vertices = (torch.tensor(r, device=device).float() @ (verts.T)).T
        transformed_vertices_h = torch.cat([transformed_vertices, torch.ones_like(transformed_vertices[:, 0:1])], axis=1)
        rast, rast_out_db = dr.rasterize(glctx, transformed_vertices_h[None], indices, 
                                            resolution=np.array([res_upsample, res_upsample]))

        uv = 0.5 * (transformed_vertices[:, :2].contiguous() + 1) 
        texc, _ = dr.interpolate(uv[None, ...], rast, indices)
        segm_mask_upsample = (dr.texture(segm_mask[None, ...], texc, filter_mode='linear') >= 0.5)[0, ..., 0].cpu().numpy()

        face_ids = rast[0, ..., -1].cpu().numpy().astype(np.int32)
        face_ids_visible = face_ids[face_ids != 0] - 1
        face_mask_visible_stack[face_ids_visible, view_idx] = 1

        face_ids_segm = face_ids[segm_mask_upsample] - 1  # face ids segmented as target
        face_mask_segm_stack[face_ids_segm, view_idx] = 1

        view_idx += 1
        
face_mask_vote = ((face_mask_segm_stack.sum(axis=-1) / (face_mask_visible_stack.sum(axis=-1) + 1e-8))) >= 0.5
np.save(f'{ROOT_FOLDER}/faces_segms.npy', face_mask_vote)

mesh_segms_obj = trimesh.Trimesh(vertices=verts.cpu().numpy(), faces=indices.cpu().numpy())
# leave only obj mesh
mesh_segms_obj.update_faces(face_mask_vote)

# clean up isolated components
# reference: https://github.com/mikedh/trimesh/issues/895
cc = trimesh.graph.connected_components(mesh_segms_obj.face_adjacency, min_len=25)
mask_cc = np.zeros(len(mesh_segms_obj.faces), dtype=bool)
mask_cc[np.concatenate(cc)] = True
mesh_segms_obj.update_faces(mask_cc)

if seg_garment_name == 'lower_clothing':
    mesh_segms_obj.export(f'{ROOT_FOLDER}/segms_lower.obj')
    print(f"Save the extracted obj in {ROOT_FOLDER}/segms_lower.obj")
else:
    mesh_segms_obj.export(f'{ROOT_FOLDER}/segms_obj.obj')
    print(f"Save the extracted obj in {ROOT_FOLDER}/segms_obj.obj")
