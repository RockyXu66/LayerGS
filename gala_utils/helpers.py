# code from fast snarf

import torch
import math
import cv2
import numpy as np
import torch.nn.functional as F
from torch import einsum
import pytorch3d.ops as ops

# transforms verts in canonical space to posed space using NN smplx vertex (incorporates shape param offsets)
def cano_to_posed(verts, tfs, shape_pose_offsets_grid, lbs_weights_grid):
    offsets = torch.nn.functional.grid_sample(shape_pose_offsets_grid , verts[None, None, None, :, :]).reshape(1, 3, -1).permute(0, 2, 1)
    lbs_weights = torch.nn.functional.grid_sample(lbs_weights_grid, verts[None, None, None, :, :]).reshape(1, 55, -1).permute(0, 2, 1)
    verts = verts + offsets[0]  # shape pose blend shapes
    verts = skinning(verts, lbs_weights, tfs)[0]  # lbs
    return verts


def query_weights_smpl(x, smpl_verts, smpl_weights):
    
    distance_batch, index_batch, neighbor_points  = ops.knn_points(x,smpl_verts[None],K=1,return_nn=True)

    index_batch = index_batch[0]

    skinning_weights = smpl_weights[None][:,index_batch][:,:,0,:]

    return skinning_weights

def create_voxel_grid(d, h, w, device='cpu'):
    x_range = (torch.linspace(-1,1,steps=w,device=device)).view(1, 1, 1, w).expand(1, d, h, w)  # [1, H, W, D]
    y_range = (torch.linspace(-1,1,steps=h,device=device)).view(1, 1, h, 1).expand(1, d, h, w)  # [1, H, W, D]
    z_range = (torch.linspace(-1,1,steps=d,device=device)).view(1, d, 1, 1).expand(1, d, h, w)  # [1, H, W, D]
    grid = torch.cat((x_range, y_range, z_range), dim=0).reshape(1, 3,-1).permute(0,2,1)

    return grid

def skinning(x, w, tfs, inverse=False):
    """Linear blend skinning

    Args:
        x (tensor): canonical points. shape: [B, N, D]
        w (tensor): conditional input. [B, N, J]
        tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]
    Returns:
        x (tensor): skinned points. shape: [B, N, D]
    """
    x_h = F.pad(x, (0, 1), value=1.0)
    b,p,n = w.shape

    if inverse:
        # p:n_point, n:n_bone, i,k: n_dim+1
        w_tf = einsum("bpn,bnij->bpij", w, tfs)

        x_h = x_h.view(b,p,1,4).expand(b,p,4,4)
        x_h = ((w_tf.inverse())*x_h).sum(-1)

    else:
        w_tf = einsum("bpn,bnij->bpij", w, tfs)

        x_h = x_h.view(b,p,1,4).expand(b,p,4,4)
        x_h = (w_tf*x_h).sum(-1)

    return x_h[:, :, :3]
