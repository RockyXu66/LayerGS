import os
import math
from datetime import datetime
import random
# Config type
from omegaconf import DictConfig
import bisect

from dataclasses import dataclass, field
from omegaconf import OmegaConf
# ============ Register OmegaConf Recolvers ============= #
OmegaConf.register_new_resolver(
    "calc_exp_lr_decay_rate", lambda factor, n: factor ** (1.0 / n)
)
OmegaConf.register_new_resolver("add", lambda a, b: a + b)
OmegaConf.register_new_resolver("sub", lambda a, b: a - b)
OmegaConf.register_new_resolver("mul", lambda a, b: a * b)
OmegaConf.register_new_resolver("div", lambda a, b: a / b)
OmegaConf.register_new_resolver("idiv", lambda a, b: a // b)
OmegaConf.register_new_resolver("basename", lambda p: os.path.basename(p))
OmegaConf.register_new_resolver("rmspace", lambda s, sub: s.replace(" ", sub))
OmegaConf.register_new_resolver("tuple2", lambda s: [float(s), float(s)])
OmegaConf.register_new_resolver("gt0", lambda s: s > 0)
OmegaConf.register_new_resolver("cmaxgt0", lambda s: C_max(s) > 0)
OmegaConf.register_new_resolver("not", lambda s: not s)
OmegaConf.register_new_resolver(
    "cmaxgt0orcmaxgt0", lambda a, b: C_max(a) > 0 or C_max(b) > 0
)
# ======================================================= #


# Basic types
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    NamedTuple,
    NewType,
    Optional,
    Sized,
    Tuple,
    Type,
    TypeVar,
    Union,
)
# Tensor dtype
# for jaxtyping usage, see https://github.com/google/jaxtyping/blob/main/API.md
from jaxtyping import Bool, Complex, Float, Inexact, Int, Integer, Num, Shaped, UInt
# PyTorch Tensor type
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset

import torch
cam_random = random.Random(99)
torch_new_generator = torch.Generator()
torch_new_generator.manual_seed(99)

def reset_random_generators(seed: int = 0):
    """Reset module-level random generators for reproducibility.
    
    This function should be called before training starts to ensure
    reproducible random number generation across training runs.
    """
    global cam_random, torch_new_generator
    cam_random = random.Random(seed)
    torch_new_generator = torch.Generator()
    torch_new_generator.manual_seed(seed)


@dataclass
class RandomCameraDataModuleConfig:
    # height, width, and batch_size should be Union[int, List[int]]
    # but OmegaConf does not support Union of containers
    height: Any = 512
    width: Any = 512
    batch_size: Any = 1
    resolution_milestones: List[int] = field(default_factory=lambda: [])
    eval_height: int = 512
    eval_width: int = 512
    eval_batch_size: int = 1
    n_val_views: int = 4
    n_test_views: int = 120
    elevation_range: Tuple[float, float] = (-30, 60)
    azimuth_range: Tuple[float, float] = (-180, 180)
    camera_distance_range: Tuple[float, float] = (4.,6.)
    fovy_range: Tuple[float, float] = (
        40,
        70,
    )  # in degrees, in vertical direction (along height)
    camera_perturb: float = 0.
    center_perturb: float = 0.
    up_perturb: float = 0.0
    light_position_perturb: float = 1.0
    light_distance_range: Tuple[float, float] = (0.8, 1.5)
    eval_elevation_deg: float = 15.0
    eval_camera_distance: float = 6.
    eval_fovy_deg: float = 70.0
    light_sample_strategy: str = "dreamfusion"
    batch_uniform_azimuth: bool = True
    progressive_until: int = 0  # progressive ranges for elevation, azimuth, r, fovy

    # near head pose
    enable_near_head_poses: bool = False
    enable_near_back_poses: bool = False
    enable_near_pant_poses: bool = False
    head_offset: float = 0.65
    back_offset: float = 0.65
    pant_offset: float = -0.65
    head_camera_distance_range: Tuple[float, float] = (0.4, 0.6)
    back_camera_distance_range: Tuple[float, float] = (0.6, 0.8)
    pant_camera_distance_range: Tuple[float, float] = (0.6, 0.8)
    head_prob: float = 0.25
    head_start_step: int = 1200
    head_end_step: int = 3600
    head_azimuth_range: Tuple[float, float] = (0, 180)
    back_prob: float = 0.20
    back_start_step: int = 1200
    back_end_step: int = 3600
    back_azimuth_range: Tuple[float, float] = (-180, 0)
    pant_prob: float = 0.20
    pant_start_step: int = 1200
    pant_end_step: int = 3600
    pant_azimuth_range: Tuple[float, float] = (-180, 0)
    frontal_prob: float = 0.0
    frontal_azimuth_range: Tuple[float, float] = (45, 135)

    from_ply: int = 1
    ply_name: str = "point_cloud_comp_cano"
    opacity_reset_list: list[int] = field(default_factory=lambda: [])
    opacity_reset_val: float = 0.02
    prune_min_opacity: float = 0.005
    inner_init_sample_num: int = 50000
    inner_dum_sample_num: int = 100000
    seg_dilation_size: int = 13
    sds_cfg_list: list[float] = field(default_factory=lambda: [])
    apose_prob: float = 0.5
    end_densify_iter: int = 12000
    apply_op_control_start_iteration: int = 0
    apply_op_control_end_iteration: int = 30000

    apply_body_shrink: bool = True
    lambda_normal_iter: int = 6000
    lambda_normal: float = 0.05
    lambda_dist_iter: int = 2000
    lambda_dist: float = 1000.0
    apply_dum_dist: bool = True
    lambda_dist_body_dum: int = 20000
    apply_dum_dist_start_iter: int = 0
    apply_dum_dist_end_iter: int = 4800
    apply_sam_seg: bool = False
    apply_evaluate: bool = False
    apply_foreground_mask: bool = False
    apply_4dDress_seg: bool = False
    seg_4dDress_outer_label: str = 'outer'
    apply_sapiens_bg: bool = False
    apply_distance_thresh: bool = False
    apply_distance_thresh_start_iter: int = 0
    apply_distance_thresh_end_iter: int = 0
    distance_thresh: float = 0.02
    apply_scale_clipping: bool = False
    max_scale: float = 0.01

    gender: str = "male"
    text: str = "a photo of a man wearing a white tank top and grey pants"
    negative_text: str = "muscular, close up, cropped, worst quality, low quality, jpeg artifacts, duplicate, morbid, mutilated, mutation, deformed, blurry, bad anatomy, bad proportions, long neck"
    text_obj: str = "a photo of a man wearing a jacket"
    negative_text_obj: str = ""
    add_directional_text: bool = True
    apply_ori_img: bool = False
    inpaint_garment_type: str = "upper"

    w_seg_loss_body: float = 3.0
    w_seg_loss_garment: float = 3.0
    w_seg_loss_bg: float = 3.0
    w_img_loss: float = 3.0
    w_dist_loss: float = 3.0
    w_normal_loss: float = 3.0

class RandomCameraIterableDataset(IterableDataset):
    def __init__(self, cfg: Any, use_opengl_camera = True) -> None:
        super().__init__()

        self.use_opengl_camera = use_opengl_camera   # whether to use opengl camera or blender camera

        self.cfg: RandomCameraDataModuleConfig = cfg
        self.heights: List[int] = (
            [self.cfg.height] if isinstance(self.cfg.height, int) else self.cfg.height
        )
        self.widths: List[int] = (
            [self.cfg.width] if isinstance(self.cfg.width, int) else self.cfg.width
        )
        self.batch_sizes: List[int] = (
            [self.cfg.batch_size]
            if isinstance(self.cfg.batch_size, int)
            else self.cfg.batch_size
        )
        assert len(self.heights) == len(self.widths) == len(self.batch_sizes)
        self.resolution_milestones: List[int]
        if (
            len(self.heights) == 1
            and len(self.widths) == 1
            and len(self.batch_sizes) == 1
        ):
            self.resolution_milestones = [-1]
        else:
            assert len(self.heights) == len(self.cfg.resolution_milestones) + 1
            self.resolution_milestones = [-1] + self.cfg.resolution_milestones

        self.directions_unit_focals = [
            get_ray_directions(H=height, W=width, focal=1.0)
            for (height, width) in zip(self.heights, self.widths)
        ]
        self.height: int = self.heights[0]
        self.width: int = self.widths[0]
        self.batch_size: int = self.batch_sizes[0]
        self.directions_unit_focal = self.directions_unit_focals[0]
        self.elevation_range = self.cfg.elevation_range
        self.azimuth_range = self.cfg.azimuth_range
        self.camera_distance_range = self.cfg.camera_distance_range
        self.head_camera_distance_range = self.cfg.head_camera_distance_range
        self.back_camera_distance_range = self.cfg.back_camera_distance_range
        self.pant_camera_distance_range = self.cfg.pant_camera_distance_range
        self.fovy_range = self.cfg.fovy_range
        self.cur_step = 0
    
    def __iter__(self):
        while True:
            yield {}

    def collate(self, batch) -> Dict[str, Any]:
        self.cur_step += 1

        # random head zoom-in
        if self.cfg.enable_near_head_poses and cam_random.random() < self.cfg.head_prob and self.cur_step >= self.cfg.head_start_step and self.cur_step <= self.cfg.head_end_step:
            zoom_in_head = True
            zoom_in_back = False
            camera_distance_range = self.head_camera_distance_range
            self.azimuth_range = self.cfg.head_azimuth_range
        elif self.cfg.enable_near_back_poses and cam_random.random() < self.cfg.back_prob and self.cur_step >= self.cfg.back_start_step and self.cur_step <= self.cfg.back_end_step:
            zoom_in_head = False
            zoom_in_back = True
            camera_distance_range = self.back_camera_distance_range
            self.azimuth_range = self.cfg.back_azimuth_range
        elif self.cfg.enable_near_pant_poses and cam_random.random() < self.cfg.pant_prob and self.cur_step >= self.cfg.pant_start_step and self.cur_step <= self.cfg.pant_end_step:
            zoom_in_head = False
            zoom_in_back = False
            zoom_in_pant = True
            camera_distance_range = self.pant_camera_distance_range
            self.azimuth_range = self.cfg.pant_azimuth_range
        else:
            zoom_in_head = False
            zoom_in_back = False
            zoom_in_pant = False
            camera_distance_range = self.camera_distance_range
            if cam_random.random() < self.cfg.frontal_prob:
                self.azimuth_range = self.cfg.frontal_azimuth_range
            else:    
                self.azimuth_range = self.cfg.azimuth_range
        
        # sample elevation angles
        elevation_deg: Float[Tensor, "B"]
        elevation: Float[Tensor, "B"]
        if cam_random.random() < 0.5:
            # sample elevation angles uniformly with a probability 0.5 (biased towards poles)
            elevation_deg = (
                torch.rand(self.batch_size, generator=torch_new_generator)
                * (self.elevation_range[1] - self.elevation_range[0])
                + self.elevation_range[0]
            )
            # elevation_deg = torch.tensor([0])
            elevation = elevation_deg * math.pi / 180
        else:
            # otherwise sample uniformly on sphere
            elevation_range_percent = [
                (self.elevation_range[0] + 90.0) / 180.0,
                (self.elevation_range[1] + 90.0) / 180.0,
            ]
            # inverse transform sampling
            elevation = torch.asin(
                2
                * (
                    torch.rand(self.batch_size, generator=torch_new_generator)
                    * (elevation_range_percent[1] - elevation_range_percent[0])
                    + elevation_range_percent[0]
                )
                - 1.0
            )
            # elevation = torch.tensor([0])
            elevation_deg = elevation / math.pi * 180.0

        # sample azimuth angles from a uniform distribution bounded by azimuth_range
        azimuth_deg: Float[Tensor, "B"]
        if self.cfg.batch_uniform_azimuth:
            # ensures sampled azimuth angles in a batch cover the whole range
            azimuth_deg = (
                torch.rand(self.batch_size, generator=torch_new_generator) + torch.arange(self.batch_size)
            ) / self.batch_size * (
                self.azimuth_range[1] - self.azimuth_range[0]
            ) + self.azimuth_range[
                0
            ]
        else:
            # simple random sampling
            azimuth_deg = (
                torch.rand(self.batch_size, generator=torch_new_generator)
                * (self.azimuth_range[1] - self.azimuth_range[0])
                + self.azimuth_range[0]
            )
        # azimuth_deg = torch.tensor([90])
        azimuth = azimuth_deg * math.pi / 180

        # sample distances from a uniform distribution bounded by distance_range
        camera_distances: Float[Tensor, "B"] = (
            torch.rand(self.batch_size, generator=torch_new_generator)
            * (camera_distance_range[1] - camera_distance_range[0])
            + camera_distance_range[0]
        )

        # convert spherical coordinates to cartesian coordinates
        # right hand coordinate system, x back, y right, z up
        # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
        if self.use_opengl_camera:  # opengl camera
            camera_positions: Float[Tensor, "B 3"] = torch.stack(
                [
                    camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                    camera_distances * torch.sin(elevation),
                    camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                ],
                dim=-1,
            )
            y_axis = 1
        else:   # blender camera
            camera_positions: Float[Tensor, "B 3"] = torch.stack(
                [
                    camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                    camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                    camera_distances * torch.sin(elevation),
                ],
                dim=-1,
            )
            y_axis = 2

        # default scene center at origin
        center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions)

        if zoom_in_head:
            # z-axis add offset to move the camera centered around head
            center[:, y_axis] += self.cfg.head_offset
            camera_positions[:, y_axis] += self.cfg.head_offset
        elif zoom_in_back:
            # z-axis add offset to move the camera centered around head
            center[:, y_axis] += self.cfg.back_offset
            camera_positions[:, y_axis] += self.cfg.back_offset
        elif zoom_in_pant:
            # z-axis add offset to move the camera centered around head
            center[:, y_axis] += self.cfg.pant_offset
            camera_positions[:, y_axis] += self.cfg.pant_offset

        # default camera up direction as +z
        if self.use_opengl_camera:
            up: Float[Tensor, "B 3"] = torch.as_tensor([0, 1, 0], dtype=torch.float32)[
                None, :
            ].repeat(self.batch_size, 1)
        else:
            up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
                None, :
            ].repeat(self.batch_size, 1)

        # sample camera perturbations from a uniform distribution [-camera_perturb, camera_perturb]
        camera_perturb: Float[Tensor, "B 3"] = (
            torch.rand(self.batch_size, 3, generator=torch_new_generator) * 2 * self.cfg.camera_perturb
            - self.cfg.camera_perturb
        )
        camera_positions = camera_positions + camera_perturb
        # sample center perturbations from a normal distribution with mean 0 and std center_perturb
        center_perturb: Float[Tensor, "B 3"] = (
            torch.randn(self.batch_size, 3, generator=torch_new_generator) * self.cfg.center_perturb
        )
        center = center + center_perturb
        # sample up perturbations from a normal distribution with mean 0 and std up_perturb
        up_perturb: Float[Tensor, "B 3"] = (
            torch.randn(self.batch_size, 3, generator=torch_new_generator) * self.cfg.up_perturb
        )
        up = up + up_perturb

        # sample fovs from a uniform distribution bounded by fov_range
        fovy_deg: Float[Tensor, "B"] = (
            torch.rand(self.batch_size, generator=torch_new_generator) * (self.fovy_range[1] - self.fovy_range[0])
            + self.fovy_range[0]
        )
        fovy = fovy_deg * math.pi / 180

        # sample light distance from a uniform distribution bounded by light_distance_range
        light_distances: Float[Tensor, "B"] = (
            torch.rand(self.batch_size, generator=torch_new_generator)
            * (self.cfg.light_distance_range[1] - self.cfg.light_distance_range[0])
            + self.cfg.light_distance_range[0]
        )

        if self.cfg.light_sample_strategy == "dreamfusion" or self.cfg.light_sample_strategy == "dreamfusion3dgs":
            # sample light direction from a normal distribution with mean camera_position and std light_position_perturb
            light_direction: Float[Tensor, "B 3"] = F.normalize(
                camera_positions
                + torch.randn(self.batch_size, 3, generator=torch_new_generator) * self.cfg.light_position_perturb,
                dim=-1,
            )
            # get light position by scaling light direction by light distance
            light_positions: Float[Tensor, "B 3"] = (
                light_direction * light_distances[:, None]
            )
        elif self.cfg.light_sample_strategy == "magic3d":
            # sample light direction within restricted angle range (pi/3)
            local_z = F.normalize(camera_positions, dim=-1)
            local_x = F.normalize(
                torch.stack(
                    [local_z[:, 1], -local_z[:, 0], torch.zeros_like(local_z[:, 0])],
                    dim=-1,
                ),
                dim=-1,
            )
            local_y = F.normalize(torch.cross(local_z, local_x, dim=-1), dim=-1)
            rot = torch.stack([local_x, local_y, local_z], dim=-1)
            light_azimuth = (
                torch.rand(self.batch_size, generator=torch_new_generator) * math.pi * 2 - math.pi
            )  # [-pi, pi]
            light_elevation = (
                torch.rand(self.batch_size, generator=torch_new_generator) * math.pi / 3 + math.pi / 6
            )  # [pi/6, pi/2]
            light_positions_local = torch.stack(
                [
                    light_distances
                    * torch.cos(light_elevation)
                    * torch.cos(light_azimuth),
                    light_distances
                    * torch.cos(light_elevation)
                    * torch.sin(light_azimuth),
                    light_distances * torch.sin(light_elevation),
                ],
                dim=-1,
            )
            light_positions = (rot @ light_positions_local[:, :, None])[:, :, 0]
        else:
            raise ValueError(
                f"Unknown light sample strategy: {self.cfg.light_sample_strategy}"
            )

        lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1)
        if self.use_opengl_camera:
            right: Float[Tensor, "B 3"] = F.normalize(torch.cross(up, lookat), dim=-1)
            up = F.normalize(torch.cross(lookat, right), dim=-1)
        else:   # blender camera
            right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
            up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w: Float[Tensor, "B 4 4"] = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
        )
        c2w[:, 3, 3] = 1.0

        # get directions by dividing directions_unit_focal by focal length
        focal_length: Float[Tensor, "B"] = 0.5 * self.height / torch.tan(0.5 * fovy)
        directions: Float[Tensor, "B H W 3"] = self.directions_unit_focal[
            None, :, :, :
        ].repeat(self.batch_size, 1, 1, 1)
        directions[:, :, :, :2] = (
            directions[:, :, :, :2] / focal_length[:, None, None, None]
        )

        proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(
            fovy, self.width / self.height, 0.1, 1000.0
        )  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx)

        return {
            "mvp_mtx": mvp_mtx,
            "camera_positions": camera_positions,
            "c2w": c2w,
            "light_positions": light_positions,
            "elevation": elevation_deg,
            "azimuth": azimuth_deg,
            "camera_distances": camera_distances,
            "height": self.height,
            "width": self.width,
            "fovy":fovy,

        }

class RandomCameraDataset(Dataset):
    def __init__(self, cfg: Any, split: str, use_opengl_camera = True) -> None:
        super().__init__()

        self.use_opengl_camera = use_opengl_camera      # whether to use opengl camera or blender camera

        self.cfg: RandomCameraDataModuleConfig = cfg
        self.split = split

        if split == "val":
            self.n_views = self.cfg.n_val_views
        else:
            self.n_views = self.cfg.n_test_views

        azimuth_deg: Float[Tensor, "B"]
        if self.split == "val":
            # make sure the first and last view are not the same
            azimuth_deg = torch.linspace(-180., 180.0, self.n_views + 1)[: self.n_views]
        else:
            azimuth_deg = torch.linspace(-180., 180.0, self.n_views)
        elevation_deg: Float[Tensor, "B"] = torch.full_like(
            azimuth_deg, self.cfg.eval_elevation_deg
        )
        camera_distances: Float[Tensor, "B"] = torch.full_like(
            elevation_deg, self.cfg.eval_camera_distance
        )

        elevation = elevation_deg * math.pi / 180
        azimuth = azimuth_deg * math.pi / 180

        # convert spherical coordinates to cartesian coordinates
        # right hand coordinate system, x back, y right, z up
        # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
        if self.use_opengl_camera:  # opengl camera
            camera_positions: Float[Tensor, "B 3"] = torch.stack(
                [
                    camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                    camera_distances * torch.sin(elevation),
                    camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                ],
                dim=-1,
            )
        else:   # blender camera
            camera_positions: Float[Tensor, "B 3"] = torch.stack(
                [
                    camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                    camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                    camera_distances * torch.sin(elevation),
                ],
                dim=-1,
            )

        # default scene center at origin
        center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions)
        # default camera up direction as +z
        # up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
        if self.use_opengl_camera:
            up: Float[Tensor, "B 3"] = torch.as_tensor([0, 1, 0], dtype=torch.float32)[
                None, :
            ].repeat(self.cfg.eval_batch_size, 1)
        else:
            up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
                None, :
            ].repeat(self.cfg.eval_batch_size, 1)

        fovy_deg: Float[Tensor, "B"] = torch.full_like(
            elevation_deg, self.cfg.eval_fovy_deg
        )
        fovy = fovy_deg * math.pi / 180
        light_positions: Float[Tensor, "B 3"] = camera_positions

        lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1)
        if self.use_opengl_camera:
            right: Float[Tensor, "B 3"] = F.normalize(torch.cross(up, lookat), dim=-1)
            up = F.normalize(torch.cross(lookat, right), dim=-1)
        else:   # blender camera
            right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
            up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w: Float[Tensor, "B 4 4"] = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
        )
        c2w[:, 3, 3] = 1.0

        # get directions by dividing directions_unit_focal by focal length
        focal_length: Float[Tensor, "B"] = (
            0.5 * self.cfg.eval_height / torch.tan(0.5 * fovy)
        )
        directions_unit_focal = get_ray_directions(
            H=self.cfg.eval_height, W=self.cfg.eval_width, focal=1.0
        )
        directions: Float[Tensor, "B H W 3"] = directions_unit_focal[
            None, :, :, :
        ].repeat(self.n_views, 1, 1, 1)
        directions[:, :, :, :2] = (
            directions[:, :, :, :2] / focal_length[:, None, None, None]
        )

        proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(
            fovy, self.cfg.eval_width / self.cfg.eval_height, 0.1, 1000.0
        )  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx)

        self.mvp_mtx = mvp_mtx
        self.c2w = c2w

        self.camera_positions = camera_positions
        self.light_positions = light_positions
        self.elevation, self.azimuth = elevation, azimuth
        self.elevation_deg, self.azimuth_deg = elevation_deg, azimuth_deg
        self.camera_distances = camera_distances
        self.fovy = fovy

    def __len__(self):
        return self.n_views

    def __getitem__(self, index):
        index = index % self.__len__()
        return {
            "index": index,
            "mvp_mtx": self.mvp_mtx[index],
            "c2w": self.c2w[index],
            "camera_positions": self.camera_positions[index],
            "light_positions": self.light_positions[index],
            "elevation": self.elevation_deg[index],
            "azimuth": self.azimuth_deg[index],
            "camera_distances": self.camera_distances[index],
            "height": self.cfg.eval_height,
            "width": self.cfg.eval_width,
            "fovy":self.fovy[index],
        }

    def collate(self, batch):
        batch = torch.utils.data.default_collate(batch)
        batch.update({"height": self.cfg.eval_height, "width": self.cfg.eval_width})
        return batch

def get_ray_directions(
    H: int,
    W: int,
    focal: Union[float, Tuple[float, float]],
    principal: Optional[Tuple[float, float]] = None,
    use_pixel_centers: bool = True,
) -> Float[Tensor, "H W 3"]:
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W, focal, principal, use_pixel_centers: image height, width, focal length, principal point and whether use pixel centers
    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    pixel_center = 0.5 if use_pixel_centers else 0

    if isinstance(focal, float):
        fx, fy = focal, focal
        cx, cy = W / 2, H / 2
    else:
        fx, fy = focal
        assert principal is not None
        cx, cy = principal

    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32) + pixel_center,
        torch.arange(H, dtype=torch.float32) + pixel_center,
        indexing="xy",
    )

    directions: Float[Tensor, "H W 3"] = torch.stack(
        [(i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i)], -1
    )

    return directions

def get_mvp_matrix(
    c2w: Float[Tensor, "B 4 4"], proj_mtx: Float[Tensor, "B 4 4"]
) -> Float[Tensor, "B 4 4"]:
    # calculate w2c from c2w: R' = Rt, t' = -Rt * t
    # mathematically equivalent to (c2w)^-1
    w2c: Float[Tensor, "B 4 4"] = torch.zeros(c2w.shape[0], 4, 4).to(c2w)
    w2c[:, :3, :3] = c2w[:, :3, :3].permute(0, 2, 1)
    w2c[:, :3, 3:] = -c2w[:, :3, :3].permute(0, 2, 1) @ c2w[:, :3, 3:]
    w2c[:, 3, 3] = 1.0
    # calculate mvp matrix by proj_mtx @ w2c (mv_mtx)
    mvp_mtx = proj_mtx @ w2c
    return mvp_mtx

def get_projection_matrix(
    fovy: Float[Tensor, "B"], aspect_wh: float, near: float, far: float
) -> Float[Tensor, "B 4 4"]:
    batch_size = fovy.shape[0]
    proj_mtx = torch.zeros(batch_size, 4, 4, dtype=torch.float32)
    proj_mtx[:, 0, 0] = 1.0 / (torch.tan(fovy / 2.0) * aspect_wh)
    proj_mtx[:, 1, 1] = -1.0 / torch.tan(
        fovy / 2.0
    )  # add a negative sign here as the y axis is flipped in nvdiffrast output
    proj_mtx[:, 2, 2] = -(far + near) / (far - near)
    proj_mtx[:, 2, 3] = -2.0 * far * near / (far - near)
    proj_mtx[:, 3, 2] = -1.0
    return proj_mtx

@dataclass
class ExperimentConfig:
    name: str = "default"
    description: str = ""
    tag: str = ""
    seed: int = 0
    use_timestamp: bool = True
    timestamp: Optional[str] = None
    exp_root_dir: str = "outputs"

    ### these shouldn't be set manually
    exp_dir: str = "outputs/default"
    trial_name: str = "exp"
    trial_dir: str = "outputs/default/exp"
    n_gpus: int = 1
    ###

    resume: Optional[str] = None

    data_type: str = ""
    data: dict = field(default_factory=dict)

    system_type: str = ""
    system: dict = field(default_factory=dict)

    # accept pytorch-lightning trainer parameters
    # see https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-class-api
    trainer: dict = field(default_factory=dict)

    # accept pytorch-lightning checkpoint callback parameters
    # see https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html#modelcheckpoint
    checkpoint: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.tag and not self.use_timestamp:
            raise ValueError("Either tag is specified or use_timestamp is True.")
        self.trial_name = self.tag
        # if resume from an existing config, self.timestamp should not be None
        if self.timestamp is None:
            self.timestamp = ""
            if self.use_timestamp:
                self.timestamp = datetime.now().strftime("@%Y%m%d-%H%M%S")
        self.trial_name += self.timestamp
        self.exp_dir = os.path.join(self.exp_root_dir, self.name)
        self.trial_dir = os.path.join(self.exp_dir, self.trial_name)
        os.makedirs(self.trial_dir, exist_ok=True)

def load_config(*yamls: str, cli_args: list = [], from_string=False, **kwargs) -> Any:
    if from_string:
        yaml_confs = [OmegaConf.create(s) for s in yamls]
    else:
        yaml_confs = [OmegaConf.load(f) for f in yamls]
    cli_conf = OmegaConf.from_cli(cli_args)
    cfg = OmegaConf.merge(*yaml_confs, cli_conf, kwargs)
    OmegaConf.resolve(cfg)
    assert isinstance(cfg, DictConfig)
    scfg = parse_structured(ExperimentConfig, cfg)
    return scfg

def parse_structured(fields: Any, cfg: Optional[Union[dict, DictConfig]] = None) -> Any:
    scfg = OmegaConf.structured(fields(**cfg))
    return scfg