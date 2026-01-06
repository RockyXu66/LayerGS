import os
from pathlib import Path
import math
import numpy
import numpy as np
import cv2

import torch
import torchvision

torch_new_generator = torch.Generator(device='cuda')
torch_new_generator.manual_seed(99)

def reset_random_generators(seed: int = 0):
    """Reset module-level CUDA random generators for reproducibility.
    
    This function should be called before training starts to ensure
    reproducible random number generation across training runs.
    """
    global torch_new_generator
    torch_new_generator = torch.Generator(device='cuda')
    torch_new_generator.manual_seed(seed)

from gala_utils import helpers
from gala_utils import rotation_converter
from gala_utils.op_3d import OpenPoseDetectorRaw
from gala_utils.sd import StableDiffusion

from omegaconf import OmegaConf
from torch.cuda.amp import custom_bwd, custom_fwd 
import random
from layergs.utils.smplx_utils import SMPLX

def joint_mapper_smplx_to_openpose18(joints):
    indices = np.array([
        56, # nose
        13, # neck
        18, # right_shoulder
        20, # right_elbow
        22, # right_wrist
        17, # left_shoulder
        19, # left_elbow
        21, # left_wrist
        3,  # right_hip
        6,  # right_knee
        9,  # right_ankle
        2,  # left_hip
        5,  # left_knee
        8,  # left_ankle
        57, # right_eye
        58, # left_eye
        59, # right_ear
        60, # left_ear
    ], dtype=np.int64) - 1
    return joints[indices]

def seed_everything(seed, local_rank):
    random.seed(seed + local_rank)
    os.environ['PYTHONHASHSEED'] = str(seed)
    numpy.random.seed(seed + local_rank)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.benchmark = True  
    
class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None

class sds_controlnet:

    def __init__(self, use_opengl_camera = True, smplx_file_path = '', dataset_type=None, prompt_type='inner', data_cfg=None, no_hand_pose=False, ref_smplx_file_path=None) -> None:

        self.use_opengl_camera = use_opengl_camera   # whether to use opengl camera or blender camera

        config_base = OmegaConf.load('gala_config/base.yaml')
        # specify config file

        if prompt_type == 'inner':
            cfg_sub = 'gala_config/inner_tex.yaml'
        elif prompt_type == 'outer':
            cfg_sub = 'gala_config/outer_tex.yaml'
        config_subject = OmegaConf.load(cfg_sub)

        # prevent config file info being saved to config_cli
        config_parent = OmegaConf.load(f'gala_config/{config_subject.parent}.yaml')

        # update from cli  e.g. exp_name="test something"
        config_cli = OmegaConf.from_cli()

        FLAGS = OmegaConf.merge(config_base, config_parent, config_subject, config_cli)
        config_for_save = OmegaConf.merge(config_parent, config_subject, config_cli)

        FLAGS.mtl_override        = None                     # Override material of model                   
        FLAGS.env_scale           = 2.0                      # Env map intensity multiplier
        FLAGS.relight             = None                     # HDR environment probe(relight)
        FLAGS.camera_space_light  = False                    # Fixed light in camera space. This is needed for setups like ethiopian head where the scanned object rotates on a stand.
        FLAGS.lock_light          = False                    # Disable light optimization in the second pass
        FLAGS.lock_pos            = False                    # Disable vertex position optimization in the second pass
        FLAGS.pre_load            = True                     # Pre-load entire dataset into memory for faster training
        FLAGS.kd_min              = [ 0.0,  0.0,  0.0,  0.0] # Limits for kd
        FLAGS.kd_max              = [ 1.0,  1.0,  1.0,  1.0]
        FLAGS.ks_min              = [ 0.0, 0.08,  0.0]       # Limits for ks
        FLAGS.ks_max              = [ 1.0,  1.0,  1.0]
        FLAGS.nrm_min             = [-1.0, -1.0,  0.0]       # Limits for normal map
        FLAGS.nrm_max             = [ 1.0,  1.0,  1.0]
        FLAGS.cam_near_far        = [1, 50]
        FLAGS.learn_light         = False
        FLAGS.gpu_number          = 1
        FLAGS.sdf_init_shape_scale=[1.0, 1.0, 1.0]
        # FLAGS.local_rank = 0
        FLAGS.multi_gpu  = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1 

        if FLAGS.multi_gpu:
            FLAGS.gpu_number = int(os.environ["WORLD_SIZE"])
            FLAGS.local_rank = int(os.environ["LOCAL_RANK"])
            torch.distributed.init_process_group(backend="nccl", world_size = FLAGS.gpu_number, rank = FLAGS.local_rank)  
            torch.cuda.set_device(FLAGS.local_rank)

        if FLAGS.display_res is None:
            FLAGS.display_res = FLAGS.train_res
        if FLAGS.out_dir is None:
            FLAGS.out_dir = 'out/cube_%d' % (FLAGS.train_res)
        else:
            # FLAGS.out_dir = Path(FLAGS.out_dir) / FLAGS.subject / FLAGS.mode / \
            #                 (datetime.strftime(datetime.now(), "%Y%m%d-%H%M_") + '_'.join(FLAGS.exp_name.split()))
            FLAGS.out_dir = Path(FLAGS.out_dir) / FLAGS.subject / FLAGS.mode / \
                            ('_'.join(FLAGS.exp_name.split()))

        FLAGS.data_dir = Path(FLAGS.data_dir) / FLAGS.subject
        mesh_name = FLAGS.subject
        FLAGS.gt_mesh             = FLAGS.data_dir / f'{mesh_name}_norm.obj'
        FLAGS.gt_mtl              = FLAGS.data_dir / f'material0.mtl'
        FLAGS.geometry_dir = FLAGS.out_dir / FLAGS.exp_name if FLAGS.geometry_dir is None \
                        else FLAGS.out_dir.parent.parent / 'geometry_modeling' / FLAGS.geometry_dir
        FLAGS.base_mesh           = FLAGS.geometry_dir / f'dmtet_mesh/human.obj'
        FLAGS.base_mesh_obj       = FLAGS.geometry_dir / f'dmtet_mesh/object.obj'
        FLAGS.faces_segm          = FLAGS.data_dir / 'faces_segms.npy'
        FLAGS.smplx               = FLAGS.data_dir / 'thuman20_smplx_param.pkl'

        self.FLAGS = FLAGS

        if FLAGS.local_rank == 0:
            print("Config / Flags:")
            print("---------")
            for key in FLAGS.__dict__.keys():
                print(key, FLAGS.__dict__[key])
            print("---------")

        seed_everything(FLAGS.seed, FLAGS.local_rank)

        if data_cfg is not None:
            FLAGS.text = data_cfg.text
            FLAGS.negative_text = data_cfg.negative_text
            FLAGS.text_obj = data_cfg.text_obj
            FLAGS.negative_text_obj = data_cfg.negative_text_obj
            FLAGS.add_directional_text = data_cfg.add_directional_text
            gender = data_cfg.gender
        else:
            gender = 'male' # default

        # FLAGS.out_dir.mkdir(exist_ok=True, parents=True)
        # OmegaConf.save(config_for_save, FLAGS.out_dir / 'config.yaml')
        self.guidance = StableDiffusion(device = 'cuda',
                                    mode = FLAGS.mode, 
                                    text = FLAGS.text,
                                    text_comp = FLAGS.text_obj,
                                    add_directional_text = FLAGS.add_directional_text,
                                    batch = FLAGS.batch,
                                    guidance_weight = FLAGS.guidance_weight,
                                    sds_weight_strategy = FLAGS.sds_weight_strategy,
                                    early_time_step_range = FLAGS.early_time_step_range,
                                    late_time_step_range= FLAGS.late_time_step_range,
                                    negative_text = FLAGS.negative_text,
                                    negative_text_comp = FLAGS.negative_text_obj,
                                    sd_model=FLAGS.sd_model,
                                    enable_controlnet=FLAGS.enable_controlnet,
                                    use_inpaint=FLAGS.use_inpaint,
                                    repaint=FLAGS.repaint,
                                    use_legacy=FLAGS.use_legacy,
                                    use_taesd=FLAGS.use_taesd)
        self.guidance.eval()
        for p in self.guidance.parameters():
            p.requires_grad_(False)

        self.img_resize_transform = torchvision.transforms.Resize((512, 512))

        class Struct:
            def __init__(self, **entries):
                self.__dict__.update(entries)
        smplx_data_folder = './deformer'
        smplx_config = {
            'topology_path': f"{smplx_data_folder}/data/SMPL_X_template_FLAME_uv.obj",
            'smplx_model_path': f'data/models/smplx/SMPLX_{gender.upper()}.npz',
            'extra_joint_path': f"{smplx_data_folder}/data/smplx_extra_joints.yaml",
            'j14_regressor_path': f"{smplx_data_folder}/data/SMPLX_to_J14.pkl",
            'mano_ids_path': f"{smplx_data_folder}/data/MANO_SMPLX_vertex_ids.pkl",
            'flame_vertex_masks_path': f"{smplx_data_folder}/data/FLAME_masks.pkl",
            'flame_ids_path': f"{smplx_data_folder}/data/SMPL-X__FLAME_vertex_ids.npy",
            'n_shape': 10,
            'n_exp': 10
        }
        assert dataset_type != None
        self.Y_OFFSET = 0.4
        if dataset_type == 'pytorch3d':
            self.OFFSET = np.array([0, 0.4, 0])
        elif dataset_type == 'colmap':
            self.OFFSET = np.array([0, 0.4, 0]) - np.array([-0.00464579, -0.14872813, -0.01218675])
        self.OFFSET = list(self.OFFSET)
        # scale = 1.93
        # self.Y_OFFSET /= scale
        # self.Y_OFFSET = 0.00
        smplx_config = Struct(**smplx_config)
        self.op_3d_mapping = numpy.array([68, 12, 17, 19, 21, 16, 18, 20, 2, 5, 8, 1, 4, 7, 75, 82, 106, 125], dtype=numpy.int32)

        self.smplx_model = SMPLX(smplx_config)


        # A pose smplx
        a_pose = torch.zeros([55,3], dtype=torch.float32, ) # 55
        # legs
        angle = 15*np.pi/180.
        a_pose[1, 2] = angle
        a_pose[2, 2] = -angle
        # arms
        angle = 0*np.pi/180.
        a_pose[13, 2] = angle
        a_pose[14, 2] = -angle
        # waist
        angle = 0*np.pi/180.
        a_pose[3, 2] = angle
        a_pose[6, 2] = angle
        pose_euler = rotation_converter.batch_euler2axis(a_pose)
        a_pose_mat = rotation_converter.batch_euler2matrix(a_pose)
        a_pose_mat = a_pose_mat[None,...]

        # GT posed smplx
        smplx_params = numpy.load(smplx_file_path, allow_pickle=True)
        smplx_params = {k: torch.from_numpy(v).float() for k, v in smplx_params.items()}
        self.gt_beta = smplx_params['betas'].squeeze()[:10]
        body_pose = smplx_params['body_pose'] if smplx_params['body_pose'].shape[1] == 63 else smplx_params['body_pose'][:, 3:]
        body_pose = body_pose.to(torch.float32) # prevent type confliction

        if no_hand_pose:
            # smplx_params['left_hand_pose'][:, :6] = 0.0
            # smplx_params['right_hand_pose'][:, :6] = 0.0

            ref_smplx_params = numpy.load(ref_smplx_file_path, allow_pickle=True)
            ref_smplx_params = {k: torch.from_numpy(v).float() for k, v in ref_smplx_params.items()}
            self.gt_beta = ref_smplx_params['betas'].squeeze()[:10]

            # xyz_c, _, joints_c, A, T, shape_offsets, pose_offsets = self.smplx_model(full_pose = a_pose_mat, return_T=True, transl=torch.tensor(self.OFFSET, dtype=torch.float32, ), shape_params=self.gt_beta[None])
            xyz_c, _, joints_c, A, T, shape_offsets, pose_offsets = self.smplx_model(full_pose = a_pose_mat, return_T=True, transl=torch.tensor(self.OFFSET, dtype=torch.float32, ))
        else:
            xyz_c, _, joints_c, A, T, shape_offsets, pose_offsets = self.smplx_model(full_pose = a_pose_mat, return_T=True, transl=torch.tensor(self.OFFSET, dtype=torch.float32, ))
        A_inv = A.squeeze(0).inverse()
        self.smplx_verts_apose = xyz_c.squeeze()
        self.joints_c = joints_c

        # # Use non-pca for thuman2.1 following the data structure explained at:
        # # https://github.com/ytrock/THuman2.0-Dataset?tab=readme-ov-file#data-explanation
        # if hasattr(data_cfg, 'dataset_name') and data_cfg.dataset_name == 'thuman2.1':
        #     self.gt_full_pose = torch.cat(
        #         [
        #             smplx_params['global_orient'], 
        #             body_pose,
        #             smplx_params['jaw_pose'], 
        #             smplx_params['leye_pose'], 
        #             smplx_params['reye_pose'],  
        #             # smplx_params['left_hand_pose'][:, :6], 
        #             # smplx_params['right_hand_pose'][:, :6],
        #             smplx_params['left_hand_pose'], 
        #             smplx_params['right_hand_pose'],
        #         ], dim=1)
        #     self.gt_exp = smplx_params['expression'].squeeze()[:10]

        #     xyz, _, joints, A, T, shape_offsets, pose_offsets = self.smplx_model(
        #         full_pose=self.gt_full_pose[None], shape_params=self.gt_beta[None], return_T=True, 
        #         transl=torch.tensor(self.OFFSET,dtype=torch.float32)[None],
        #         expression_params=self.gt_exp[None], 
        #         # axis_pca=True,
        #         axis_pca=False,
        #     )
        # else:
        #     self.gt_full_pose = torch.cat(
        #         [
        #             smplx_params['global_orient'], 
        #             body_pose,
        #             smplx_params['jaw_pose'], 
        #             smplx_params['leye_pose'], 
        #             smplx_params['reye_pose'],  
        #             smplx_params['left_hand_pose'], 
        #             smplx_params['right_hand_pose'],
        #         ], dim=1)
        #     self.gt_exp = smplx_params['expression'].squeeze()[:10]

        #     xyz, _, joints, A, T, shape_offsets, pose_offsets = self.smplx_model(
        #         full_pose=self.gt_full_pose[None], shape_params=self.gt_beta[None], return_T=True, 
        #         transl=torch.tensor(self.OFFSET,dtype=torch.float32)[None],
        #         expression_params=self.gt_exp[None], 
        #         axis_pca=True,
        #     )

        self.gt_full_pose = torch.cat([smplx_params['global_orient'], body_pose,
                            smplx_params['jaw_pose'], smplx_params['leye_pose'], smplx_params['reye_pose'],  
                            smplx_params['left_hand_pose'][:, :6], smplx_params['right_hand_pose'][:, :6]], dim=1)
        self.gt_exp = smplx_params['expression'].squeeze()[:10]

        xyz, _, joints, A, T, shape_offsets, pose_offsets = self.smplx_model(full_pose=self.gt_full_pose[None], shape_params=self.gt_beta[None], return_T=True, 
                                                        transl=torch.tensor(self.OFFSET,dtype=torch.float32)[None],
                                                        expression_params=self.gt_exp[None], axis_pca=True)
        gt_A = A
        self.gt_smplx_tfs = torch.einsum('bnij,njk->bnik', gt_A, A_inv).cuda()
        # self.gt_smplx_tfs = self.apose_smplx_tfs.clone()
        gt_smplx_offsets = shape_offsets + pose_offsets

        # A pose modified smplx
        sds_pose = torch.zeros([55,3], dtype=torch.float32, ) # 55
        # legs
        angle = 15*np.pi/180.
        sds_pose[1, 2] = angle
        sds_pose[2, 2] = -angle
        # arms
        angle = -25*np.pi/180.
        sds_pose[13, 2] = angle
        sds_pose[14, 2] = -angle
        # waist
        angle = 0*np.pi/180.
        sds_pose[3, 2] = angle
        sds_pose[6, 2] = angle
        sds_pose = rotation_converter.batch_euler2matrix(sds_pose)
        sds_pose = sds_pose[None,...]
        # _, _, joints_sds, sds_A, _, _ = self.smplx_model(full_pose=sds_pose, shape_params=self.gt_beta[None], return_T=True, 
        #                                                 transl=torch.tensor([0, self.Y_OFFSET, 0],dtype=torch.float32)[None],
        #                                                 expression_params=self.gt_exp[None], axis_pca=True)
        _, _, joints_sds, sds_A, _, _, _ = self.smplx_model(full_pose = sds_pose, return_T=True, transl=torch.tensor(self.OFFSET,dtype=torch.float32, ))
        self.sds_smplx_tfs = torch.einsum('bnij,njk->bnik', sds_A, A_inv).cuda()


        
        # canonical_joints = joints_c.squeeze()
        # canonical_transform = T
        # canonical_offsets = shape_offsets + pose_offsets
        # smplx_verts_cano = v_template
        # smplx_faces = self.smplx_model.faces_tensor

        d, h, w = 64, 256, 256
        grid = helpers.create_voxel_grid(d, h, w, device='cuda') # precompute lbs grid
        self.shape_pose_offsets_grid = helpers.query_weights_smpl(grid, self.smplx_verts_apose.cuda(), gt_smplx_offsets[0].cuda()).permute(0,2,1).reshape(1,-1,d,h,w)
        self.lbs_weights_grid = helpers.query_weights_smpl(grid, self.smplx_verts_apose.cuda(), self.smplx_model.lbs_weights.cuda()).permute(0,2,1).reshape(1,-1,d,h,w)





        # This is from HumanGaussian
        import smplx 

        smplx_model = smplx.create(
            'data/models', 
            model_type='smplx',
            gender=gender, 
            use_face_contour=False,
            num_betas=10,
            num_expression_coeffs=10,
            ext='npz',
            flat_hand_mean=True,
        )

        body_pose = np.zeros((21, 3), dtype=np.float32)

        smplx_output = smplx_model(
            body_pose=torch.tensor(body_pose, dtype=torch.float32).unsqueeze(0),
            betas=self.gt_beta[None], 
            expression=None, 
            return_verts=True,
            # translate=torch.tensor([0, self.Y_OFFSET, 0],dtype=torch.float32, ),
        )
        joints = smplx_output.joints.detach().cpu().numpy()[0] # [127, 3]
        joints = joint_mapper_smplx_to_openpose18(joints)
        self.points3D = np.concatenate([joints, np.ones_like(joints[:, :1])], axis=1) # [18, 4]
        self.vertices = smplx_output.vertices.detach().cpu().numpy()[0] # [10475, 3]

        import trimesh
        smplx_genereated_mesh = trimesh.Trimesh(self.vertices, smplx_model.faces)
        os.makedirs('debug', exist_ok=True)
        smplx_genereated_mesh.export('debug/smplx_genereated.obj')

        smplx_apose_mesh = trimesh.Trimesh(self.smplx_verts_apose, smplx_model.faces)
        smplx_apose_mesh.export('debug/smplx_apose.obj')


        # Save the apose smplx obj model for the postprocessing
        smplx_output = smplx_model(
            body_pose=a_pose[1:22].unsqueeze(0),
            betas=self.gt_beta[None], 
            expression=None, 
            return_verts=True,
            # translate=torch.tensor([0, self.Y_OFFSET, 0],dtype=torch.float32, ),
        )
        xyz_c, _, _, _, _, _, _ = self.smplx_model(shape_params=self.gt_beta[None], full_pose = a_pose_mat, return_T=True, transl=torch.tensor(self.OFFSET,dtype=torch.float32, ))
        apose_verts = xyz_c[0].detach().cpu().numpy()
        apose_faces = smplx_model.faces
        apose_model = trimesh.Trimesh(vertices=apose_verts, faces=apose_faces)
        apose_model_path = f'{Path(smplx_file_path).parent}/apose.obj'
        apose_model.export(apose_model_path)
        print(f'Save the apose smplx obj model in {apose_model_path}')


        # Use smplx apose as canonical pose
        self.vertices = self.smplx_verts_apose
        apose_joints = self.joints_c.detach().cpu().numpy()[0]
        # apose_joints = joint_mapper_smplx_to_openpose18(apose_joints)
        apose_joints = apose_joints[self.op_3d_mapping]
        self.apose_points3D = np.concatenate([apose_joints, np.ones_like(apose_joints[:, :1])], axis=1) # [18, 4]

        # Use smplx sds pose (modified from apose) for sds loss training
        sdspose_joints = joints_sds.detach().cpu().numpy()[0]
        sdspose_joints = sdspose_joints[self.op_3d_mapping]
        self.sds_points3D = np.concatenate([sdspose_joints, np.ones_like(sdspose_joints[:, :1])], axis=1) # [18, 4]

        self.faces = smplx_model.faces # [20908, 3]

        if not self.use_opengl_camera:  # use blender camera
            # coordinate system: opengl --> blender (switch y/z)
            self.vertices[:, [1, 2]] = self.vertices[:, [2, 1]]
            self.apose_points3D[:, [1, 2]] = self.apose_points3D[:, [2, 1]]
            self.sds_points3D[:, [1, 2]] = self.sds_points3D[:, [2, 1]]
        self.points3D = self.apose_points3D.copy()

        # lines [17, 2]
        self.lines = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13], [0, 14], [14, 16], [0, 15], [15, 17]], dtype=np.int32)

        # keypoint color [18, 3]
        # color as in controlnet_aux (https://github.com/patrickvonplaten/controlnet_aux/blob/master/src/controlnet_aux/open_pose/util.py#L94C5-L96C73)
        self.colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
                       [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
                       [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]



        
    def get_op_3d(self, pose_type='apose'):

        if pose_type == 'tpose':
            # convert posed smplx to canonical smplx
            full_pose = self.gt_full_pose.clone()
            full_pose[:, :-12] = 0.0
        elif pose_type == 'apose':
            # A pose
            full_pose = torch.zeros([55,3], dtype=torch.float32, ) # 55
            # legs
            angle = 15*np.pi/180.
            full_pose[1, 2] = angle
            full_pose[2, 2] = -angle
            # arms
            angle = 0*np.pi/180.
            full_pose[13, 2] = angle
            full_pose[14, 2] = -angle
            # waist
            angle = 0*np.pi/180.
            full_pose[3, 2] = angle
            full_pose[6, 2] = angle
            pose_euler = rotation_converter.batch_euler2axis(full_pose)
            # pose = rotation_converter.batch_euler2matrix(pose)
            # pose = pose[None,...]

        T_xyz, _, T_joints, T_A, T_Trans_mat, T_shape_offsets, T_pose_offsets = self.smplx_model(full_pose=full_pose[None], shape_params=self.gt_beta[None], return_T=True, 
                                                                # transl=torch.tensor([0, 0, 0],dtype=torch.float32)[None],
                                                                transl=torch.tensor(self.OFFSET,dtype=torch.float32)[None],
                                                                expression_params=self.gt_exp[None], axis_pca=True)
        use_op_control = True
        if use_op_control:
            op_3d = T_joints[:, self.op_3d_mapping] # get from gt SMPLX joints
            op_3d = op_3d[0]

        else:
            op_3d = None
        return op_3d

    def get_op_img(self, op_kpts_img_xy, img_size):

        sd_img_size = 512
        op_img = [OpenPoseDetectorRaw.draw_bodypose((op_3d_rot/img_size*2) - 1, H=sd_img_size, W=sd_img_size)[None] for op_3d_rot in op_kpts_img_xy.cpu()]  # multi-view openpose images
        # op_img = [OpenPoseDetectorRaw.draw_bodypose_wo_face((op_3d_rot/img_size*2) - 1)[None] for op_3d_rot in op_kpts_img_xy.cpu()]  # multi-view openpose images
        op_img = torch.cat(op_img)

        return op_img
    
    def compute_sds_loss(self, img_size, rendered_imgs, mvp, azimuth, comp=False):

        op_img_list = []
        prompt_index_list = []
        for id in range(azimuth.shape[0]):
            backview = abs(azimuth[id]) > 120 * np.pi / 180
            angle = azimuth[id].detach().cpu().numpy()
            angle = (angle - 90) if (angle - 90) > -180 else -(angle - 90) - 180
            prompt_index = get_view_direction(np.deg2rad(angle), front=np.deg2rad(45))
            prompt_index_list.append(prompt_index)
            
            op_3d = self.points3D
            op_img, _ = self.draw_skel(op_3d, mvp[id], 512, 512, enable_occlusion=backview)
            op_img = torch.from_numpy(op_img).to(rendered_imgs[0].device) # [H, W, 3]
            # op_img = op_img[None]
            # op_img = op_img.permute(0, 3,1,2)
            op_img = op_img.permute(2, 0, 1)
            op_img_list.append(op_img)
        op_img_list = torch.stack(op_img_list, dim=0)
        prompt_index_list = np.stack(prompt_index_list)

        if self.FLAGS.add_directional_text:
            if comp:
                text_embeddings = torch.cat([self.guidance.uncond_z_comp[prompt_index_list], self.guidance.text_z_comp[prompt_index_list]])
            else:
                text_embeddings = torch.cat([self.guidance.uncond_z[prompt_index_list], self.guidance.text_z[prompt_index_list]])
        else:
            # prompt_index = [0] * op_img.shape[0]
            text_embeddings = torch.cat([self.guidance.uncond_z.repeat(op_img_list.shape[0], 1, 1), self.guidance.text_z.repeat(op_img_list.shape[0], 1, 1)])
        t = torch.randint(self.guidance.min_step_early, self.guidance.max_step_early+1, [op_img_list.shape[0]], dtype=torch.long, device='cuda', generator=torch_new_generator) # [B]

        pred_rgb_512 = self.img_resize_transform(rendered_imgs).contiguous().to(torch.float16) # [1, 3, H, W]
        if self.guidance.use_legacy:
            latents = self.guidance.encode_imgs(pred_rgb_512)
        control_images = op_img_list.to(torch.float16)

        '''
        import cv2; import numpy;
        bgr_arr = numpy.transpose(op_img_list[0].detach().cpu().numpy(), (1, 2, 0))
        cv2.imwrite('debug/control_images_gs.png', bgr_arr[:,:,::-1]*255)

        bgr_arr = numpy.transpose(rendered_imgs[0].detach().cpu().numpy(), (1, 2, 0))
        bgr_arr = cv2.resize(bgr_arr, (512, 512))
        cv2.imwrite('debug/image.png', bgr_arr[:,:,::-1]*255)
        '''
        
        # generate noise and predict noise
        # noise = torch.randn_like(latents)
        noise = torch.empty_like(latents).normal_(generator=torch_new_generator)
        noise_pred = self.guidance(latents, noise, t, text_embeddings.to(torch.float16), control_images=control_images)
        w = self.guidance.alphas[t] ** 0.5 * (1 - self.guidance.alphas[t])
        w = w[:, None, None, None] # [B, 1, 1, 1]
        grad =  w * (noise_pred - noise) #*w1 s
        grad = torch.nan_to_num(grad)
        sds_loss = SpecifyGradient.apply(latents, grad)

        return sds_loss

    def draw_skel(self, points3D, mvp, H, W, enable_occlusion=False):
        # mvp: [4, 4]
        canvas = np.zeros((H, W, 3), dtype=np.uint8)

        points = points3D @ mvp.T # [18, 4]
        points = points[:, :3] / points[:, 3:] # NDC in [-1, 1]

        xs = (points[:, 0] + 1) / 2 * H # [18]
        ys = (points[:, 1] + 1) / 2 * W # [18]
        mask = (xs >= 0) & (xs < H) & (ys >= 0) & (ys < W)

        # hide certain keypoints based on empirical occlusion
        if enable_occlusion:
            # decide view by the position of nose between two ears
            if points[0, 2] > points[-1, 2] and points[0, 2] < points[-2, 2]:
                # left view
                mask[-2] = False # no right ear
                if xs[-4] > xs[-3]:
                    mask[-4] = False # no right eye if it's "righter" than left eye
            elif points[0, 2] < points[-1, 2] and points[0, 2] > points[-2, 2]:
                # right view
                mask[-1] = False
                if xs[-3] < xs[-4]:
                    mask[-3] = False
            elif points[0, 2] > points[-1, 2] and points[0, 2] > points[-2, 2]:
                # back view
                mask[0] = False # no nose
                mask[-3] = False # no eyes
                mask[-4] = False

        # 18 points
        for i in range(18):
            if not mask[i]: continue
            cv2.circle(canvas, (int(xs[i]), int(ys[i])), 4, self.colors[i], thickness=-1)

        # 17 lines
        for i in range(17):
            cur_canvas = canvas.copy()
            if not mask[self.lines[i]].all(): 
                continue
            X = xs[self.lines[i]]
            Y = ys[self.lines[i]]
            mY = np.mean(Y)
            mX = np.mean(X)
            length = ((Y[0] - Y[1]) ** 2 + (X[0] - X[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
            polygon = cv2.ellipse2Poly((int(mX), int(mY)), (int(length / 2), 4), int(angle), 0, 360, 1)
            
            cv2.fillConvexPoly(cur_canvas, polygon, self.colors[i])
            
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
        
        canvas = canvas.astype(np.float32) / 255
        return canvas, np.stack([xs, ys], axis=1)
    
@torch.no_grad()
def get_view_direction(phis, front):
    #                   phis [B,];  -pi~pi        thetas: [B,] -pi/2~pi/2 
    # front = 0         [-front, front) 
    # side (left) = 1   [front, pi - front)
    # back = 2          [pi - front, pi) or [-pi, -pi+front)
    # side (right) = 3  [-pi+front, - front)
    
    if (phis >= -front) and (phis < front) :
        prompt_index = 0
    elif  (phis >= front ) and (phis < np.pi - front ):
        prompt_index = 1
    elif (phis >= np.pi - front) or  (phis < -np.pi + front):
        prompt_index = 2
    elif (phis >= -np.pi + front) and (phis < -front):
        prompt_index = 3
    
    return prompt_index