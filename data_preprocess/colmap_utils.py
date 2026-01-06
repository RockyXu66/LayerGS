from pathlib import Path
import os
import pickle
from loguru import logger

import numpy as np
from scipy.spatial.transform import Rotation
import torch

def save_torch3d_to_colmap_cam(folder, H, W, focal_length_y, focal_length_x, R, T, img_idx, image_name):
    '''
    Convert pytorch3d-format camera to colmap-format camera
    img_idx: starts from 1
    '''

    """
    The camera coordinate sysmte in COLMAP is right-down-forward
    Pytorch3D is left-up-forward
    """
    # extrinsic
    R, T = R.permute(1, 0), T.T         # convert row-major matrix (used in pytorch3d) to column-major matrix (colmap / opencv)
    w2c = torch.eye(4)                  # this 'w2c' is the world-to-camera matrix in torch3d (but column-major)
    w2c[:3, :3] = R
    w2c[:3, 3] = T
    new_c2w = torch.linalg.inv(w2c)     # this is camera-to-world matrix in torch3d

    # from pytorch3d's left-up-forward to colmap's right-down-forward
    # get colmap rotation matrix
    R = new_c2w[:3, :3]                 # this is rotation matrix for camera-to-world in torch3d
    T = new_c2w[:3, 3]
    R = torch.stack([-R[:, 0], -R[:, 1], R[:, 2]], 1)   # from LUF to RDF for Rotation (this is rotation matrix for camera-to-world in colmap)
    c2w = torch.eye(4)                  # this is camera-to-world matrix in colmap
    c2w[:3, :3] = R
    c2w[:3, 3] = T
    colmap_w2c = torch.inverse(c2w)     # this is world-to-camera matrix in colmap
    qvec = Rotation.from_matrix(colmap_w2c[:3, :3]).as_quat()
    qvec = np.array([qvec[3], qvec[0], qvec[1], qvec[2]])
    tvec = colmap_w2c[:3, 3]

    id = img_idx
    camera_id = 1
    xys = None
    point3d_ids = None

    # intrinsic
    id = 1
    model = 'PINHOLE'
    width = W
    height = H
    params = np.array([H/2*focal_length_y, W/2*focal_length_x, H/2, W/2])

    np.savez(os.path.join(folder, f'cam_{img_idx}.npz'), **{
        'width': width,
        'height': height,
        'params': params,
        'qvec': qvec,
        'tvec': tvec,
        'image_name': image_name,
    })

import numpy as np
import collections
import struct

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model)
                           for camera_model in CAMERA_MODELS])

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_extrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":   # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images


def read_intrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8*num_params,
                                     format_char_sequence="d"*num_params)
            cameras[camera_id] = Camera(id=camera_id,
                                        model=model_name,
                                        width=width,
                                        height=height,
                                        params=np.array(params))
        assert len(cameras) == num_cameras
    return cameras


# def save_colmap_to_pytorch3d(path_to_model_ext_file, path_to_model_intr_file):
def convert_colmap_to_pytorch3d(cam_extrinsics, cam_intrinsics):

    torch3d_cam_list = {}
    for k, v in cam_extrinsics.items():
        intrinsics = np.eye(3)
        intrinsics[0, 0] = cam_intrinsics[1].params[0]
        intrinsics[1, 1] = cam_intrinsics[1].params[1]
        intrinsics[0, 2] = cam_intrinsics[1].params[2]
        intrinsics[1, 2] = cam_intrinsics[1].params[3]

        extrinsics = np.eye(4)
        colmap_w2c = np.eye(4)
        qvec = v.qvec
        tvec = v.tvec
        colmap_w2c[:3, 3] = tvec
        qvec = np.array([qvec[1], qvec[2], qvec[3], qvec[0]])
        colmap_w2c[:3, :3] = Rotation.from_quat(qvec).as_matrix()
        c2w = np.linalg.inv(colmap_w2c)
        T = c2w[:3, 3]
        R = c2w[:3, :3]
        R = np.stack([-R[:, 0], -R[:, 1], R[:, 2]], axis=1)
        new_c2w = np.eye(4)
        new_c2w[:3, :3] = R
        new_c2w[:3, 3] = T
        w2c = np.linalg.inv(new_c2w)
        R = w2c[:3, :3]
        T = w2c[:3, 3]
        R = R.T
        T = T.T
        
        # scale = 1.75
        scale = 1
        T /= scale

        extrinsics[:3, :3] = R
        extrinsics[:3, 3] = T
        transformed_extrinsics = np.eye(4)


        torch3d_cam_list[f'{k-1:04d}'] = {
            'intrinsics': intrinsics,
            'extrinsics': extrinsics[:3],
            'extrinsics_transformed': transformed_extrinsics[:3],
            'name': v.name,
        }

    # sort by name
    torch3d_cam_list = dict(sorted(torch3d_cam_list.items(), key=lambda x: x[0]))
    return torch3d_cam_list

def convert_pytorch3d_to_colmap(cam_list):
    colmap_images_ext = {}
    colmap_images_intr = None
    for image_id, cam in cam_list.items():
        R = torch.from_numpy(cam['extrinsics'][:3, :3])
        T = torch.from_numpy(cam['extrinsics'][:3, 3])
        # extrinsic
        R, T = R.permute(1, 0), T.T         # convert row-major matrix (used in pytorch3d) to column-major matrix (colmap / opencv)
        w2c = torch.eye(4)                  # this 'w2c' is the world-to-camera matrix in torch3d (but column-major)
        w2c[:3, :3] = R
        w2c[:3, 3] = T
        new_c2w = torch.linalg.inv(w2c)     # this is camera-to-world matrix in torch3d

        # from pytorch3d's left-up-forward to colmap's right-down-forward
        # get colmap rotation matrix
        R = new_c2w[:3, :3]                 # this is rotation matrix for camera-to-world in torch3d
        T = new_c2w[:3, 3]
        R = torch.stack([-R[:, 0], -R[:, 1], R[:, 2]], 1)   # from LUF to RDF for Rotation (this is rotation matrix for camera-to-world in colmap)
        c2w = torch.eye(4)                  # this is camera-to-world matrix in colmap
        c2w[:3, :3] = R
        c2w[:3, 3] = T
        colmap_w2c = torch.inverse(c2w)     # this is world-to-camera matrix in colmap
        qvec = Rotation.from_matrix(colmap_w2c[:3, :3]).as_quat()
        qvec = np.array([qvec[3], qvec[0], qvec[1], qvec[2]])
        tvec = colmap_w2c[:3, 3]
        colmap_images_ext[int(image_id)+1] = Image(
            id=int(image_id)+1, qvec=qvec, tvec=tvec,
            camera_id=1, name=cam['name'],
            xys=np.array([[0,0]]), point3D_ids=np.array([0])) 
    return colmap_images_ext, colmap_images_intr

def write_extrinsics_binary(path_to_model_file, images):
    """
    Writes the camera extrinsics (images) to a binary file with the same format
    as read_extrinsics_binary() expects.
    
    The binary file layout is:
      1. 8-byte unsigned long long: number of images.
      2. For each image:
         a. 64 bytes containing:
            - image id (int, 4 bytes)
            - qvec (4 doubles, 4*8 bytes)
            - tvec (3 doubles, 3*8 bytes)
            - camera id (int, 4 bytes)
         b. The image name as a null-terminated UTF-8 string.
         c. 8-byte unsigned long long: number of 2D points.
         d. For each 2D point:
            - x (double, 8 bytes)
            - y (double, 8 bytes)
            - point3D_id (long long, 8 bytes)
    """
    with open(path_to_model_file, "wb") as fid:
        # Write the number of images (as an unsigned long long "<Q")
        num_reg_images = len(images)
        fid.write(struct.pack("<Q", num_reg_images))
        
        # It is a good idea to write images in a deterministic order;
        # here we sort them by image id.
        for image_id in sorted(images.keys()):
            image = images[image_id]
            # Write the 64-byte block of image properties using format "<idddddddi"
            # where the fields are: image_id, qvec[0:4], tvec[0:3], camera_id.
            tvec = np.array(image.tvec)
            # scale = 1.88
            # tvec /= scale
            fid.write(struct.pack("<idddddddi",
                                  image.id,
                                  image.qvec[0], image.qvec[1], image.qvec[2], image.qvec[3],
                                  tvec[0], tvec[1], tvec[2],
                                  image.camera_id))
            # Write the image name as a UTF-8 encoded string, followed by a null terminator.
            fid.write(image.name.encode("utf-8") + b"\x00")
            
            # Write the number of 2D points (unsigned long long "<Q")
            num_points2D = len(image.point3D_ids)
            fid.write(struct.pack("<Q", num_points2D))
            
            # Write the 2D point data if there are any points.
            # For each point, we need to write x (double), y (double), and point3D_id (long long).
            if num_points2D > 0:
                # Prepare a flat list of numbers in the proper order.
                data = []
                for i in range(num_points2D):
                    data.append(float(image.xys[i, 0]))  # x coordinate
                    data.append(float(image.xys[i, 1]))  # y coordinate
                    data.append(int(image.point3D_ids[i]))  # corresponding 3D point id
                # Build the format string: "ddq" repeated num_points2D times.
                fmt = "<" + "ddq" * num_points2D
                fid.write(struct.pack(fmt, *data))

def write_intrinsics_binary(path_to_output, cameras):
    """
    Writes the camera intrinsics to a binary file using the same format as read_intrinsics_binary.
    
    The binary format written is:
      - Number of cameras (unsigned long long, 'Q').
      - For each camera:
          - Camera id (int, 'i')
          - Model id (int, 'i')
          - Width (unsigned long long, 'Q')
          - Height (unsigned long long, 'Q')
          - Camera parameters (num_params doubles, 'd' * num_params)
    
    :param path_to_output: Path to the output binary file.
    :param cameras: Dictionary mapping camera ids to Camera namedtuples.
    """
    with open(path_to_output, "wb") as fid:
        # Write the number of cameras.
        fid.write(struct.pack("<Q", len(cameras)))
        
        # Loop over cameras. Sorting by camera id ensures a consistent order.
        for camera_id, camera in sorted(cameras.items()):
            # Retrieve the model id based on the camera's model name.
            model_id = CAMERA_MODEL_NAMES[camera.model].model_id
            
            # Pack camera properties: id, model_id, width, height.
            fid.write(struct.pack("<iiQQ", camera.id, model_id, camera.width, camera.height))
            
            # Retrieve the number of parameters for this model.
            num_params = CAMERA_MODEL_NAMES[camera.model].num_params
            
            # Ensure the parameters are in a list/tuple format.
            params_list = camera.params.tolist() if isinstance(camera.params, np.ndarray) else camera.params
            
            # Pack and write the camera parameters (doubles).
            fid.write(struct.pack("<" + "d" * num_params, *params_list))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ext_file', type=str, help='extrinsics bin file from COLMAP')
    parser.add_argument('--intr_file', type=str, help='intrinsics bin file from COLMAP')
    parser.add_argument('--mode', type=str, default='', help='operation mode')
    args = parser.parse_args()

    # ============ Convert COLMAP to torch3d cam_list ================
    if args.mode == 'colmap2torch3d':
        cam_extrinsics_colmap = read_extrinsics_binary(args.ext_file)
        cam_intrinsics_colmap = read_intrinsics_binary(args.intr_file)
        torch3d_cam_list = convert_colmap_to_pytorch3d(cam_extrinsics_colmap, cam_intrinsics_colmap)
        torch3d_cam_file = f'{Path(args.ext_file).parent.parent.parent}/colmap_to_torch3d_cam_list.pkl'
        with open(torch3d_cam_file, 'wb') as file:
            pickle.dump(torch3d_cam_list, file)
        logger.info(f'Save the pytorch3d cameras in {torch3d_cam_file}')

    # ============ Load cam_list.pkl file and convert to COLMAP binary file =============
    elif args.mode == 'campkl2colmap':
        # Load cam_list.pkl file
        torch3d_cam_list = pickle.load(open(f'{Path(args.ext_file).parent.parent.parent}/colmap_to_torch3d_cam_list_transformed.pkl', 'rb'))
        # torch3d_cam_list = pickle.load(open(f'{Path(args.ext_file).parent.parent.parent}/colmap_to_torch3d_cam_list.pkl', 'rb'))

        cam_extrinsics_colmap_new, cam_intrinsics_colmap_new = convert_pytorch3d_to_colmap(torch3d_cam_list)

        cameras_extrinsic_file_new = args.ext_file.replace('.bin', '_new.bin')
        write_extrinsics_binary(cameras_extrinsic_file_new, cam_extrinsics_colmap_new)
        logger.info(f'Write the new transformed extrinsics in "{cameras_extrinsic_file_new}"')
    else:
        logger.error(f'Error: unknown mode "{args.mode}"')