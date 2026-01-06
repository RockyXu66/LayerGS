from controlnet_aux import OpenposeDetector
from controlnet_aux.util import HWC3, resize_image
from pathlib import Path
from torchvision import transforms
import warnings
import numpy as np
import torch
import math
from scipy.spatial.transform import Rotation as R
import cv2

class OpenPoseDetectorRaw(OpenposeDetector): 
    limbSeq = [
        [2, 3], [2, 6], [3, 4], [4, 5], 
        [6, 7], [7, 8], [2, 9], [9, 10], 
        [10, 11], [2, 12], [12, 13], [13, 14], 
        [2, 1], [1, 15], [15, 17], [1, 16], 
        [16, 18],
    ]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
                [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
                [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]   

    # TODO: batch implementation
    def __call__(self, input_image, detect_resolution=512, image_resolution=512, 
                 include_body=True, include_hand=False, include_face=False, hand_and_face=None, 
                 output_type="pil", **kwargs):
        if hand_and_face is not None:
            warnings.warn("hand_and_face is deprecated. Use include_hand and include_face instead.", DeprecationWarning)
            include_hand = hand_and_face
            include_face = hand_and_face

        if "return_pil" in kwargs:
            warnings.warn("return_pil is deprecated. Use output_type instead.", DeprecationWarning)
            output_type = "pil" if kwargs["return_pil"] else "np"
        if type(output_type) is bool:
            warnings.warn("Passing `True` or `False` to `output_type` is deprecated and will raise an error in future versions")
            if output_type:
                output_type = "pil"

        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)

        input_image = resize_image(input_image, detect_resolution)
        H, W, C = input_image.shape
        
        poses = self.detect_poses(input_image, include_hand, include_face)

        return poses

    @classmethod
    def draw_bodypose(cls, keypoints: np.ndarray, H: int, W: int):
        # H, W = (512, 512)
        # H, W = (1084, 1084)
        keypoints = 0.5 * (keypoints + 1)  # [-1, 1] => [0, 1]
        canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
        stickwidth = 4
        for (k1_index, k2_index), color in zip(cls.limbSeq, cls.colors):
            keypoint1 = keypoints[k1_index - 1]
            keypoint2 = keypoints[k2_index - 1]

            if keypoint1.sum() == 0 or keypoint2.sum() == 0:
                continue

            Y = np.array([keypoint1[0], keypoint2[0]]) * W
            X = np.array([keypoint1[1], keypoint2[1]]) * H
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, polygon, [int(float(c) * 0.6) for c in color])

        for keypoint, color in zip(keypoints, cls.colors):
            if keypoint.sum() == 0:
                continue

            x, y = keypoint
            x = int(x * W)
            y = int(y * H)
            cv2.circle(canvas, (int(x), int(y)), 4, color, thickness=-1)

        return transforms.ToTensor()(canvas).to(dtype=torch.float16)
    
    @classmethod
    def draw_bodypose_wo_face(cls, keypoints: np.ndarray):
        H, W = (512, 512)
        keypoints = 0.5 * (keypoints + 1)  # [-1, 1] => [0, 1]
        canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
        stickwidth = 4
        for (k1_index, k2_index), color in zip(cls.limbSeq[:12], cls.colors[:12]):
            keypoint1 = keypoints[k1_index - 1]
            keypoint2 = keypoints[k2_index - 1]

            if keypoint1.sum() == 0 or keypoint2.sum() == 0:
                continue

            Y = np.array([keypoint1[0], keypoint2[0]]) * W
            X = np.array([keypoint1[1], keypoint2[1]]) * H
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, polygon, [int(float(c) * 0.6) for c in color])

        for keypoint, color in zip(keypoints[1:13], cls.colors[1:13]):
            if keypoint.sum() == 0:
                continue

            x, y = keypoint
            x = int(x * W)
            y = int(y * H)
            cv2.circle(canvas, (int(x), int(y)), 4, color, thickness=-1)

        return transforms.ToTensor()(canvas).to(dtype=torch.float16)

class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)
