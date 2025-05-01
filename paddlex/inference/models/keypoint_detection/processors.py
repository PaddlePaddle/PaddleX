# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy import ndarray

from ....utils.deps import class_requires_deps, is_dep_available
from ...utils.benchmark import benchmark
from ..object_detection.processors import get_affine_transform

if is_dep_available("opencv-contrib-python"):
    import cv2

Number = Union[int, float]
Kpts = List[dict]


def get_warp_matrix(
    theta: float, size_input: ndarray, size_dst: ndarray, size_target: ndarray
) -> ndarray:
    """This code is based on
        https://github.com/open-mmlab/mmpose/blob/master/mmpose/core/post_processing/post_transforms.py
        Calculate the transformation matrix under the constraint of unbiased.
    Paper ref: Huang et al. The Devil is in the Details: Delving into Unbiased
    Data Processing for Human Pose Estimation (CVPR 2020).
    Args:
        theta (float): Rotation angle in degrees.
        size_input (np.ndarray): Size of input image [w, h].
        size_dst (np.ndarray): Size of output image [w, h].
        size_target (np.ndarray): Size of ROI in input plane [w, h].
    Returns:
        matrix (np.ndarray): A matrix for transformation.
    """
    theta = np.deg2rad(theta)
    matrix = np.zeros((2, 3), dtype=np.float32)

    scale_x = size_dst[0] / size_target[0]
    scale_y = size_dst[1] / size_target[1]

    matrix[0, 0] = np.cos(theta) * scale_x
    matrix[0, 1] = -np.sin(theta) * scale_x
    matrix[0, 2] = scale_x * (
        -0.5 * size_input[0] * np.cos(theta)
        + 0.5 * size_input[1] * np.sin(theta)
        + 0.5 * size_target[0]
    )
    matrix[1, 0] = np.sin(theta) * scale_y
    matrix[1, 1] = np.cos(theta) * scale_y
    matrix[1, 2] = scale_y * (
        -0.5 * size_input[0] * np.sin(theta)
        - 0.5 * size_input[1] * np.cos(theta)
        + 0.5 * size_target[1]
    )

    return matrix


@benchmark.timeit
@class_requires_deps("opencv-contrib-python")
class TopDownAffine:
    """refer to https://github.com/open-mmlab/mmpose/blob/71ec36ebd63c475ab589afc817868e749a61491f/mmpose/datasets/transforms/topdown_transforms.py#L13
    Get the bbox image as the model input by affine transform.

    Args:
        input_size (Tuple[int, int]): The input image size of the model in
            [w, h]. The bbox region will be cropped and resize to `input_size`
        use_udp (bool): Whether use unbiased data processing. See
            `UDP (CVPR 2020)`_ for details. Defaults to ``False``

    .. _`UDP (CVPR 2020)`: https://arxiv.org/abs/1911.07524
    """

    def __init__(self, input_size: Tuple[int, int], use_udp: bool = False):
        assert (
            all([isinstance(i, int) for i in input_size]) and len(input_size) == 2
        ), f"Invalid input_size {input_size}"
        self.input_size = input_size
        self.use_udp = use_udp

    def apply(
        self,
        img: ndarray,
        center: Optional[Union[Tuple[Number, Number], ndarray]] = None,
        scale: Optional[Union[Tuple[Number, Number], ndarray]] = None,
    ) -> Tuple[ndarray, ndarray, ndarray]:
        """Applies a wrapaffine to the input image based on the specified center, scale.

        Args:
            img (ndarray): The input image as a NumPy ndarray.
            center (Optional[Union[Tuple[Number, Number], ndarray]], optional): Center of the bounding box (x, y)
            scale (Optional[Union[Tuple[Number, Number], ndarray]], optional): Scale of the bounding box
            wrt [width, height].

        Returns:
            Tuple[ndarray, ndarray, ndarray]: The transformed image,
            the center used for the transformation, and the scale used for the transformation.
        """
        rot = 0
        imshape = np.array(img.shape[:2][::-1])
        if isinstance(center, Sequence):
            center = np.array(center)
        if isinstance(scale, Sequence):
            scale = np.array(scale)

        center = center if center is not None else imshape / 2.0
        scale = scale if scale is not None else imshape
        if self.use_udp:
            trans = get_warp_matrix(
                rot,
                center * 2.0,
                [self.input_size[0] - 1.0, self.input_size[1] - 1.0],
                scale,
            )
            img = cv2.warpAffine(
                img,
                trans,
                (int(self.input_size[0]), int(self.input_size[1])),
                flags=cv2.INTER_LINEAR,
            )
        else:
            trans = get_affine_transform(center, scale, rot, self.input_size)
            img = cv2.warpAffine(
                img,
                trans,
                (int(self.input_size[0]), int(self.input_size[1])),
                flags=cv2.INTER_LINEAR,
            )

        return img, center, scale

    def __call__(self, datas: List[dict]) -> List[dict]:
        for data in datas:
            ori_img = data["img"]
            if "ori_img" not in data:
                data["ori_img"] = ori_img
            if "ori_img_size" not in data:
                data["ori_img_size"] = [ori_img.shape[1], ori_img.shape[0]]

            img, center, scale = self.apply(
                ori_img, data.get("center", None), data.get("scale", None)
            )
            data["img"] = img
            data["center"] = center
            data["scale"] = scale

            img_size = [img.shape[1], img.shape[0]]
            data["img_size"] = img_size  # [size_w, size_h]

        return datas


def affine_transform(pt: ndarray, t: ndarray):
    """Apply an affine transformation to a 2D point.

    Args:
        pt (numpy.ndarray): A 2D point represented as a 2-element array.
        t (numpy.ndarray): A 3x3 affine transformation matrix.

    Returns:
        numpy.ndarray: The transformed 2D point.
    """
    new_pt = np.array([pt[0], pt[1], 1.0]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def transform_preds(
    coords: ndarray,
    center: Tuple[float, float],
    scale: Tuple[float, float],
    output_size: Tuple[int, int],
) -> ndarray:
    """Transform coordinates to the target space using an affine transformation.

    Args:
        coords (numpy.ndarray): Original coordinates, shape (N, 2).
        center (tuple): Center point for the transformation.
        scale (tuple): Scale factor for the transformation.
        output_size (tuple): Size of the output space.

    Returns:
        numpy.ndarray: Transformed coordinates, shape (N, 2).
    """
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


@benchmark.timeit
@class_requires_deps("opencv-contrib-python")
class KptPostProcess:
    """Save Result Transform"""

    def __init__(self, use_dark=True):
        self.use_dark = use_dark

    def apply(self, heatmap: ndarray, center: ndarray, scale: ndarray) -> Kpts:
        """apply"""
        # TODO: add batch support
        heatmap, center, scale = heatmap[None, ...], center[None, ...], scale[None, ...]
        preds, maxvals = self.get_final_preds(heatmap, center, scale)
        keypoints, scores = np.concatenate((preds, maxvals), axis=-1), np.mean(
            maxvals.squeeze(-1), axis=1
        )

        return [
            {"keypoints": kpt, "kpt_score": score}
            for kpt, score in zip(keypoints, scores)
        ]

    def __call__(self, batch_outputs: List[dict], datas: List[dict]) -> List[Kpts]:
        """Apply the post-processing to a batch of outputs.

        Args:
            batch_outputs (List[dict]): The list of detection outputs.
            datas (List[dict]): The list of input data.

        Returns:
            List[dict]: The list of post-processed keypoints.
        """
        return [
            self.apply(output["heatmap"], data["center"], data["scale"])
            for data, output in zip(datas, batch_outputs)
        ]

    def get_final_preds(
        self, heatmaps: ndarray, center: ndarray, scale: ndarray, kernelsize: int = 3
    ):
        """the highest heatvalue location with a quarter offset in the
        direction from the highest response to the second highest response.
        Args:
            heatmaps (numpy.ndarray): The predicted heatmaps
            center (numpy.ndarray): The boxes center
            scale (numpy.ndarray): The scale factor
        Returns:
            preds: numpy.ndarray([batch_size, num_joints, 2]), keypoints coords
            maxvals: numpy.ndarray([batch_size, num_joints, 1]), the maximum confidence of the keypoints
        """
        coords, maxvals = self.get_max_preds(heatmaps)
        heatmap_height = heatmaps.shape[2]
        heatmap_width = heatmaps.shape[3]

        if self.use_dark:
            coords = self.dark_postprocess(heatmaps, coords, kernelsize)
        else:
            for n in range(coords.shape[0]):
                for p in range(coords.shape[1]):
                    hm = heatmaps[n][p]
                    px = int(math.floor(coords[n][p][0] + 0.5))
                    py = int(math.floor(coords[n][p][1] + 0.5))
                    if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
                        diff = np.array(
                            [
                                hm[py][px + 1] - hm[py][px - 1],
                                hm[py + 1][px] - hm[py - 1][px],
                            ]
                        )
                        coords[n][p] += np.sign(diff) * 0.25
        preds = coords.copy()
        # Transform back
        for i in range(coords.shape[0]):
            preds[i] = transform_preds(
                coords[i], center[i], scale[i], [heatmap_width, heatmap_height]
            )

        return preds, maxvals

    def get_max_preds(self, heatmaps: ndarray) -> Tuple[ndarray, ndarray]:
        """get predictions from score maps
        Args:
            heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
        Returns:
            preds: numpy.ndarray([batch_size, num_joints, 2]), keypoints coords
            maxvals: numpy.ndarray([batch_size, num_joints, 2]), the maximum confidence of the keypoints
        """
        assert isinstance(heatmaps, np.ndarray), "heatmaps should be numpy.ndarray"
        assert heatmaps.ndim == 4, "batch_images should be 4-ndim"
        batch_size = heatmaps.shape[0]
        num_joints = heatmaps.shape[1]
        width = heatmaps.shape[3]
        heatmaps_reshaped = heatmaps.reshape((batch_size, num_joints, -1))
        idx = np.argmax(heatmaps_reshaped, 2)
        maxvals = np.amax(heatmaps_reshaped, 2)
        maxvals = maxvals.reshape((batch_size, num_joints, 1))
        idx = idx.reshape((batch_size, num_joints, 1))
        preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
        preds[:, :, 0] = (preds[:, :, 0]) % width
        preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)
        pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
        pred_mask = pred_mask.astype(np.float32)
        preds *= pred_mask

        return preds, maxvals

    def gaussian_blur(self, heatmap: ndarray, kernel: int) -> ndarray:
        border = (kernel - 1) // 2
        batch_size = heatmap.shape[0]
        num_joints = heatmap.shape[1]
        height = heatmap.shape[2]
        width = heatmap.shape[3]
        for i in range(batch_size):
            for j in range(num_joints):
                origin_max = np.max(heatmap[i, j])
                dr = np.zeros((height + 2 * border, width + 2 * border))
                dr[border:-border, border:-border] = heatmap[i, j].copy()
                dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
                heatmap[i, j] = dr[border:-border, border:-border].copy()
                heatmap[i, j] *= origin_max / np.max(heatmap[i, j])

        return heatmap

    def dark_parse(self, hm: ndarray, coord: ndarray):
        heatmap_height = hm.shape[0]
        heatmap_width = hm.shape[1]
        px = int(coord[0])
        py = int(coord[1])
        if 1 < px < heatmap_width - 2 and 1 < py < heatmap_height - 2:
            dx = 0.5 * (hm[py][px + 1] - hm[py][px - 1])
            dy = 0.5 * (hm[py + 1][px] - hm[py - 1][px])
            dxx = 0.25 * (hm[py][px + 2] - 2 * hm[py][px] + hm[py][px - 2])
            dxy = 0.25 * (
                hm[py + 1][px + 1]
                - hm[py - 1][px + 1]
                - hm[py + 1][px - 1]
                + hm[py - 1][px - 1]
            )
            dyy = 0.25 * (hm[py + 2 * 1][px] - 2 * hm[py][px] + hm[py - 2 * 1][px])
            derivative = np.matrix([[dx], [dy]])
            hessian = np.matrix([[dxx, dxy], [dxy, dyy]])
            if dxx * dyy - dxy**2 != 0:
                hessianinv = hessian.I
                offset = -hessianinv * derivative
                offset = np.squeeze(np.array(offset.T), axis=0)
                coord += offset

        return coord

    def dark_postprocess(
        self, hm: ndarray, coords: ndarray, kernelsize: int
    ) -> ndarray:
        """
        refer to https://github.com/ilovepose/DarkPose/lib/core/inference.py
        """
        hm = self.gaussian_blur(hm, kernelsize)
        hm = np.maximum(hm, 1e-10)
        hm = np.log(hm)
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                coords[n, p] = self.dark_parse(hm[n][p], coords[n][p])

        return coords
