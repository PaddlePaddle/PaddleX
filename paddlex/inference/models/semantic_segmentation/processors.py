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

import numpy as np

from ....utils.deps import class_requires_deps, is_dep_available
from ...utils.benchmark import benchmark
from ..common.vision import funcs as F
from ..common.vision.processors import _BaseResize

if is_dep_available("opencv-contrib-python"):
    import cv2


@benchmark.timeit
class Resize(_BaseResize):
    """Resize the image."""

    def __init__(
        self, target_size=-1, keep_ratio=False, size_divisor=None, interp="LINEAR"
    ):
        """
        Initialize the instance.

        Args:
            target_size (list|tuple|int, optional): Target width and height. -1 will return the images directly without resizing.
            keep_ratio (bool, optional): Whether to keep the aspect ratio of resized
                image. Default: False.
            size_divisor (int|None, optional): Divisor of resized image size.
                Default: None.
            interp (str, optional): Interpolation method. Choices are 'NEAREST',
                'LINEAR', 'CUBIC', 'AREA', and 'LANCZOS4'. Default: 'LINEAR'.
        """
        super().__init__(size_divisor=size_divisor, interp=interp)

        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        F.check_image_size(target_size)
        self.target_size = target_size

        self.keep_ratio = keep_ratio

    def __call__(self, imgs, target_size=None):
        """apply"""
        target_size = self.target_size if target_size is None else target_size
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        F.check_image_size(target_size)
        return [self.resize(img, target_size) for img in imgs]

    def resize(self, img, target_size):

        if target_size == (-1, -1):
            # If the final target_size == (-1, -1), it means use the source input image directly.
            return img
        original_size = img.shape[:2][::-1]
        assert target_size[0] > 0 and target_size[1] > 0

        if self.keep_ratio:
            h, w = img.shape[0:2]
            target_size, _ = self._rescale_size((w, h), target_size)

        if self.size_divisor:
            target_size = [
                math.ceil(i / self.size_divisor) * self.size_divisor
                for i in target_size
            ]
        img = F.resize(img, target_size, interp=self.interp)
        return img


@benchmark.timeit
@class_requires_deps("opencv-contrib-python")
class SegPostProcess:
    """Semantic Segmentation PostProcess

    This class is responsible for post-processing detection results, only including
    restoring the prediction segmentation map to the original image size for now.
    """

    def __call__(self, imgs, src_images):
        assert len(imgs) == len(src_images)

        src_sizes = [src_image.shape[:2][::-1] for src_image in src_images]
        return [
            self.reverse_resize(img, src_size) for img, src_size in zip(imgs, src_sizes)
        ]

    def reverse_resize(self, img, src_size):
        """Restore the prediction map to source image size using nearest interpolation.

        Args:
             img (np.ndarray): prediction map with shape of (1, width, height)
             src_size (Tuple[int, int]): source size of the input image, with format of (width, height).
        """
        assert isinstance(src_size, (tuple, list)) and len(src_size) == 2
        assert src_size[0] > 0 and src_size[1] > 0
        assert img.ndim == 3

        reversed_img = cv2.resize(
            img[0], dsize=src_size, interpolation=cv2.INTER_NEAREST
        )
        reversed_img = np.expand_dims(reversed_img, axis=0)
        return reversed_img
