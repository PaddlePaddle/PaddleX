# copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.
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

import os
import ast
import math
from pathlib import Path
from copy import deepcopy

import numpy as np
import cv2
from PIL import Image

from . import funcs as F
from ....utils.benchmark import benchmark


class _BaseResize:
    _CV2_INTERP_DICT = {
        "NEAREST": cv2.INTER_NEAREST,
        "LINEAR": cv2.INTER_LINEAR,
        "BICUBIC": cv2.INTER_CUBIC,
        "AREA": cv2.INTER_AREA,
        "LANCZOS4": cv2.INTER_LANCZOS4,
    }
    _PIL_INTERP_DICT = {
        "NEAREST": Image.NEAREST,
        "BILINEAR": Image.BILINEAR,
        "BICUBIC": Image.BICUBIC,
        "BOX": Image.BOX,
        "LANCZOS4": Image.LANCZOS,
    }

    def __init__(self, size_divisor, interp, backend="cv2"):
        super().__init__()

        if size_divisor is not None:
            assert isinstance(
                size_divisor, int
            ), "`size_divisor` should be None or int."
        self.size_divisor = size_divisor

        try:
            interp = interp.upper()
            if backend == "cv2":
                interp = self._CV2_INTERP_DICT[interp]
            elif backend == "pil":
                interp = self._PIL_INTERP_DICT[interp]
            else:
                raise ValueError("backend must be `cv2` or `pil`")
        except KeyError:
            raise ValueError(
                "For backend '{}', `interp` should be one of {}. Please ensure the interpolation method matches the selected backend.".format(
                    backend,
                    (
                        self._CV2_INTERP_DICT.keys()
                        if backend == "cv2"
                        else self._PIL_INTERP_DICT.keys()
                    ),
                )
            )
        self.interp = interp
        self.backend = backend

    @staticmethod
    def _rescale_size(img_size, target_size):
        """rescale size"""
        scale = min(max(target_size) / max(img_size), min(target_size) / min(img_size))
        rescaled_size = [round(i * scale) for i in img_size]
        return rescaled_size, scale


@benchmark.timeit
class Resize(_BaseResize):
    """Resize the image."""

    def __init__(
        self,
        target_size,
        keep_ratio=False,
        size_divisor=None,
        interp="LINEAR",
        backend="cv2",
    ):
        """
        Initialize the instance.

        Args:
            target_size (list|tuple|int): Target width and height.
            keep_ratio (bool, optional): Whether to keep the aspect ratio of resized
                image. Default: False.
            size_divisor (int|None, optional): Divisor of resized image size.
                Default: None.
            interp (str, optional): Interpolation method. Choices are 'NEAREST',
                'LINEAR', 'CUBIC', 'AREA', and 'LANCZOS4'. Default: 'LINEAR'.
        """
        super().__init__(size_divisor=size_divisor, interp=interp, backend=backend)

        if isinstance(target_size, int):
            target_size = [target_size, target_size]
        F.check_image_size(target_size)
        self.target_size = target_size

        self.keep_ratio = keep_ratio

    def __call__(self, imgs):
        """apply"""
        return [self.resize(img) for img in imgs]

    def resize(self, img):
        target_size = self.target_size
        original_size = img.shape[:2][::-1]

        if self.keep_ratio:
            h, w = img.shape[0:2]
            target_size, _ = self._rescale_size((w, h), self.target_size)

        if self.size_divisor:
            target_size = [
                math.ceil(i / self.size_divisor) * self.size_divisor
                for i in target_size
            ]
        img = F.resize(img, target_size, interp=self.interp, backend=self.backend)
        return img


@benchmark.timeit
class ResizeByLong(_BaseResize):
    """
    Proportionally resize the image by specifying the target length of the
    longest side.
    """

    def __init__(
        self, target_long_edge, size_divisor=None, interp="LINEAR", backend="cv2"
    ):
        """
        Initialize the instance.

        Args:
            target_long_edge (int): Target length of the longest side of image.
            size_divisor (int|None, optional): Divisor of resized image size.
                Default: None.
            interp (str, optional): Interpolation method. Choices are 'NEAREST',
                'LINEAR', 'CUBIC', 'AREA', and 'LANCZOS4'. Default: 'LINEAR'.
        """
        super().__init__(size_divisor=size_divisor, interp=interp, backend=backend)
        self.target_long_edge = target_long_edge

    def __call__(self, imgs):
        """apply"""
        return [self.resize(img) for img in imgs]

    def resize(self, img):
        h, w = img.shape[:2]
        scale = self.target_long_edge / max(h, w)
        h_resize = round(h * scale)
        w_resize = round(w * scale)
        if self.size_divisor is not None:
            h_resize = math.ceil(h_resize / self.size_divisor) * self.size_divisor
            w_resize = math.ceil(w_resize / self.size_divisor) * self.size_divisor

        img = F.resize(
            img, (w_resize, h_resize), interp=self.interp, backend=self.backend
        )
        return img


@benchmark.timeit
class ResizeByShort(_BaseResize):
    """
    Proportionally resize the image by specifying the target length of the
    shortest side.
    """

    def __init__(
        self, target_short_edge, size_divisor=None, interp="LINEAR", backend="cv2"
    ):
        """
        Initialize the instance.

        Args:
            target_short_edge (int): Target length of the shortest side of image.
            size_divisor (int|None, optional): Divisor of resized image size.
                Default: None.
            interp (str, optional): Interpolation method. Choices are 'NEAREST',
                'LINEAR', 'CUBIC', 'AREA', and 'LANCZOS4'. Default: 'LINEAR'.
        """
        super().__init__(size_divisor=size_divisor, interp=interp, backend=backend)
        self.target_short_edge = target_short_edge

    def __call__(self, imgs):
        """apply"""
        return [self.resize(img) for img in imgs]

    def resize(self, img):
        h, w = img.shape[:2]
        scale = self.target_short_edge / min(h, w)
        h_resize = round(h * scale)
        w_resize = round(w * scale)
        if self.size_divisor is not None:
            h_resize = math.ceil(h_resize / self.size_divisor) * self.size_divisor
            w_resize = math.ceil(w_resize / self.size_divisor) * self.size_divisor

        img = F.resize(
            img, (w_resize, h_resize), interp=self.interp, backend=self.backend
        )
        return img


@benchmark.timeit
class Normalize:
    """Normalize the image."""

    def __init__(self, scale=1.0 / 255, mean=0.5, std=0.5, preserve_dtype=False):
        """
        Initialize the instance.

        Args:
            scale (float, optional): Scaling factor to apply to the image before
                applying normalization. Default: 1/255.
            mean (float|tuple|list, optional): Means for each channel of the image.
                Default: 0.5.
            std (float|tuple|list, optional): Standard deviations for each channel
                of the image. Default: 0.5.
            preserve_dtype (bool, optional): Whether to preserve the original dtype
                of the image.
        """
        super().__init__()

        self.scale = np.float32(scale)
        if isinstance(mean, float):
            mean = [mean]
        self.mean = np.asarray(mean).astype("float32")
        if isinstance(std, float):
            std = [std]
        self.std = np.asarray(std).astype("float32")
        self.preserve_dtype = preserve_dtype

    def __call__(self, imgs):
        """apply"""
        old_type = imgs[0].dtype
        # XXX: If `old_type` has higher precision than float32,
        # we will lose some precision.
        imgs = np.array(imgs).astype("float32", copy=False)
        imgs *= self.scale
        imgs -= self.mean
        imgs /= self.std
        if self.preserve_dtype:
            imgs = imgs.astype(old_type, copy=False)
        return list(imgs)


@benchmark.timeit
class ToCHWImage:
    """Reorder the dimensions of the image from HWC to CHW."""

    def __call__(self, imgs):
        """apply"""
        return [img.transpose((2, 0, 1)) for img in imgs]


@benchmark.timeit
class ToBatch:
    def __call__(self, imgs):
        return [np.stack(imgs, axis=0).astype(dtype=np.float32, copy=False)]
