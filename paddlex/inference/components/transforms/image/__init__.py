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
import math
from pathlib import Path

import numpy as np
import cv2

from .....utils.download import download
from .....utils.cache import CACHE_DIR
from ....utils.io import ImageReader, ImageWriter
from ...base import BaseComponent
from . import funcs as F

__all__ = [
    "ReadImage",
    "Flip",
    "Crop",
    "Resize",
    "ResizeByLong",
    "ResizeByShort",
    "Pad",
    "Normalize",
    "ToCHWImage",
]


def _check_image_size(input_):
    """check image size"""
    if not (
        isinstance(input_, (list, tuple))
        and len(input_) == 2
        and isinstance(input_[0], int)
        and isinstance(input_[1], int)
    ):
        raise TypeError(f"{input_} cannot represent a valid image size.")


class ReadImage(BaseComponent):
    """Load image from the file."""

    INPUT_KEYS = ["img"]
    OUTPUT_KEYS = ["img", "img_size"]
    DEAULT_INPUTS = {"img": "img"}
    DEAULT_OUTPUTS = {"img": "img", "img_path": "img_path", "img_size": "img_size"}

    _FLAGS_DICT = {
        "BGR": cv2.IMREAD_COLOR,
        "RGB": cv2.IMREAD_COLOR,
        "GRAY": cv2.IMREAD_GRAYSCALE,
    }
    SUFFIX = ["jpg", "png", "jpeg", "JPEG", "JPG", "bmp"]

    def __init__(self, batch_size=1, format="BGR"):
        """
        Initialize the instance.

        Args:
            format (str, optional): Target color format to convert the image to.
                Choices are 'BGR', 'RGB', and 'GRAY'. Default: 'BGR'.
        """
        super().__init__()
        self.batch_size = batch_size
        self.format = format
        flags = self._FLAGS_DICT[self.format]
        self._reader = ImageReader(backend="opencv", flags=flags)
        self._writer = ImageWriter(backend="opencv")

    def apply(self, img):
        """apply"""
        if not isinstance(img, str):
            img_path = (Path(CACHE_DIR) / "predict_input" / "tmp_img.jpg").as_posix()
            self._writer.write(img_path, img)
            yield [
                {
                    "img_path": img_path,
                    "img": img,
                    "img_size": [img.shape[1], img.shape[0]],
                }
            ]
        else:
            img_path = img
            # XXX: auto download for url
            img_path = self._download_from_url(img_path)
            image_list = self._get_image_list(img_path)
            batch = []
            for img_path in image_list:
                img = self._read_img(img_path)
                batch.append(img)
                if len(batch) >= self.batch_size:
                    yield batch
                    batch = []
            if len(batch) > 0:
                yield batch

    def _read_img(self, img_path):
        blob = self._reader.read(img_path)
        if blob is None:
            raise Exception("Image read Error")

        if self.format == "RGB":
            if blob.ndim != 3:
                raise RuntimeError("Array is not 3-dimensional.")
            # BGR to RGB
            blob = blob[..., ::-1]
        return {
            "img_path": img_path,
            "img": blob,
            "img_size": [blob.shape[1], blob.shape[0]],
        }

    def _download_from_url(self, in_path):
        if in_path.startswith("http"):
            file_name = Path(in_path).name
            save_path = Path(CACHE_DIR) / "predict_input" / file_name
            download(in_path, save_path, overwrite=True)
            return save_path.as_posix()
        return in_path

    def _get_image_list(self, img_file):
        imgs_lists = []
        if img_file is None or not os.path.exists(img_file):
            raise Exception("not found any img file in {}".format(img_file))

        if os.path.isfile(img_file) and img_file.split(".")[-1] in self.SUFFIX:
            imgs_lists.append(img_file)
        elif os.path.isdir(img_file):
            for root, dirs, files in os.walk(img_file):
                for single_file in files:
                    if single_file.split(".")[-1] in self.SUFFIX:
                        imgs_lists.append(os.path.join(root, single_file))
        if len(imgs_lists) == 0:
            raise Exception("not found any img file in {}".format(img_file))
        imgs_lists = sorted(imgs_lists)
        return imgs_lists


class GetImageInfo(BaseComponent):
    """Get Image Info"""

    INPUT_KEYS = "img"
    OUTPUT_KEYS = "img_size"
    DEAULT_INPUTS = {"img": "img"}
    DEAULT_OUTPUTS = {"img_size": "img_size"}

    def __init__(self):
        super().__init__()

    def apply(self, img):
        """apply"""
        return {"img_size": [img.shape[1], img.shape[0]]}


class Flip(BaseComponent):
    """Flip the image vertically or horizontally."""

    INPUT_KEYS = "img"
    OUTPUT_KEYS = "img"
    DEAULT_INPUTS = {"img": "img"}
    DEAULT_OUTPUTS = {"img": "img"}

    def __init__(self, mode="H"):
        """
        Initialize the instance.

        Args:
            mode (str, optional): 'H' for horizontal flipping and 'V' for vertical
                flipping. Default: 'H'.
        """
        super().__init__()
        if mode not in ("H", "V"):
            raise ValueError("`mode` should be 'H' or 'V'.")
        self.mode = mode

    def apply(self, img):
        """apply"""
        if self.mode == "H":
            img = F.flip_h(img)
        elif self.mode == "V":
            img = F.flip_v(img)
        return {"img": img}


class Crop(BaseComponent):
    """Crop region from the image."""

    INPUT_KEYS = "img"
    OUTPUT_KEYS = ["img", "img_size"]
    DEAULT_INPUTS = {"img": "img"}
    DEAULT_OUTPUTS = {"img": "img", "img_size": "img_size"}

    def __init__(self, crop_size, mode="C"):
        """
        Initialize the instance.

        Args:
            crop_size (list|tuple|int): Width and height of the region to crop.
            mode (str, optional): 'C' for cropping the center part and 'TL' for
                cropping the top left part. Default: 'C'.
        """
        super().__init__()
        if isinstance(crop_size, int):
            crop_size = [crop_size, crop_size]
        _check_image_size(crop_size)

        self.crop_size = crop_size

        if mode not in ("C", "TL"):
            raise ValueError("Unsupported interpolation method")
        self.mode = mode

    def apply(self, img):
        """apply"""
        h, w = img.shape[:2]
        cw, ch = self.crop_size
        if self.mode == "C":
            x1 = max(0, (w - cw) // 2)
            y1 = max(0, (h - ch) // 2)
        elif self.mode == "TL":
            x1, y1 = 0, 0
        x2 = min(w, x1 + cw)
        y2 = min(h, y1 + ch)
        coords = (x1, y1, x2, y2)
        if coords == (0, 0, w, h):
            raise ValueError(
                f"Input image ({w}, {h}) smaller than the target size ({cw}, {ch})."
            )
        img = F.slice(img, coords=coords)
        return {"img": img, "img_size": [img.shape[1], img.shape[0]]}


class _BaseResize(BaseComponent):
    _INTERP_DICT = {
        "NEAREST": cv2.INTER_NEAREST,
        "LINEAR": cv2.INTER_LINEAR,
        "CUBIC": cv2.INTER_CUBIC,
        "AREA": cv2.INTER_AREA,
        "LANCZOS4": cv2.INTER_LANCZOS4,
    }

    def __init__(self, size_divisor, interp):
        super().__init__()

        if size_divisor is not None:
            assert isinstance(
                size_divisor, int
            ), "`size_divisor` should be None or int."
        self.size_divisor = size_divisor

        try:
            interp = self._INTERP_DICT[interp]
        except KeyError:
            raise ValueError(
                "`interp` should be one of {}.".format(self._INTERP_DICT.keys())
            )
        self.interp = interp

    @staticmethod
    def _rescale_size(img_size, target_size):
        """rescale size"""
        scale = min(max(target_size) / max(img_size), min(target_size) / min(img_size))
        rescaled_size = [round(i * scale) for i in img_size]
        return rescaled_size, scale


class Resize(_BaseResize):
    """Resize the image."""

    INPUT_KEYS = "img"
    OUTPUT_KEYS = ["img", "img_size", "scale_factors"]
    DEAULT_INPUTS = {"img": "img"}
    DEAULT_OUTPUTS = {
        "img": "img",
        "img_size": "img_size",
        "scale_factors": "scale_factors",
    }

    def __init__(
        self, target_size, keep_ratio=False, size_divisor=None, interp="LINEAR"
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
        super().__init__(size_divisor=size_divisor, interp=interp)

        if isinstance(target_size, int):
            target_size = [target_size, target_size]
        _check_image_size(target_size)
        self.target_size = target_size

        self.keep_ratio = keep_ratio

    def apply(self, img):
        """apply"""
        target_size = self.target_size
        original_size = img.shape[:2]

        if self.keep_ratio:
            h, w = img.shape[0:2]
            target_size, _ = self._rescale_size((w, h), self.target_size)

        if self.size_divisor:
            target_size = [
                math.ceil(i / self.size_divisor) * self.size_divisor
                for i in target_size
            ]

        img_scale_w, img_scale_h = [
            target_size[1] / original_size[1],
            target_size[0] / original_size[0],
        ]
        img = F.resize(img, target_size, interp=self.interp)
        return {
            "img": img,
            "img_size": [img.shape[1], img.shape[0]],
            "scale_factors": [img_scale_w, img_scale_h],
        }


class ResizeByLong(_BaseResize):
    """
    Proportionally resize the image by specifying the target length of the
    longest side.
    """

    INPUT_KEYS = "img"
    OUTPUT_KEYS = ["img", "img_size"]
    DEAULT_INPUTS = {"img": "img"}
    DEAULT_OUTPUTS = {"img": "img", "img_size": "img_size"}

    def __init__(self, target_long_edge, size_divisor=None, interp="LINEAR"):
        """
        Initialize the instance.

        Args:
            target_long_edge (int): Target length of the longest side of image.
            size_divisor (int|None, optional): Divisor of resized image size.
                Default: None.
            interp (str, optional): Interpolation method. Choices are 'NEAREST',
                'LINEAR', 'CUBIC', 'AREA', and 'LANCZOS4'. Default: 'LINEAR'.
        """
        super().__init__(size_divisor=size_divisor, interp=interp)
        self.target_long_edge = target_long_edge

    def apply(self, img):
        """apply"""
        h, w = img.shape[:2]
        scale = self.target_long_edge / max(h, w)
        h_resize = round(h * scale)
        w_resize = round(w * scale)
        if self.size_divisor is not None:
            h_resize = math.ceil(h_resize / self.size_divisor) * self.size_divisor
            w_resize = math.ceil(w_resize / self.size_divisor) * self.size_divisor

        img = F.resize(img, (w_resize, h_resize), interp=self.interp)
        return {"img": img, "img_size": [img.shape[1], img.shape[0]]}


class ResizeByShort(_BaseResize):
    """
    Proportionally resize the image by specifying the target length of the
    shortest side.
    """

    INPUT_KEYS = "img"
    OUTPUT_KEYS = ["img", "img_size"]
    DEAULT_INPUTS = {"img": "img"}
    DEAULT_OUTPUTS = {"img": "img", "img_size": "img_size"}

    def __init__(self, target_short_edge, size_divisor=None, interp="LINEAR"):
        """
        Initialize the instance.

        Args:
            target_short_edge (int): Target length of the shortest side of image.
            size_divisor (int|None, optional): Divisor of resized image size.
                Default: None.
            interp (str, optional): Interpolation method. Choices are 'NEAREST',
                'LINEAR', 'CUBIC', 'AREA', and 'LANCZOS4'. Default: 'LINEAR'.
        """
        super().__init__(size_divisor=size_divisor, interp=interp)
        self.target_short_edge = target_short_edge

    def apply(self, img):
        """apply"""
        h, w = img.shape[:2]
        scale = self.target_short_edge / min(h, w)
        h_resize = round(h * scale)
        w_resize = round(w * scale)
        if self.size_divisor is not None:
            h_resize = math.ceil(h_resize / self.size_divisor) * self.size_divisor
            w_resize = math.ceil(w_resize / self.size_divisor) * self.size_divisor

        img = F.resize(img, (w_resize, h_resize), interp=self.interp)
        return {"img": img, "img_size": [img.shape[1], img.shape[0]]}


class Pad(BaseComponent):
    """Pad the image."""

    INPUT_KEYS = "img"
    OUTPUT_KEYS = ["img", "img_size"]
    DEAULT_INPUTS = {"img": "img"}
    DEAULT_OUTPUTS = {"img": "img", "img_size": "img_size"}

    def __init__(self, target_size, val=127.5):
        """
        Initialize the instance.

        Args:
            target_size (list|tuple|int): Target width and height of the image after
                padding.
            val (float, optional): Value to fill the padded area. Default: 127.5.
        """
        super().__init__()

        if isinstance(target_size, int):
            target_size = [target_size, target_size]
        _check_image_size(target_size)
        self.target_size = target_size

        self.val = val

    def apply(self, img):
        """apply"""
        h, w = img.shape[:2]
        tw, th = self.target_size
        ph = th - h
        pw = tw - w

        if ph < 0 or pw < 0:
            raise ValueError(
                f"Input image ({w}, {h}) smaller than the target size ({tw}, {th})."
            )
        else:
            img = F.pad(img, pad=(0, ph, 0, pw), val=self.val)
        return {"img": img, "img_size": [img.shape[1], img.shape[0]]}


class Normalize(BaseComponent):
    """Normalize the image."""

    INPUT_KEYS = "img"
    OUTPUT_KEYS = "img"
    DEAULT_INPUTS = {"img": "img"}
    DEAULT_OUTPUTS = {"img": "img"}

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

    def apply(self, img):
        """apply"""
        old_type = img.dtype
        # XXX: If `old_type` has higher precision than float32,
        # we will lose some precision.
        img = img.astype("float32", copy=False)
        img *= self.scale
        img -= self.mean
        img /= self.std
        if self.preserve_dtype:
            img = img.astype(old_type, copy=False)
        return {"img": img}


class ToCHWImage(BaseComponent):
    """Reorder the dimensions of the image from HWC to CHW."""

    INPUT_KEYS = "img"
    OUTPUT_KEYS = "img"
    DEAULT_INPUTS = {"img": "img"}
    DEAULT_OUTPUTS = {"img": "img"}

    def apply(self, img):
        """apply"""
        img = img.transpose((2, 0, 1))
        return {"img": img}
