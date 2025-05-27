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
import cv2

from ..base import PyOnlyProcessor

__all__ = [
    "GetImageInfo",
    "Flip",
    "Crop",
    "Resize",
    "ResizeByLong",
    "ResizeByShort",
    "Pad",
    "PadStride",
    "Normalize",
    "ToCHWImage",
    "LaTeXOCRReisizeNormImg",
]


def _resize(im, target_size, interp):
    w, h = target_size
    im = cv2.resize(im, (w, h), interpolation=interp)
    return im


def _flip_h(im):
    if len(im.shape) == 3:
        im = im[:, ::-1, :]
    elif len(im.shape) == 2:
        im = im[:, ::-1]
    return im


def _flip_v(im):
    if len(im.shape) == 3:
        im = im[::-1, :, :]
    elif len(im.shape) == 2:
        im = im[::-1, :]
    return im


def _slice(im, coords):
    x1, y1, x2, y2 = coords
    im = im[y1:y2, x1:x2, ...]
    return im


def _pad(im, pad, val):
    if isinstance(pad, int):
        pad = [pad] * 4
    if len(pad) != 4:
        raise ValueError
    chns = 1 if im.ndim == 2 else im.shape[2]
    im = cv2.copyMakeBorder(im, *pad, cv2.BORDER_CONSTANT, value=(val,) * chns)
    return im


def _check_image_size(input_):
    if not (
        isinstance(input_, (list, tuple))
        and len(input_) == 2
        and isinstance(input_[0], int)
        and isinstance(input_[1], int)
    ):
        raise TypeError(f"{input_} cannot represent a valid image size.")


class GetImageInfo(PyOnlyProcessor):
    def __call__(self, data):
        img = data["img"]

        return {**data, "img_size": [img.shape[1], img.shape[0]]}


class Flip(PyOnlyProcessor):
    def __init__(self, mode="H"):
        super().__init__()
        if mode not in ("H", "V"):
            raise ValueError("`mode` should be 'H' or 'V'.")
        self._mode = mode

    def __call__(self, data):
        img = data["img"]

        if self._mode == "H":
            img = _flip_h(img)
        elif self._mode == "V":
            img = _flip_v(img)

        return {**data, "img": img}


class Crop(PyOnlyProcessor):
    def __init__(self, crop_size, mode="C"):
        super().__init__()
        if isinstance(crop_size, int):
            crop_size = [crop_size, crop_size]
        _check_image_size(crop_size)

        self._crop_size = crop_size

        if mode not in ("C", "TL"):
            raise ValueError("Unsupported interpolation method")
        self._mode = mode

    def __call__(self, data):
        img = data["img"]

        h, w = img.shape[:2]
        cw, ch = self._crop_size
        if self._mode == "C":
            x1 = max(0, (w - cw) // 2)
            y1 = max(0, (h - ch) // 2)
        elif self._mode == "TL":
            x1, y1 = 0, 0
        x2 = min(w, x1 + cw)
        y2 = min(h, y1 + ch)
        coords = (x1, y1, x2, y2)
        if coords == (0, 0, w, h):
            raise ValueError(
                f"Input image ({w}, {h}) smaller than the target size ({cw}, {ch})."
            )
        img = _slice(img, coords=coords)

        return {**data, "img": img, "img_size": [img.shape[1], img.shape[0]]}


class _BaseResize(PyOnlyProcessor):
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
        self._size_divisor = size_divisor

        try:
            interp = self._INTERP_DICT[interp]
        except KeyError:
            raise ValueError(
                "`interp` should be one of {}.".format(self._INTERP_DICT.keys())
            )
        self._interp = interp

    @staticmethod
    def _rescale_size(img_size, target_size):
        scale = min(max(target_size) / max(img_size), min(target_size) / min(img_size))
        rescaled_size = [round(i * scale) for i in img_size]
        return rescaled_size, scale


class Resize(_BaseResize):
    def __init__(
        self, target_size, keep_ratio=False, size_divisor=None, interp="LINEAR"
    ):
        super().__init__(size_divisor=size_divisor, interp=interp)

        if isinstance(target_size, int):
            target_size = [target_size, target_size]
        _check_image_size(target_size)
        self._target_size = target_size

        self._keep_ratio = keep_ratio

    def __call__(self, data):
        img = data["img"]

        target_size = self._target_size
        original_size = img.shape[:2][::-1]

        if self._keep_ratio:
            h, w = img.shape[0:2]
            target_size, _ = self._rescale_size((w, h), self._target_size)

        if self._size_divisor:
            target_size = [
                math.ceil(i / self._size_divisor) * self._size_divisor
                for i in target_size
            ]

        img_scale_w, img_scale_h = [
            target_size[0] / original_size[0],
            target_size[1] / original_size[1],
        ]
        img = _resize(img, target_size, interp=self._interp)

        return {
            **data,
            "img": img,
            "img_size": [img.shape[1], img.shape[0]],
            "scale_factors": [img_scale_w, img_scale_h],
        }


class ResizeByLong(_BaseResize):
    def __init__(self, target_long_edge, size_divisor=None, interp="LINEAR"):
        super().__init__(size_divisor=size_divisor, interp=interp)
        self._target_long_edge = target_long_edge

    def __call__(self, data):
        img = data["img"]

        h, w = img.shape[:2]
        scale = self._target_long_edge / max(h, w)
        h_resize = round(h * scale)
        w_resize = round(w * scale)
        if self._size_divisor is not None:
            h_resize = math.ceil(h_resize / self._size_divisor) * self._size_divisor
            w_resize = math.ceil(w_resize / self._size_divisor) * self._size_divisor

        img = _resize(img, (w_resize, h_resize), interp=self._interp)

        return {**data, "img": img, "img_size": [img.shape[1], img.shape[0]]}


class ResizeByShort(_BaseResize):
    INPUT_KEYS = "img"
    OUTPUT_KEYS = ["img", "img_size"]
    DEAULT_INPUTS = {"img": "img"}
    DEAULT_OUTPUTS = {"img": "img", "img_size": "img_size"}

    def __init__(self, target_short_edge, size_divisor=None, interp="LINEAR"):
        super().__init__(size_divisor=size_divisor, interp=interp)
        self._target_short_edge = target_short_edge

    def __call__(self, data):
        img = data["img"]

        h, w = img.shape[:2]
        scale = self._target_short_edge / min(h, w)
        h_resize = round(h * scale)
        w_resize = round(w * scale)
        if self._size_divisor is not None:
            h_resize = math.ceil(h_resize / self._size_divisor) * self._size_divisor
            w_resize = math.ceil(w_resize / self._size_divisor) * self._size_divisor

        img = _resize(img, (w_resize, h_resize), interp=self._interp)

        return {**data, "img": img, "img_size": [img.shape[1], img.shape[0]]}


class Pad(PyOnlyProcessor):
    def __init__(self, target_size, val=127.5):
        super().__init__()

        if isinstance(target_size, int):
            target_size = [target_size, target_size]
        _check_image_size(target_size)
        self._target_size = target_size

        self._val = val

    def __call__(self, data):
        img = data["img"]

        h, w = img.shape[:2]
        tw, th = self._target_size
        ph = th - h
        pw = tw - w

        if ph < 0 or pw < 0:
            raise ValueError(
                f"Input image ({w}, {h}) smaller than the target size ({tw}, {th})."
            )
        else:
            img = _pad(img, pad=(0, ph, 0, pw), val=self._val)

        return {**data, "img": img, "img_size": [img.shape[1], img.shape[0]]}


class PadStride(PyOnlyProcessor):
    INPUT_KEYS = "img"
    OUTPUT_KEYS = "img"
    DEAULT_INPUTS = {"img": "img"}
    DEAULT_OUTPUTS = {"img": "img"}

    def __init__(self, stride=0):
        super().__init__()
        self._coarsest_stride = stride

    def __call__(self, data):
        img = data["img"]

        im = img
        coarsest_stride = self._coarsest_stride
        if coarsest_stride <= 0:
            return {"img": im}
        im_c, im_h, im_w = im.shape
        pad_h = int(np.ceil(float(im_h) / coarsest_stride) * coarsest_stride)
        pad_w = int(np.ceil(float(im_w) / coarsest_stride) * coarsest_stride)
        padding_im = np.zeros((im_c, pad_h, pad_w), dtype=np.float32)
        padding_im[:, :im_h, :im_w] = im

        return {**data, "img": padding_im}


class Normalize(PyOnlyProcessor):
    def __init__(self, scale=1.0 / 255, mean=0.5, std=0.5, preserve_dtype=False):
        super().__init__()
        self._scale = np.float32(scale)
        if isinstance(mean, float):
            mean = [mean]
        self._mean = np.asarray(mean).astype("float32")
        if isinstance(std, float):
            std = [std]
        self._std = np.asarray(std).astype("float32")
        self._preserve_dtype = preserve_dtype

    def __call__(self, data):
        img = data["img"]

        old_type = img.dtype
        # XXX: If `old_type` has higher precision than float32,
        # we will lose some precision.
        img = img.astype("float32", copy=False)
        img *= self._scale
        img -= self._mean
        img /= self._std
        if self._preserve_dtype:
            img = img.astype(old_type, copy=False)

        return {**data, "img": img}


class ToCHWImage(PyOnlyProcessor):
    def __call__(self, data):
        img = data["img"]

        img = img.transpose((2, 0, 1))

        return {**data, "img": img}


class BGR2RGB(PyOnlyProcessor):
    def __call__(self, data):
        img = data["img"]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return {**data, "img": img}


class LaTeXOCRReisizeNormImg(PyOnlyProcessor):
    """for ocr image resize and normalization"""

    def __init__(self, rec_image_shape=(3, 48, 320)):
        super().__init__()
        self.rec_image_shape = rec_image_shape

    def pad_(self, img, divable=32):
        from PIL import Image

        threshold = 128
        data = np.array(img.convert("LA"))
        if data[..., -1].var() == 0:
            data = (data[..., 0]).astype(np.uint8)
        else:
            data = (255 - data[..., -1]).astype(np.uint8)
        data = (data - data.min()) / (data.max() - data.min()) * 255
        if data.mean() > threshold:
            # To invert the text to white
            gray = 255 * (data < threshold).astype(np.uint8)
        else:
            gray = 255 * (data > threshold).astype(np.uint8)
            data = 255 - data

        coords = cv2.findNonZero(gray)  # Find all non-zero points (text)
        a, b, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
        rect = data[b : b + h, a : a + w]
        im = Image.fromarray(rect).convert("L")
        dims = []
        for x in [w, h]:
            div, mod = divmod(x, divable)
            dims.append(divable * (div + (1 if mod > 0 else 0)))
        padded = Image.new("L", dims, 255)
        padded.paste(im, (0, 0, im.size[0], im.size[1]))
        return padded

    def minmax_size_(
        self,
        img,
        max_dimensions,
        min_dimensions,
    ):
        from PIL import Image

        if max_dimensions is not None:
            ratios = [a / b for a, b in zip(img.size, max_dimensions)]
            if any([r > 1 for r in ratios]):
                size = np.array(img.size) // max(ratios)
                img = img.resize(tuple(size.astype(int)), Image.BILINEAR)
        if min_dimensions is not None:
            # hypothesis: there is a dim in img smaller than min_dimensions, and return a proper dim >= min_dimensions
            padded_size = [
                max(img_dim, min_dim)
                for img_dim, min_dim in zip(img.size, min_dimensions)
            ]
            if padded_size != list(img.size):  # assert hypothesis
                padded_im = Image.new("L", padded_size, 255)
                padded_im.paste(img, img.getbbox())
                img = padded_im
        return img

    def norm_img_latexocr(self, img):
        # CAN only predict gray scale image
        from PIL import Image

        shape = (1, 1, 3)
        mean = [0.7931, 0.7931, 0.7931]
        std = [0.1738, 0.1738, 0.1738]
        scale = np.float32(1.0 / 255.0)
        min_dimensions = [32, 32]
        max_dimensions = [672, 192]
        mean = np.array(mean).reshape(shape).astype("float32")
        std = np.array(std).reshape(shape).astype("float32")

        im_h, im_w = img.shape[:2]
        if (
            min_dimensions[0] <= im_w <= max_dimensions[0]
            and min_dimensions[1] <= im_h <= max_dimensions[1]
        ):
            pass
        else:
            img = Image.fromarray(np.uint8(img))
            img = self.minmax_size_(self.pad_(img), max_dimensions, min_dimensions)
            img = np.array(img)
            im_h, im_w = img.shape[:2]
            img = np.dstack([img, img, img])
        img = (img.astype("float32") * scale - mean) / std
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        divide_h = math.ceil(im_h / 16) * 16
        divide_w = math.ceil(im_w / 16) * 16
        img = np.pad(
            img, ((0, divide_h - im_h), (0, divide_w - im_w)), constant_values=(1, 1)
        )
        img = img[:, :, np.newaxis].transpose(2, 0, 1)
        img = img.astype("float32")
        return img

    def __call__(self, data):
        """apply"""
        img = data["img"]
        img = self.norm_img_latexocr(img)
        return {"img": img}
