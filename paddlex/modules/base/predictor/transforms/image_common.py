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



import math
from pathlib import Path

import numpy as np
import cv2

from .....utils.download import download
from .....utils.cache import CACHE_DIR
from ..transform import BaseTransform
from ..io.readers import ImageReader
from ..io.writers import ImageWriter
from . import image_functions as F

__all__ = [
    'ReadImage', 'Flip', 'Crop', 'Resize', 'ResizeByLong', 'ResizeByShort',
    'Pad', 'Normalize', 'ToCHWImage'
]


def _check_image_size(input_):
    """ check image size """
    if not (isinstance(input_, (list, tuple)) and len(input_) == 2 and
            isinstance(input_[0], int) and isinstance(input_[1], int)):
        raise TypeError(f"{input_} cannot represent a valid image size.")


class ReadImage(BaseTransform):
    """Load image from the file."""

    _FLAGS_DICT = {
        'BGR': cv2.IMREAD_COLOR,
        'RGB': cv2.IMREAD_COLOR,
        'GRAY': cv2.IMREAD_GRAYSCALE
    }

    def __init__(self, format='BGR'):
        """
        Initialize the instance.

        Args:
            format (str, optional): Target color format to convert the image to.
                Choices are 'BGR', 'RGB', and 'GRAY'. Default: 'BGR'.
        """
        super().__init__()
        self.format = format
        flags = self._FLAGS_DICT[self.format]
        self._reader = ImageReader(backend='opencv', flags=flags)
        self._writer = ImageWriter(backend='opencv')

    def apply(self, data):
        """ apply """
        if 'image' in data:
            img = data['image']
            img_path = (Path(CACHE_DIR) / "predict_input" /
                        "tmp_img.jpg").as_posix()
            self._writer.write(img_path, img)
            data['input_path'] = img_path
            data['original_image'] = img
            data['original_image_size'] = [img.shape[1], img.shape[0]]
            return data

        elif 'input_path' not in data:
            raise KeyError(
                f"Key {repr('input_path')} is required, but not found.")

        im_path = data['input_path']
        # XXX: auto download for url
        im_path = self._download_from_url(im_path)
        blob = self._reader.read(im_path)
        if self.format == 'RGB':
            if blob.ndim != 3:
                raise RuntimeError("Array is not 3-dimensional.")
            # BGR to RGB
            blob = blob[..., ::-1]
        data['input_path'] = im_path
        data['image'] = blob
        data['original_image'] = blob
        data['original_image_size'] = [blob.shape[1], blob.shape[0]]
        return data

    def _download_from_url(self, in_path):
        if in_path.startswith("http"):
            file_name = Path(in_path).name
            save_path = Path(CACHE_DIR) / "predict_input" / file_name
            download(in_path, save_path, overwrite=True)
            return save_path.as_posix()
        return in_path

    @classmethod
    def get_input_keys(cls):
        """ get input keys """
        # input_path: Path of the image.
        return [['input_path'], ['image']]

    @classmethod
    def get_output_keys(cls):
        """ get output keys """
        # image: Image in hw or hwc format.
        # original_image: Original image in hw or hwc format.
        # original_image_size: Width and height of the original image.
        return ['image', 'original_image', 'original_image_size']


class GetImageInfo(BaseTransform):
    """Get Image Info
    """

    def __init__(self):
        super().__init__()

    def apply(self, data):
        """ apply """
        blob = data['image']
        data['original_image'] = blob
        data['original_image_size'] = [blob.shape[1], blob.shape[0]]
        return data

    @classmethod
    def get_input_keys(cls):
        """ get input keys """
        # input_path: Path of the image.
        return ['image']

    @classmethod
    def get_output_keys(cls):
        """ get output keys """
        # image: Image in hw or hwc format.
        # original_image: Original image in hw or hwc format.
        # original_image_size: Width and height of the original image.
        return ['original_image', 'original_image_size']


class Flip(BaseTransform):
    """Flip the image vertically or horizontally."""

    def __init__(self, mode='H'):
        """
        Initialize the instance.

        Args:
            mode (str, optional): 'H' for horizontal flipping and 'V' for vertical
                flipping. Default: 'H'.
        """
        super().__init__()
        if mode not in ('H', 'V'):
            raise ValueError("`mode` should be 'H' or 'V'.")
        self.mode = mode

    def apply(self, data):
        """ apply """
        im = data['image']
        if self.mode == 'H':
            im = F.flip_h(im)
        elif self.mode == 'V':
            im = F.flip_v(im)
        data['image'] = im
        return data

    @classmethod
    def get_input_keys(cls):
        """ get input keys """
        # image: Image in hw or hwc format.
        return ['image']

    @classmethod
    def get_output_keys(cls):
        """ get output keys """
        # image: Image in hw or hwc format.
        return ['image']


class Crop(BaseTransform):
    """Crop region from the image."""

    def __init__(self, crop_size, mode='C'):
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

        if mode not in ('C', 'TL'):
            raise ValueError("Unsupported interpolation method")
        self.mode = mode

    def apply(self, data):
        """ apply """
        im = data['image']
        h, w = im.shape[:2]
        cw, ch = self.crop_size
        if self.mode == 'C':
            x1 = max(0, (w - cw) // 2)
            y1 = max(0, (h - ch) // 2)
        elif self.mode == 'TL':
            x1, y1 = 0, 0
        x2 = min(w, x1 + cw)
        y2 = min(h, y1 + ch)
        coords = (x1, y1, x2, y2)
        if coords == (0, 0, w, h):
            raise ValueError(
                f"Input image ({w}, {h}) smaller than the target size ({cw}, {ch})."
            )
        im = F.slice(im, coords=coords)
        data['image'] = im
        data['image_size'] = [im.shape[1], im.shape[0]]
        return data

    @classmethod
    def get_input_keys(cls):
        """ get input keys """
        # image: Image in hw or hwc format.
        return ['image']

    @classmethod
    def get_output_keys(cls):
        """ get output keys """
        # image: Image in hw or hwc format.
        # image_size: Width and height of the image.
        return ['image', 'image_size']


class _BaseResize(BaseTransform):
    _INTERP_DICT = {
        'NEAREST': cv2.INTER_NEAREST,
        'LINEAR': cv2.INTER_LINEAR,
        'CUBIC': cv2.INTER_CUBIC,
        'AREA': cv2.INTER_AREA,
        'LANCZOS4': cv2.INTER_LANCZOS4
    }

    def __init__(self, size_divisor, interp):
        super().__init__()

        if size_divisor is not None:
            assert isinstance(size_divisor,
                              int), "`size_divisor` should be None or int."
        self.size_divisor = size_divisor

        try:
            interp = self._INTERP_DICT[interp]
        except KeyError:
            raise ValueError("`interp` should be one of {}.".format(
                self._INTERP_DICT.keys()))
        self.interp = interp

    @staticmethod
    def _rescale_size(img_size, target_size):
        """ rescale size """
        scale = min(
            max(target_size) / max(img_size), min(target_size) / min(img_size))
        rescaled_size = [round(i * scale) for i in img_size]
        return rescaled_size, scale


class Resize(_BaseResize):
    """Resize the image."""

    def __init__(self,
                 target_size,
                 keep_ratio=False,
                 size_divisor=None,
                 interp='LINEAR'):
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

    def apply(self, data):
        """ apply """
        target_size = self.target_size
        im = data['image']
        original_size = im.shape[:2]

        if self.keep_ratio:
            h, w = im.shape[0:2]
            target_size, _ = self._rescale_size((w, h), self.target_size)

        if self.size_divisor:
            target_size = [
                math.ceil(i / self.size_divisor) * self.size_divisor
                for i in target_size
            ]

        im_scale_w, im_scale_h = [
            target_size[1] / original_size[1], target_size[0] / original_size[0]
        ]
        im = F.resize(im, target_size, interp=self.interp)

        data['image'] = im
        data['image_size'] = [im.shape[1], im.shape[0]]
        data['scale_factors'] = [im_scale_w, im_scale_h]
        return data

    @classmethod
    def get_input_keys(cls):
        """ get input keys """
        # image: Image in hw or hwc format.
        return ['image']

    @classmethod
    def get_output_keys(cls):
        """ get output keys """
        # image: Image in hw or hwc format.
        # image_size: Width and height of the image.
        # scale_factors: Scale factors for image width and height.
        return ['image', 'image_size', 'scale_factors']


class ResizeByLong(_BaseResize):
    """
    Proportionally resize the image by specifying the target length of the
    longest side.
    """

    def __init__(self, target_long_edge, size_divisor=None, interp='LINEAR'):
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

    def apply(self, data):
        """ apply """
        im = data['image']

        h, w = im.shape[:2]
        scale = self.target_long_edge / max(h, w)
        h_resize = round(h * scale)
        w_resize = round(w * scale)
        if self.size_divisor is not None:
            h_resize = math.ceil(h_resize /
                                 self.size_divisor) * self.size_divisor
            w_resize = math.ceil(w_resize /
                                 self.size_divisor) * self.size_divisor

        im = F.resize(im, (w_resize, h_resize), interp=self.interp)

        data['image'] = im
        data['image_size'] = [im.shape[1], im.shape[0]]
        return data

    @classmethod
    def get_input_keys(cls):
        """ get input keys """
        # image: Image in hw or hwc format.
        return ['image']

    @classmethod
    def get_output_keys(cls):
        """ get output keys """
        # image: Image in hw or hwc format.
        # image_size: Width and height of the image.
        return ['image', 'image_size']


class ResizeByShort(_BaseResize):
    """
    Proportionally resize the image by specifying the target length of the
    shortest side.
    """

    def __init__(self, target_short_edge, size_divisor=None, interp='LINEAR'):
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

    def apply(self, data):
        """ apply """
        im = data['image']

        h, w = im.shape[:2]
        scale = self.target_short_edge / min(h, w)
        h_resize = round(h * scale)
        w_resize = round(w * scale)
        if self.size_divisor is not None:
            h_resize = math.ceil(h_resize /
                                 self.size_divisor) * self.size_divisor
            w_resize = math.ceil(w_resize /
                                 self.size_divisor) * self.size_divisor

        im = F.resize(im, (w_resize, h_resize), interp=self.interp)

        data['image'] = im
        data['image_size'] = [im.shape[1], im.shape[0]]
        return data

    @classmethod
    def get_input_keys(cls):
        """ get input keys """
        # image: Image in hw or hwc format.
        return ['image']

    @classmethod
    def get_output_keys(cls):
        """ get output keys """
        # image: Image in hw or hwc format.
        # image_size: Width and height of the image.
        return ['image', 'image_size']


class Pad(BaseTransform):
    """Pad the image."""

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

    def apply(self, data):
        """ apply """
        im = data['image']

        h, w = im.shape[:2]
        tw, th = self.target_size
        ph = th - h
        pw = tw - w

        if ph < 0 or pw < 0:
            raise ValueError(
                f"Input image ({w}, {h}) smaller than the target size ({tw}, {th})."
            )
        else:
            im = F.pad(im, pad=(0, ph, 0, pw), val=self.val)

        data['image'] = im
        data['image_size'] = [im.shape[1], im.shape[0]]

        return data

    @classmethod
    def get_input_keys(cls):
        """ get input keys """
        # image: Image in hw or hwc format.
        return ['image']

    @classmethod
    def get_output_keys(cls):
        """ get output keys """
        # image: Image in hw or hwc format.
        # image_size: Width and height of the image.
        return ['image', 'image_size']


class Normalize(BaseTransform):
    """Normalize the image."""

    def __init__(self, scale=1. / 255, mean=0.5, std=0.5, preserve_dtype=False):
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
        self.mean = np.asarray(mean).astype('float32')
        if isinstance(std, float):
            std = [std]
        self.std = np.asarray(std).astype('float32')
        self.preserve_dtype = preserve_dtype

    def apply(self, data):
        """ apply """
        im = data['image']
        old_type = im.dtype
        # XXX: If `old_type` has higher precision than float32,
        # we will lose some precision.
        im = im.astype('float32', copy=False)
        im *= self.scale
        im -= self.mean
        im /= self.std
        if self.preserve_dtype:
            im = im.astype(old_type, copy=False)
        data['image'] = im
        return data

    @classmethod
    def get_input_keys(cls):
        """ get input keys """
        # image: Image in hw or hwc format.
        return ['image']

    @classmethod
    def get_output_keys(cls):
        """ get output keys """
        # image: Image in hw or hwc format.
        return ['image']


class ToCHWImage(BaseTransform):
    """Reorder the dimensions of the image from HWC to CHW."""

    def apply(self, data):
        """ apply """
        im = data['image']
        im = im.transpose((2, 0, 1))
        data['image'] = im
        return data

    @classmethod
    def get_input_keys(cls):
        """ get input keys """
        # image: Image in hwc format.
        return ['image']

    @classmethod
    def get_output_keys(cls):
        """ get output keys """
        # image: Image in chw format.
        return ['image']


def rotate_point(pt, angle_rad):
    """Rotate a point by an angle.

    Args:
        pt (list[float]): 2 dimensional point to be rotated
        angle_rad (float): rotation angle by radian

    Returns:
        list[float]: Rotated point.
    """
    assert len(pt) == 2
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    new_x = pt[0] * cs - pt[1] * sn
    new_y = pt[0] * sn + pt[1] * cs
    rotated_pt = [new_x, new_y]

    return rotated_pt


def _get_3rd_point(a, b):
    """To calculate the affine matrix, three pairs of points are required. This
    function is used to get the 3rd point, given 2D points a & b.

    The 3rd point is defined by rotating vector `a - b` by 90 degrees
    anticlockwise, using b as the rotation center.

    Args:
        a (np.ndarray): point(x,y)
        b (np.ndarray): point(x,y)

    Returns:
        np.ndarray: The 3rd point.
    """
    assert len(a) == 2
    assert len(b) == 2
    direction = a - b
    third_pt = b + np.array([-direction[1], direction[0]], dtype=np.float32)

    return third_pt


def get_affine_transform(center,
                         input_size,
                         rot,
                         output_size,
                         shift=(0., 0.),
                         inv=False):
    """Get the affine transform matrix, given the center/scale/rot/output_size.

    Args:
        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        rot (float): Rotation angle (degree).
        output_size (np.ndarray[2, ]): Size of the destination heatmaps.
        shift (0-100%): Shift translation ratio wrt the width/height.
            Default (0., 0.).
        inv (bool): Option to inverse the affine transform direction.
            (inv=False: src->dst or inv=True: dst->src)

    Returns:
        np.ndarray: The transform matrix.
    """
    assert len(center) == 2
    assert len(output_size) == 2
    assert len(shift) == 2
    if not isinstance(input_size, (np.ndarray, list)):
        input_size = np.array([input_size, input_size], dtype=np.float32)
    scale_tmp = input_size

    shift = np.array(shift)
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = rotate_point([0., src_w * -0.5], rot_rad)
    dst_dir = np.array([0., dst_w * -0.5])

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


class WarpAffine(object):
    """Warp affine the image
    """

    def __init__(self,
                 keep_res=False,
                 pad=31,
                 input_h=512,
                 input_w=512,
                 scale=0.4,
                 shift=0.1,
                 down_ratio=4):
        self.keep_res = keep_res
        self.pad = pad
        self.input_h = input_h
        self.input_w = input_w
        self.scale = scale
        self.shift = shift
        self.down_ratio = down_ratio

    def __call__(self, data):
        """
        Args:
            im (np.ndarray): image (np.ndarray)
            im_info (dict): info of image
        Returns:
            im (np.ndarray):  processed image (np.ndarray)
            im_info (dict): info of processed image
        """

        im = data['image']
        img = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

        h, w = img.shape[:2]

        if self.keep_res:
            # True in detection eval/infer
            input_h = (h | self.pad) + 1
            input_w = (w | self.pad) + 1
            s = np.array([input_w, input_h], dtype=np.float32)
            c = np.array([w // 2, h // 2], dtype=np.float32)

        else:
            # False in centertrack eval_mot/eval_mot
            s = max(h, w) * 1.0
            input_h, input_w = self.input_h, self.input_w
            c = np.array([w / 2., h / 2.], dtype=np.float32)

        trans_input = get_affine_transform(c, s, 0, [input_w, input_h])
        img = cv2.resize(img, (w, h))
        inp = cv2.warpAffine(
            img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)

        if not self.keep_res:
            out_h = input_h // self.down_ratio
            out_w = input_w // self.down_ratio
            trans_output = get_affine_transform(c, s, 0, [out_w, out_h])

        data['image'] = inp

        im_scale_w, im_scale_h = [
            input_w / w, input_h / h
        ]
        data['image_size'] = [inp.shape[1], inp.shape[0]]
        data['scale_factors'] = [im_scale_w, im_scale_h]
        return data

    # @classmethod
    def get_input_keys(cls):
        """ get input keys """
        # image: Image in hwc format.
        return ['image']

    # @classmethod
    def get_output_keys(cls):
        """ get output keys """
        # image: Image in chw format.
