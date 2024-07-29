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

import numpy as np
import math
import PIL
from PIL import Image, ImageDraw, ImageFont

from .keys import DetKeys as K
from ...base import BaseTransform
from ...base.predictor.io import ImageWriter, ImageReader
from ...base.predictor.transforms import image_functions as F
from ...base.predictor.transforms.image_common import _BaseResize, _check_image_size
from ....utils.fonts import PINGFANG_FONT_FILE_PATH
from ....utils import logging

__all__ = ['SaveDetResults', 'PadStride', 'DetResize', 'PrintResult']


def get_color_map_list(num_classes):
    """
    Args:
        num_classes (int): number of class
    Returns:
        color_map (list): RGB color list
    """
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    color_map = [color_map[i:i + 3] for i in range(0, len(color_map), 3)]
    return color_map


def colormap(rgb=False):
    """
    Get colormap

    The code of this function is copied from https://github.com/facebookresearch/Detectron/blob/main/detectron/\
utils/colormap.py
    """
    color_list = np.array([
        0xFF, 0x00, 0x00, 0xCC, 0xFF, 0x00, 0x00, 0xFF, 0x66, 0x00, 0x66, 0xFF,
        0xCC, 0x00, 0xFF, 0xFF, 0x4D, 0x00, 0x80, 0xff, 0x00, 0x00, 0xFF, 0xB2,
        0x00, 0x1A, 0xFF, 0xFF, 0x00, 0xE5, 0xFF, 0x99, 0x00, 0x33, 0xFF, 0x00,
        0x00, 0xFF, 0xFF, 0x33, 0x00, 0xFF, 0xff, 0x00, 0x99, 0xFF, 0xE5, 0x00,
        0x00, 0xFF, 0x1A, 0x00, 0xB2, 0xFF, 0x80, 0x00, 0xFF, 0xFF, 0x00, 0x4D
    ]).astype(np.float32)
    color_list = (color_list.reshape((-1, 3)))
    if not rgb:
        color_list = color_list[:, ::-1]
    return color_list.astype('int32')


def font_colormap(color_index):
    """
    Get font color according to the index of colormap
    """
    dark = np.array([0x14, 0x0E, 0x35])
    light = np.array([0xFF, 0xFF, 0xFF])
    light_indexs = [0, 3, 4, 8, 9, 13, 14, 18, 19]
    if color_index in light_indexs:
        return light.astype('int32')
    else:
        return dark.astype('int32')


def draw_box(img, np_boxes, labels, threshold=0.5):
    """
    Args:
        img (PIL.Image.Image): PIL image
        np_boxes (np.ndarray): shape:[N,6], N: number of box,
                               matix element:[class, score, x_min, y_min, x_max, y_max]
        labels (list): labels:['class1', ..., 'classn']
        threshold (float): threshold of box
    Returns:
        img (PIL.Image.Image): visualized image
    """
    font_size = int(0.024 * int(img.width)) + 2
    font = ImageFont.truetype(
        PINGFANG_FONT_FILE_PATH, font_size, encoding="utf-8")

    draw_thickness = int(max(img.size) * 0.005)
    draw = ImageDraw.Draw(img)
    clsid2color = {}
    catid2fontcolor = {}
    color_list = colormap(rgb=True)
    expect_boxes = (np_boxes[:, 1] > threshold) & (np_boxes[:, 0] > -1)
    np_boxes = np_boxes[expect_boxes, :]

    for i, dt in enumerate(np_boxes):
        clsid, bbox, score = int(dt[0]), dt[2:], dt[1]
        if clsid not in clsid2color:
            color_index = i % len(color_list)
            clsid2color[clsid] = color_list[color_index]
            catid2fontcolor[clsid] = font_colormap(color_index)
        color = tuple(clsid2color[clsid])
        font_color = tuple(catid2fontcolor[clsid])

        xmin, ymin, xmax, ymax = bbox
        # draw bbox
        draw.line(
            [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin),
             (xmin, ymin)],
            width=draw_thickness,
            fill=color)

        # draw label
        text = "{} {:.2f}".format(labels[clsid], score)
        if tuple(map(int, PIL.__version__.split('.'))) <= (10, 0, 0):
            tw, th = draw.textsize(text, font=font)
        else:
            left, top, right, bottom = draw.textbbox((0, 0), text, font)
            tw, th = right - left, bottom - top
        if ymin < th:
            draw.rectangle(
                [(xmin, ymin), (xmin + tw + 4, ymin + th + 1)], fill=color)
            draw.text((xmin + 2, ymin - 2), text, fill=font_color, font=font)
        else:
            draw.rectangle(
                [(xmin, ymin - th), (xmin + tw + 4, ymin + 1)], fill=color)
            draw.text(
                (xmin + 2, ymin - th - 2), text, fill=font_color, font=font)

    return img


def draw_mask(im, np_boxes, np_masks, labels, threshold=0.5):
    """
    Args:
        im (PIL.Image.Image): PIL image
        np_boxes (np.ndarray): shape:[N,6], N: number of box,
            matix element:[class, score, x_min, y_min, x_max, y_max]
        np_masks (np.ndarray): shape:[N, im_h, im_w]
        labels (list): labels:['class1', ..., 'classn']
        threshold (float): threshold of mask
    Returns:
        im (PIL.Image.Image): visualized image
    """
    color_list = get_color_map_list(len(labels))
    w_ratio = 0.4
    alpha = 0.7
    im = np.array(im).astype('float32')
    clsid2color = {}
    expect_boxes = (np_boxes[:, 1] > threshold) & (np_boxes[:, 0] > -1)
    np_boxes = np_boxes[expect_boxes, :]
    np_masks = np_masks[expect_boxes, :, :]
    im_h, im_w = im.shape[:2]
    np_masks = np_masks[:, :im_h, :im_w]
    for i in range(len(np_masks)):
        clsid, score = int(np_boxes[i][0]), np_boxes[i][1]
        mask = np_masks[i]
        if clsid not in clsid2color:
            clsid2color[clsid] = color_list[clsid]
        color_mask = clsid2color[clsid]
        for c in range(3):
            color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio * 255
        idx = np.nonzero(mask)
        color_mask = np.array(color_mask)
        im[idx[0], idx[1], :] *= 1.0 - alpha
        im[idx[0], idx[1], :] += alpha * color_mask
    return Image.fromarray(im.astype('uint8'))


class SaveDetResults(BaseTransform):
    """ Save Result Transform """

    def __init__(self, save_dir, threshold=0.5, labels=None):
        super().__init__()
        self.save_dir = save_dir
        self.threshold = threshold
        self.labels = labels

        # We use pillow backend to save both numpy arrays and PIL Image objects
        self._writer = ImageWriter(backend='pillow')

    def apply(self, data):
        """ apply """
        ori_path = data[K.IM_PATH]
        file_name = os.path.basename(ori_path)
        save_path = os.path.join(self.save_dir, file_name)

        labels = self.labels
        image = ImageReader(backend='pil').read(ori_path)
        if K.MASKS in data:
            image = draw_mask(
                image,
                data[K.BOXES],
                data[K.MASKS],
                threshold=self.threshold,
                labels=labels)
        image = draw_box(
            image, data[K.BOXES], threshold=self.threshold, labels=labels)

        self._write_image(save_path, image)
        return data

    def _write_image(self, path, image):
        """ write image """
        if os.path.exists(path):
            logging.warning(f"{path} already exists. Overwriting it.")
        self._writer.write(path, image)

    @classmethod
    def get_input_keys(cls):
        """ get input keys """
        return [K.IM_PATH, K.BOXES]

    @classmethod
    def get_output_keys(cls):
        """ get output keys """
        return []


class PadStride(BaseTransform):
    """ padding image for model with FPN , instead PadBatch(pad_to_stride, pad_gt) in original config
    Args:
        stride (bool): model with FPN need image shape % stride == 0
    """

    def __init__(self, stride=0):
        self.coarsest_stride = stride

    def apply(self, data):
        """
        Args:
            im (np.ndarray): image (np.ndarray)
        Returns:
            im (np.ndarray):  processed image (np.ndarray)
        """
        im = data[K.IMAGE]
        coarsest_stride = self.coarsest_stride
        if coarsest_stride <= 0:
            return data
        im_c, im_h, im_w = im.shape
        pad_h = int(np.ceil(float(im_h) / coarsest_stride) * coarsest_stride)
        pad_w = int(np.ceil(float(im_w) / coarsest_stride) * coarsest_stride)
        padding_im = np.zeros((im_c, pad_h, pad_w), dtype=np.float32)
        padding_im[:, :im_h, :im_w] = im
        data[K.IMAGE] = padding_im
        return data

    @classmethod
    def get_input_keys(cls):
        """ get input keys """
        return [K.IMAGE]

    @classmethod
    def get_output_keys(cls):
        """ get output keys """
        return [K.IMAGE]


class DetResize(_BaseResize):
    """
    Resize the image.

    Args:
        target_size (list|tuple|int): Target height and width.
        keep_ratio (bool, optional): Whether to keep the aspect ratio of resized
            image. Default: False.
        size_divisor (int|None, optional): Divisor of resized image size.
            Default: None.
        interp (str, optional): Interpolation method. Choices are 'NEAREST',
            'LINEAR', 'CUBIC', 'AREA', and 'LANCZOS4'. Default: 'LINEAR'.
    """

    def __init__(self,
                 target_hw,
                 keep_ratio=False,
                 size_divisor=None,
                 interp='LINEAR'):
        super().__init__(size_divisor=size_divisor, interp=interp)

        if isinstance(target_hw, int):
            target_hw = [target_hw, target_hw]
        _check_image_size(target_hw)
        self.target_hw = target_hw

        self.keep_ratio = keep_ratio

    def apply(self, data):
        """ apply """
        target_hw = self.target_hw
        im = data['image']
        original_size = im.shape[:2]

        if self.keep_ratio:
            h, w = im.shape[0:2]
            target_hw, _ = self._rescale_size((h, w), self.target_hw)

        if self.size_divisor:
            target_hw = [
                math.ceil(i / self.size_divisor) * self.size_divisor
                for i in target_hw
            ]

        im_scale_w, im_scale_h = [
            target_hw[1] / original_size[1], target_hw[0] / original_size[0]
        ]
        im = F.resize(im, target_hw[::-1], interp=self.interp)

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


class PrintResult(BaseTransform):
    """ Print Result Transform """

    def apply(self, data):
        """ apply """
        logging.info("The prediction result is:")
        logging.info(data[K.BOXES])
        return data

    @classmethod
    def get_input_keys(cls):
        """ get input keys """
        return [K.BOXES]

    @classmethod
    def get_output_keys(cls):
        """ get output keys """
        return []
