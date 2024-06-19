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
from PIL import Image, ImageDraw, ImageFile

from ....utils import logging
from ...base import BaseTransform
from .keys import DetKeys as K
from ...base.predictor.io.writers import ImageWriter
from ...base.predictor.transforms import image_functions as F
from ...base.predictor.transforms.image_common import _BaseResize, _check_image_size

__all__ = [
    'SaveDetResults', 'PadStride', 'DetResize', 'PrintResult', 'LoadLabels'
]


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
    draw_thickness = min(img.size) // 320
    draw = ImageDraw.Draw(img)
    clsid2color = {}
    color_list = get_color_map_list(len(labels))
    expect_boxes = (np_boxes[:, 1] > threshold) & (np_boxes[:, 0] > -1)
    np_boxes = np_boxes[expect_boxes, :]

    for dt in np_boxes:
        clsid, bbox, score = int(dt[0]), dt[2:], dt[1]
        if clsid not in clsid2color:
            clsid2color[clsid] = color_list[clsid]
        color = tuple(clsid2color[clsid])

        xmin, ymin, xmax, ymax = bbox
        # draw bbox
        draw.line(
            [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin),
             (xmin, ymin)],
            width=draw_thickness,
            fill=color)

        # draw label
        text = "{} {:.4f}".format(labels[clsid], score)
        tw, th = draw.textsize(text)
        draw.rectangle(
            [(xmin + 1, ymin - th), (xmin + tw + 1, ymin)], fill=color)
        draw.text((xmin + 1, ymin - th), text, fill=(255, 255, 255))
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
        ori_path = data[K.IMAGE_PATH]
        file_name = os.path.basename(ori_path)
        save_path = os.path.join(self.save_dir, file_name)

        labels = data[
            K.
            LABELS] if self.labels is None and K.LABELS in data else self.labels
        image = Image.open(ori_path)
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
        return [K.IMAGE_PATH, K.BOXES]

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


class LoadLabels(BaseTransform):
    def __init__(self, labels=None):
        super().__init__()
        self.labels = labels

    def apply(self, data):
        """ apply """
        if self.labels:
            data[K.LABELS] = self.labels
        return data

    @classmethod
    def get_input_keys(cls):
        """ get input keys """
        return []

    @classmethod
    def get_output_keys(cls):
        """ get output keys """
        return [K.LABELS]