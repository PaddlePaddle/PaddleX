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

from ...utils import logging
from ...utils.fonts import PINGFANG_FONT_FILE_PATH
from ..utils.io import ImageWriter, ImageReader
from ..utils.color_map import get_colormap, font_colormap
from .base import BaseResult
from .det import draw_box


def restore_to_draw_masks(img_size, boxes, masks):
    """
    Restores extracted masks to the original shape and draws them on a blank image.

    """

    restored_masks = []

    for i, (box, mask) in enumerate(zip(boxes, masks)):
        restored_mask = np.zeros(img_size, dtype=np.uint8)
        x_min, y_min, x_max, y_max = map(lambda x: int(round(x)), box["coordinate"])
        restored_mask[y_min:y_max, x_min:x_max] = mask
        restored_masks.append(restored_mask)

    return np.array(restored_masks)


def draw_mask(im, boxes, np_masks, img_size):
    """
    Args:
        im (PIL.Image.Image): PIL image
        boxes (list): a list of dictionaries representing detection box information.
        np_masks (np.ndarray): shape:[N, im_h, im_w]
    Returns:
        im (PIL.Image.Image): visualized image
    """
    color_list = get_colormap(rgb=True)
    w_ratio = 0.4
    alpha = 0.7
    im = np.array(im).astype("float32")
    clsid2color = {}
    np_masks = restore_to_draw_masks(img_size, boxes, np_masks)
    im_h, im_w = im.shape[:2]
    np_masks = np_masks[:, :im_h, :im_w]
    for i in range(len(np_masks)):
        clsid, score = int(boxes[i]["cls_id"]), boxes[i]["score"]
        mask = np_masks[i]
        if clsid not in clsid2color:
            color_index = i % len(color_list)
            clsid2color[clsid] = color_list[color_index]
        color_mask = clsid2color[clsid]
        for c in range(3):
            color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio * 255
        idx = np.nonzero(mask)
        color_mask = np.array(color_mask)
        im[idx[0], idx[1], :] *= 1.0 - alpha
        im[idx[0], idx[1], :] += alpha * color_mask
    return Image.fromarray(im.astype("uint8"))


class InstanceSegResult(BaseResult):
    """Save Result Transform"""

    def __init__(self, data):
        super().__init__(data)
        # We use pillow backend to save both numpy arrays and PIL Image objects
        self._img_reader.set_backend("pillow")
        self._img_writer.set_backend("pillow")

    def _get_res_img(self):
        """apply"""
        boxes = np.array(self["boxes"])
        masks = self["masks"]
        img_path = self["img_path"]
        file_name = os.path.basename(img_path)

        image = self._img_reader.read(img_path)
        ori_img_size = list(image.size)[::-1]
        image = draw_mask(image, boxes, masks, ori_img_size)
        image = draw_box(image, boxes)

        return image
