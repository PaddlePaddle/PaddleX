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
from ..utils.color_map import get_color_map_list, font_colormap
from .base import BaseResult
from .det import draw_box


def draw_mask(im, np_boxes, np_masks, labels):
    """
    Args:
        im (PIL.Image.Image): PIL image
        np_boxes (np.ndarray): shape:[N,6], N: number of box,
            matix element:[class, score, x_min, y_min, x_max, y_max]
        np_masks (np.ndarray): shape:[N, im_h, im_w]
        labels (list): labels:['class1', ..., 'classn']
    Returns:
        im (PIL.Image.Image): visualized image
    """
    color_list = get_color_map_list(len(labels))
    w_ratio = 0.4
    alpha = 0.7
    im = np.array(im).astype("float32")
    clsid2color = {}
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
    return Image.fromarray(im.astype("uint8"))


class InstanceSegResults(BaseResult):
    """Save Result Transform"""

    def __init__(self, data):
        super().__init__(data)
        self.data = data
        # We use pillow backend to save both numpy arrays and PIL Image objects
        self._img_reader.set_backend("pillow")
        self._img_writer.set_backend("pillow")

    def _get_res_img(self):
        """apply"""
        boxes = self["boxes"]
        masks = self["masks"]
        img_path = self["img_path"]
        labels = self.data["labels"]
        file_name = os.path.basename(img_path)

        image = self._img_reader.read(img_path)
        image = draw_mask(image, boxes, masks, labels)
        image = draw_box(image, boxes, labels=labels)
        self["boxes"] = boxes.tolist()
        self["masks"] = masks.tolist()

        return image
