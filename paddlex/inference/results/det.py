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


def draw_box(img, np_boxes, labels):
    """
    Args:
        img (PIL.Image.Image): PIL image
        np_boxes (np.ndarray): shape:[N,6], N: number of box,
                               matix element:[class, score, x_min, y_min, x_max, y_max]
        labels (list): labels:['class1', ..., 'classn']
    Returns:
        img (PIL.Image.Image): visualized image
    """
    font_size = int(0.024 * int(img.width)) + 2
    font = ImageFont.truetype(PINGFANG_FONT_FILE_PATH, font_size, encoding="utf-8")

    draw_thickness = int(max(img.size) * 0.005)
    draw = ImageDraw.Draw(img)
    clsid2color = {}
    catid2fontcolor = {}
    color_list = get_colormap(rgb=True)
    expect_boxes = np_boxes[:, 0] > -1
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
            [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin), (xmin, ymin)],
            width=draw_thickness,
            fill=color,
        )

        # draw label
        text = "{} {:.2f}".format(labels[clsid], score)
        if tuple(map(int, PIL.__version__.split("."))) <= (10, 0, 0):
            tw, th = draw.textsize(text, font=font)
        else:
            left, top, right, bottom = draw.textbbox((0, 0), text, font)
            tw, th = right - left, bottom - top
        if ymin < th:
            draw.rectangle([(xmin, ymin), (xmin + tw + 4, ymin + th + 1)], fill=color)
            draw.text((xmin + 2, ymin - 2), text, fill=font_color, font=font)
        else:
            draw.rectangle([(xmin, ymin - th), (xmin + tw + 4, ymin + 1)], fill=color)
            draw.text((xmin + 2, ymin - th - 2), text, fill=font_color, font=font)

    return img


class DetResults(BaseResult):
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
        img_path = self["img_path"]
        labels = self.data["labels"]
        file_name = os.path.basename(img_path)

        image = self._img_reader.read(img_path)
        image = draw_box(image, boxes, labels=labels)
        self["boxes"] = boxes.tolist()

        return image
