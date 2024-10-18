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
import cv2
import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont

from ...utils.fonts import PINGFANG_FONT_FILE_PATH
from ..utils.color_map import get_colormap, font_colormap
from .base import CVResult


def draw_box(img, boxes):
    """
    Args:
        img (PIL.Image.Image): PIL image
        boxes (list): a list of dictionaries representing detection box information.
    Returns:
        img (PIL.Image.Image): visualized image
    """
    font_size = int(0.024 * int(img.width)) + 2
    font = ImageFont.truetype(PINGFANG_FONT_FILE_PATH, font_size, encoding="utf-8")

    draw_thickness = int(max(img.size) * 0.005)
    draw = ImageDraw.Draw(img)
    label2color = {}
    catid2fontcolor = {}
    color_list = get_colormap(rgb=True)

    for i, dt in enumerate(boxes):
        # clsid = dt["cls_id"]
        label, bbox, score = dt["label"], dt["coordinate"], dt["score"]
        if label not in label2color:
            color_index = i % len(color_list)
            label2color[label] = color_list[color_index]
            catid2fontcolor[label] = font_colormap(color_index)
        color = tuple(label2color[label])
        font_color = tuple(catid2fontcolor[label])

        xmin, ymin, xmax, ymax = bbox
        # draw bbox
        draw.line(
            [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin), (xmin, ymin)],
            width=draw_thickness,
            fill=color,
        )

        # draw label
        text = "{} {:.2f}".format(dt["label"], score)
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


class DetResult(CVResult):
    """Save Result Transform"""

    _HARD_FLAG = False

    def _to_img(self):
        """apply"""
        boxes = self["boxes"]
        image = self._img_reader.read(self["input_path"])
        if self._HARD_FLAG:
            image_np = np.array(image)
            image = Image.fromarray(image_np[:, :, ::-1])
        image = draw_box(image, boxes)
        return image
