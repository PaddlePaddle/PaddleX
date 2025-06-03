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

import copy

import PIL
from PIL import Image, ImageDraw, ImageFont

from ....utils.deps import class_requires_deps, is_dep_available
from ....utils.fonts import PINGFANG_FONT_FILE_PATH
from ...common.result import BaseCVResult, JsonMixin
from ...utils.color_map import font_colormap, get_colormap

if is_dep_available("opencv-contrib-python"):
    import cv2


def draw_attribute_result(img, boxes):
    """
    Args:
        img (PIL.Image.Image): PIL image
        boxes (list): a list of dictionaries representing detection box information.
    Returns:
        img (PIL.Image.Image): visualized image
    """
    font_size = int((0.024 * int(img.width) + 2) * 0.7)
    font = ImageFont.truetype(PINGFANG_FONT_FILE_PATH, font_size, encoding="utf-8")

    draw_thickness = int(max(img.size) * 0.005)
    draw = ImageDraw.Draw(img)
    label2color = {}
    catid2fontcolor = {}
    color_list = get_colormap(rgb=True)

    for i, dt in enumerate(boxes):
        text_lines, bbox, score = dt["label"], dt["coordinate"], dt["score"]
        if i not in label2color:
            color_index = i % len(color_list)
            label2color[i] = color_list[color_index]
            catid2fontcolor[i] = font_colormap(color_index)
        color = tuple(label2color[i]) + (255,)
        tuple(catid2fontcolor[i])

        xmin, ymin, xmax, ymax = bbox
        # draw box
        draw.line(
            [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin), (xmin, ymin)],
            width=draw_thickness,
            fill=color,
        )
        # draw label
        current_y = ymin
        for line in text_lines:
            if tuple(map(int, PIL.__version__.split("."))) <= (10, 0, 0):
                tw, th = draw.textsize(line, font=font)
            else:
                left, top, right, bottom = draw.textbbox((0, 0), line, font)
                tw, th = right - left, bottom - top + 4

            draw.text((5 + xmin + 1, current_y + 1), line, fill=(0, 0, 0), font=font)
            draw.text((5 + xmin, current_y), line, fill=color, font=font)
            current_y += th
    return img


@class_requires_deps("opencv-contrib-python")
class AttributeRecResult(BaseCVResult):

    def _to_str(self, *args, **kwargs):
        data = copy.deepcopy(self)
        data.pop("input_img")
        return JsonMixin._to_str(data, *args, **kwargs)

    def _to_json(self, *args, **kwargs):
        data = copy.deepcopy(self)
        data.pop("input_img")
        return JsonMixin._to_json(data, *args, **kwargs)

    def _to_img(self):
        """apply"""
        image = Image.fromarray(cv2.cvtColor(self["input_img"], cv2.COLOR_BGR2RGB))
        boxes = [
            {
                "coordinate": box["coordinate"],
                "label": box["labels"],
                "score": box["det_score"],
            }
            for box in self["boxes"]
        ]
        image = draw_attribute_result(image, boxes)
        return {"res": image}
