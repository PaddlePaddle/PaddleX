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

from ....utils.fonts import PINGFANG_FONT_FILE_PATH
from ...common.result import BaseCVResult, JsonMixin
from ...utils.color_map import font_colormap, get_colormap


def draw_box(img, boxes):
    """
    Args:
        img (PIL.Image.Image): PIL image
        boxes (list): a list of dictionaries representing detection box information.
    Returns:
        img (PIL.Image.Image): visualized image
    """
    img = Image.fromarray(img)
    font_size = int(0.018 * int(img.width)) + 2
    font = ImageFont.truetype(PINGFANG_FONT_FILE_PATH, font_size, encoding="utf-8")

    draw_thickness = int(max(img.size) * 0.002)
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

        if len(bbox) == 4:
            # draw bbox of normal object detection
            xmin, ymin, xmax, ymax = bbox
            rectangle = [
                (xmin, ymin),
                (xmin, ymax),
                (xmax, ymax),
                (xmax, ymin),
                (xmin, ymin),
            ]
        elif len(bbox) == 8:
            # draw bbox of rotated object detection
            x1, y1, x2, y2, x3, y3, x4, y4 = bbox
            rectangle = [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x1, y1)]
            xmin = min(x1, x2, x3, x4)
            ymin = min(y1, y2, y3, y4)
        else:
            raise ValueError(
                f"Only support bbox format of [xmin,ymin,xmax,ymax] or [x1,y1,x2,y2,x3,y3,x4,y4], got bbox of shape {len(bbox)}."
            )

        # draw bbox
        draw.line(
            rectangle,
            width=draw_thickness,
            fill=color,
        )

        # draw label
        if score is not None:
            text = "{} {:.2f}".format(dt["label"], score)
        else:
            text = "{}".format(dt["label"])

        if tuple(map(int, PIL.__version__.split("."))) <= (10, 0, 0):
            tw, th = draw.textsize(text, font=font)
        else:
            left, top, right, bottom = draw.textbbox((0, 0), text, font)
            tw, th = right - left, bottom - top + 4
        if ymin < th:
            draw.rectangle([(xmin, ymin), (xmin + tw + 4, ymin + th + 1)], fill=color)
            draw.text((xmin + 2, ymin - 2), text, fill=font_color, font=font)
        else:
            draw.rectangle([(xmin, ymin - th), (xmin + tw + 4, ymin + 1)], fill=color)
            draw.text((xmin + 2, ymin - th - 2), text, fill=font_color, font=font)

    return img


class ShiTuResult(BaseCVResult):

    def _to_img(self):
        """apply"""
        boxes = [
            {
                "coordinate": box["coordinate"],
                "label": box["labels"][0],
                "score": box["rec_scores"][0],
            }
            for box in self["boxes"]
            if box["rec_scores"] is not None
        ]
        image = draw_box(self["input_img"][..., ::-1], boxes)
        return {"res": image}

    def _to_str(self, *args, **kwargs):
        data = copy.deepcopy(self)
        data.pop("input_img")
        return JsonMixin._to_str(data, *args, **kwargs)

    def _to_json(self, *args, **kwargs):
        data = copy.deepcopy(self)
        data.pop("input_img")
        return JsonMixin._to_json(data, *args, **kwargs)
