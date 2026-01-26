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
from typing import List

import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont

from ....utils.deps import function_requires_deps, is_dep_available
from ....utils.fonts import PINGFANG_FONT
from ...common.result import BaseCVResult, JsonMixin
from ...utils.color_map import font_colormap, get_colormap

if is_dep_available("opencv-contrib-python"):
    import cv2


def draw_box(img: Image.Image, boxes: List[dict]) -> Image.Image:
    """
    Args:
        img (PIL.Image.Image): PIL image
        boxes (list): a list of dictionaries representing detection box information.
    Returns:
        img (PIL.Image.Image): visualized image
    """
    font_size = int(0.018 * int(img.width)) + 2
    font = ImageFont.truetype(PINGFANG_FONT.path, font_size, encoding="utf-8")

    draw_thickness = int(max(img.size) * 0.002)
    draw = ImageDraw.Draw(img)
    label2color = {}
    catid2fontcolor = {}
    color_list = get_colormap(rgb=True)

    for i, dt in enumerate(boxes):
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
        text = "{} {:.2f}".format(dt["label"], score)
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

        text_position = (bbox[2] + 2, bbox[1] - font_size // 2)
        if int(img.width) - bbox[2] < font_size:
            text_position = (
                int(bbox[2] - font_size * 1.1),
                bbox[1] - font_size // 2,
            )
        draw.text(text_position, str(i + 1), font=font, fill="red")

    return img


@function_requires_deps("opencv-contrib-python")
def restore_to_draw_masks(img_size, boxes):
    """
    Restores extracted masks to the original shape and draws them on a blank image.

    """
    restored_masks = []

    for i, box_info in enumerate(boxes):
        restored_mask = np.zeros(img_size, dtype=np.uint8)
        polygon = np.array(box_info["polygon_points"], dtype=np.int32)
        polygon = polygon.reshape((-1, 1, 2))  # shape: (N, 1, 2)
        cv2.fillPoly(restored_mask, [polygon], 1)
        restored_masks.append(restored_mask)

    return np.array(restored_masks)


def draw_mask(im, boxes, img_size):
    """
    Args:
        im (PIL.Image.Image): PIL image
        boxes (list): a list of dicts representing detection box information.
    Returns:
        img (PIL.Image.Image): visualized image
    """
    color_list = get_colormap(rgb=True)
    alpha = 0.5

    im = np.array(im).astype("float32")
    clsid2color = {}

    np_masks = restore_to_draw_masks(img_size, boxes)
    im_h, im_w = im.shape[:2]
    np_masks = np_masks[:, :im_h, :im_w]

    # draw mask
    for i, mask in enumerate(np_masks):
        clsid = int(boxes[i]["cls_id"])
        if clsid not in clsid2color:
            color_index = i % len(color_list)
            clsid2color[clsid] = np.array(color_list[color_index])
        color_mask = clsid2color[clsid]
        idx = np.nonzero(mask)
        im[idx[0], idx[1], :] = (1.0 - alpha) * im[
            idx[0], idx[1], :
        ] + alpha * color_mask

    img = Image.fromarray(np.uint8(im))
    font_size = int(0.018 * img.width) + 2
    font = ImageFont.truetype(PINGFANG_FONT.path, font_size, encoding="utf-8")
    draw = ImageDraw.Draw(img)
    label2color = {}
    catid2fontcolor = {}

    for i, box_info in enumerate(boxes):
        label = box_info["label"]
        score = box_info["score"]
        if label not in label2color:
            color_index = i % len(color_list)
            label2color[label] = color_list[color_index]
            catid2fontcolor[label] = font_colormap(color_index)
        color = tuple(label2color[label])
        font_color = tuple(catid2fontcolor[label])

        polygon_points = box_info["polygon_points"]

        image_left_top = (0, 0)
        image_right_top = (img.width, 0)
        left_top = min(
            polygon_points,
            key=lambda p: (
                (p[0] - image_left_top[0]) ** 2 + (p[1] - image_left_top[1]) ** 2
            ),
        )
        right_top = min(
            polygon_points,
            key=lambda p: (
                (p[0] - image_right_top[0]) ** 2 + (p[1] - image_right_top[1]) ** 2
            ),
        )

        # label
        text = "{} {:.2f}".format(label, score)
        if tuple(map(int, PIL.__version__.split("."))) <= (10, 0, 0):
            tw, th = draw.textsize(text, font=font)
        else:
            left, top, right, bottom = draw.textbbox((0, 0), text, font)
            tw, th = right - left, bottom - top + 4
        lx, ly = left_top
        if ly < th:
            draw.rectangle([(lx, ly), (lx + tw + 4, ly + th + 1)], fill=color)
            draw.text((lx + 2, ly - 2), text, fill=font_color, font=font)
        else:
            draw.rectangle([(lx, ly - th), (lx + tw + 4, ly + 1)], fill=color)
            draw.text((lx + 2, ly - th - 2), text, fill=font_color, font=font)

        # order
        order = box_info.get("order", None)
        if order:
            order_text = str(order)
            rx, ry = right_top
            text_position = (rx + 2, ry - font_size // 2)
            if int(img.width) - rx < font_size:
                text_position = (
                    int(rx - font_size * 1.1),
                    ry - font_size // 2,
                )
            draw.text(text_position, order_text, font=font, fill="red")

    return img


class LayoutAnalysisResult(BaseCVResult):

    def _to_img(self) -> Image.Image:
        """apply"""
        boxes = self["boxes"]
        image = Image.fromarray(self["input_img"][..., ::-1])
        ori_img_size = list(image.size)[::-1]
        if len(boxes) > 0 and "polygon_points" in boxes[0]:
            image = draw_mask(image, boxes, ori_img_size)
        else:
            image = draw_box(image, boxes)
        return {"res": image}

    def _to_str(self, *args, **kwargs):
        data = copy.deepcopy(self)
        data.pop("input_img")
        return JsonMixin._to_str(data, *args, **kwargs)

    def _to_json(self, *args, **kwargs):
        data = copy.deepcopy(self)
        data.pop("input_img")
        return JsonMixin._to_json(data, *args, **kwargs)
