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

import numpy as np
import PIL
from PIL import ImageDraw, ImageFont

from ......utils import logging
from ......utils.deps import function_requires_deps, is_dep_available
from ......utils.fonts import PINGFANG_FONT_FILE_PATH

if is_dep_available("pycocotools"):
    from pycocotools.coco import COCO


def colormap(rgb=False):
    """
    Get colormap

    The code of this function is copied from https://github.com/facebookresearch/Detectron/blob/main/detectron/\
utils/colormap.py
    """
    color_list = np.array(
        [
            0xFF,
            0x00,
            0x00,
            0xCC,
            0xFF,
            0x00,
            0x00,
            0xFF,
            0x66,
            0x00,
            0x66,
            0xFF,
            0xCC,
            0x00,
            0xFF,
            0xFF,
            0x4D,
            0x00,
            0x80,
            0xFF,
            0x00,
            0x00,
            0xFF,
            0xB2,
            0x00,
            0x1A,
            0xFF,
            0xFF,
            0x00,
            0xE5,
            0xFF,
            0x99,
            0x00,
            0x33,
            0xFF,
            0x00,
            0x00,
            0xFF,
            0xFF,
            0x33,
            0x00,
            0xFF,
            0xFF,
            0x00,
            0x99,
            0xFF,
            0xE5,
            0x00,
            0x00,
            0xFF,
            0x1A,
            0x00,
            0xB2,
            0xFF,
            0x80,
            0x00,
            0xFF,
            0xFF,
            0x00,
            0x4D,
        ]
    ).astype(np.float32)
    color_list = color_list.reshape((-1, 3))
    if not rgb:
        color_list = color_list[:, ::-1]
    return color_list.astype("int32")


def font_colormap(color_index):
    """
    Get font color according to the index of colormap
    """
    dark = np.array([0x14, 0x0E, 0x35])
    light = np.array([0xFF, 0xFF, 0xFF])
    light_indexs = [0, 3, 4, 8, 9, 13, 14, 18, 19]
    if color_index in light_indexs:
        return light.astype("int32")
    else:
        return dark.astype("int32")


@function_requires_deps("pycocotools")
def draw_bbox(image, coco_info: "COCO", img_id):
    """
    Draw bbox on image
    """
    try:
        image_info = coco_info.loadImgs(img_id)[0]
        font_size = int(0.024 * int(image_info["width"])) + 2
    except:
        font_size = 12
    font = ImageFont.truetype(PINGFANG_FONT_FILE_PATH, font_size, encoding="utf-8")

    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    image_size = image.size
    width = int(max(image_size) * 0.005)

    catid2color = {}
    catid2fontcolor = {}
    catid_num_dict = {}
    color_list = colormap(rgb=True)
    annotations = coco_info.loadAnns(coco_info.getAnnIds(imgIds=img_id))

    for ann in annotations:
        catid = ann["category_id"]
        catid_num_dict[catid] = catid_num_dict.get(catid, 0) + 1
    for i, (catid, _) in enumerate(
        sorted(catid_num_dict.items(), key=lambda x: x[1], reverse=True)
    ):
        if catid not in catid2color:
            color_index = i % len(color_list)
            catid2color[catid] = color_list[color_index]
            catid2fontcolor[catid] = font_colormap(color_index)
    for ann in annotations:
        catid, bbox = ann["category_id"], ann["bbox"]
        color = tuple(catid2color[catid])
        font_color = tuple(catid2fontcolor[catid])

        if len(bbox) == 4:
            # draw bbox
            xmin, ymin, w, h = bbox
            xmax = xmin + w
            ymax = ymin + h
            draw.line(
                [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin), (xmin, ymin)],
                width=width,
                fill=color,
            )
        elif len(bbox) == 8:
            x1, y1, x2, y2, x3, y3, x4, y4 = bbox
            draw.line(
                [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x1, y1)],
                width=width,
                fill=color,
            )
            xmin = min(x1, x2, x3, x4)
            ymin = min(y1, y2, y3, y4)
        else:
            logging.info("Error: The shape of bbox must be [M, 4] or [M, 8]!")

        # draw label
        label = coco_info.loadCats(catid)[0]["name"]
        text = "{}".format(label)
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

    return image
