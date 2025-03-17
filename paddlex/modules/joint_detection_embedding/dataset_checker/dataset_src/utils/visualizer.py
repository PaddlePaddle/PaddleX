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

import numpy as np
from PIL import ImageDraw, ImageFont

from ......utils.fonts import PINGFANG_FONT_FILE_PATH


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


def draw_bbox(image, cls_ids, identities, xywhs):
    """
    Draws a bounding box on the image with given class ID, identity, and bbox coordinates.
    """
    # Convert image to RGB if not already
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    image_width, image_height = image.width, image.height

    # Compute font size based on image width
    font_size = int(0.015 * image_width) + 2

    # Load the font
    font = ImageFont.truetype(PINGFANG_FONT_FILE_PATH, font_size, encoding="utf-8")

    # Set line width
    line_width = int(max(image.size) * 0.002)

    # Get color mappings
    color_list = colormap(rgb=True)
    for cls_id, identity, xywh in zip(cls_ids, identities, xywhs):
        cls_id, identity = int(cls_id), int(identity)
        color_index = identity % len(color_list)
        color = tuple(color_list[color_index])
        font_color = tuple(font_colormap(color_index))

        # Denormalize bbox coordinates
        x_center = xywh[0] * image_width
        y_center = xywh[1] * image_height
        width = xywh[2] * image_width
        height = xywh[3] * image_height

        xmin = x_center - width / 2
        ymin = y_center - height / 2
        xmax = x_center + width / 2
        ymax = y_center + height / 2

        # Draw the bounding box
        draw.line(
            [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin), (xmin, ymin)],
            width=line_width,
            fill=color,
        )

        # Prepare the label text
        label = f"Class: {cls_id}"
        if identity != -1:
            label += f" ID: {identity}"
        text = label

        # Calculate text size
        try:
            # For PIL versions > 10.0.0
            left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
            tw, th = right - left, bottom - top
        except AttributeError:
            # For older PIL versions
            tw, th = draw.textsize(text, font=font)

        # Adjust label position
        if ymin - th >= 0:
            # Draw label above the bounding box
            text_origin = (xmin + 2, ymin - th - 8)
            box_coords = [(xmin, ymin - th), (xmin + tw + 4, ymin + 1)]
        else:
            # Draw label below the bounding box
            text_origin = (xmin + 2, ymin - 8)
            box_coords = [(xmin, ymin), (xmin + tw + 4, ymin + th + 1)]

        # Draw rectangle behind the text
        draw.rectangle(box_coords, fill=color)

        # Draw the label text
        draw.text(text_origin, text, fill=font_color, font=font)

    return image
