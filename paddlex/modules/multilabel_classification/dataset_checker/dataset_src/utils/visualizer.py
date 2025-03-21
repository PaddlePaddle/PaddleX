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
from PIL import Image, ImageDraw, ImageFont

from ......utils.fonts import PINGFANG_FONT_FILE_PATH


def colormap(rgb=False):
    """
    Get colormap
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
    Get font colormap
    """
    dark = np.array([0x14, 0x0E, 0x35])
    light = np.array([0xFF, 0xFF, 0xFF])
    light_indexs = [0, 3, 4, 8, 9, 13, 14, 18, 19]
    if color_index in light_indexs:
        return light.astype("int32")
    else:
        return dark.astype("int32")


def draw_multi_label(image, label, label_map_dict):
    labels = label.split(",")
    label_names = [
        label_map_dict[i] for i, label in enumerate(labels) if int(label) == 1
    ]
    image = image.convert("RGB")
    image_width, image_height = image.size
    font_size = int(image_width * 0.06)

    font = ImageFont.truetype(PINGFANG_FONT_FILE_PATH, font_size)
    text_lines = []
    row_width = 0
    row_height = 0
    row_text = "\t"
    for label_name in label_names:
        text = f"{label_name}\t"
        x1, y1, x2, y2 = font.getbbox(text)
        text_width, row_height = x2 - x1, y2 - y1
        if row_width + text_width <= image_width:
            row_text += text
            row_width += text_width
        else:
            text_lines.append(row_text)
            row_text = "\t" + text
            row_width = text_width
    text_lines.append(row_text)
    color_list = colormap(rgb=True)
    color = tuple(color_list[0])
    new_image_height = image_height + len(text_lines) * int(row_height * 1.8)
    new_image = Image.new("RGB", (image_width, new_image_height), color)
    new_image.paste(image, (0, 0))

    draw = ImageDraw.Draw(new_image)
    font_color = tuple(font_colormap(3))
    for i, text in enumerate(text_lines):
        draw.text(
            (0, image_height + i * int(row_height * 1.2)),
            text,
            fill=font_color,
            font=font,
        )
    return new_image
