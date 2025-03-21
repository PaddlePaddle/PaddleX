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

import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont

from ....utils.fonts import PINGFANG_FONT_FILE_PATH
from ...common.result import BaseCVResult, JsonMixin
from ...utils.color_map import get_colormap


class MLClassResult(BaseCVResult):
    def _to_str(self, *args, **kwargs):
        data = copy.deepcopy(self)
        data.pop("input_img")
        return JsonMixin._to_str(data, *args, **kwargs)

    def _to_json(self, *args, **kwargs):
        data = copy.deepcopy(self)
        data.pop("input_img")
        return JsonMixin._to_json(data, *args, **kwargs)

    def _to_img(self):
        """Draw label on image"""
        image = Image.fromarray(self["input_img"])
        label_names = self["label_names"]
        scores = self["scores"]
        image = image.convert("RGB")
        image_width, image_height = image.size
        font_size = int(image_width * 0.06)

        font = ImageFont.truetype(PINGFANG_FONT_FILE_PATH, font_size)
        text_lines = []
        row_width = 0
        row_height = 0
        row_text = "\t"
        for label_name, score in zip(label_names, scores):
            text = f"{label_name}({score})\t"
            if int(PIL.__version__.split(".")[0]) < 10:
                text_width, row_height = font.getsize(text)
            else:
                text_width, row_height = font.getbbox(text)[2:]
            if row_width + text_width <= image_width:
                row_text += text
                row_width += text_width
            else:
                text_lines.append(row_text)
                row_text = "\t" + text
                row_width = text_width
        text_lines.append(row_text)
        color_list = get_colormap(rgb=True)
        color = tuple(color_list[0])
        new_image_height = image_height + len(text_lines) * int(row_height * 1.2)
        new_image = Image.new("RGB", (image_width, new_image_height), color)
        new_image.paste(image, (0, 0))

        draw = ImageDraw.Draw(new_image)
        font_color = tuple(self._get_font_colormap(3))
        for i, text in enumerate(text_lines):
            if int(PIL.__version__.split(".")[0]) < 10:
                text_width, _ = font.getsize(text)
            else:
                text_width, _ = font.getbbox(text)[2:]
            draw.text(
                (0, image_height + i * int(row_height * 1.2)),
                text,
                fill=font_color,
                font=font,
            )
        return {"res": new_image}

    def _get_font_colormap(self, color_index):
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
