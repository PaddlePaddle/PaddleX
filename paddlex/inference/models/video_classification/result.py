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
from PIL import Image, ImageDraw, ImageFont

from ....utils.deps import class_requires_deps, is_dep_available
from ....utils.fonts import PINGFANG_FONT_FILE_PATH
from ...common.result import BaseVideoResult
from ...utils.color_map import get_colormap
from ...utils.io import VideoReader

if is_dep_available("opencv-contrib-python"):
    import cv2


@class_requires_deps("opencv-contrib-python")
class TopkVideoResult(BaseVideoResult):

    def _to_video(self):
        """Draw label on image"""
        labels = self.get("label_names", self["class_ids"])
        label_str = f"{labels[0]} {self['scores'][0]:.2f}"
        video_reader = VideoReader(backend="decord")
        video = video_reader.read(self["input_path"])
        video = list(video)
        write_fps = video_reader.get_fps()

        video_list = []
        for i in range(len(video)):
            image = Image.fromarray(video[i].asnumpy())
            image_size = image.size
            draw = ImageDraw.Draw(image)
            min_font_size = int(image_size[0] * 0.02)
            max_font_size = int(image_size[0] * 0.05)
            for font_size in range(max_font_size, min_font_size - 1, -1):
                font = ImageFont.truetype(
                    PINGFANG_FONT_FILE_PATH, font_size, encoding="utf-8"
                )
                if tuple(map(int, PIL.__version__.split("."))) <= (10, 0, 0):
                    text_width_tmp, text_height_tmp = draw.textsize(label_str, font)
                else:
                    left, top, right, bottom = draw.textbbox((0, 0), label_str, font)
                    text_width_tmp, text_height_tmp = right - left, bottom - top
                if text_width_tmp <= image_size[0]:
                    break
                else:
                    font = ImageFont.truetype(PINGFANG_FONT_FILE_PATH, min_font_size)
            color_list = get_colormap(rgb=True)
            color = tuple(color_list[0])
            font_color = tuple(self._get_font_colormap(3))
            if tuple(map(int, PIL.__version__.split("."))) <= (10, 0, 0):
                text_width, text_height = draw.textsize(label_str, font)
            else:
                left, top, right, bottom = draw.textbbox((0, 0), label_str, font)
                text_width, text_height = right - left, bottom - top

            rect_left = 3
            rect_top = 3
            rect_right = rect_left + text_width + 3
            rect_bottom = rect_top + text_height + 6

            draw.rectangle(
                [(rect_left, rect_top), (rect_right, rect_bottom)], fill=color
            )

            text_x = rect_left + 3
            text_y = rect_top
            draw.text((text_x, text_y), label_str, fill=font_color, font=font)
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            video_list.append(image)
        return {"res": (np.array(video_list), write_fps)}

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
