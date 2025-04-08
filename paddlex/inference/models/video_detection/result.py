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

import random

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
class DetVideoResult(BaseVideoResult):

    def _to_video(self):
        """Draw label on image"""
        video_reader = VideoReader(backend="decord")
        video = video_reader.read(self["input_path"])
        video = list(video)
        write_fps = video_reader.get_fps()
        label2color = {}
        catid2fontcolor = {}
        color_list = get_colormap(rgb=True)
        video_list = []

        for i in range(len(video)):
            image = Image.fromarray(video[i].asnumpy())
            image.size
            font_size = int(0.018 * int(image.width)) + 2
            font = ImageFont.truetype(
                PINGFANG_FONT_FILE_PATH, font_size, encoding="utf-8"
            )
            draw_thickness = int(max(image.size) * 0.002)
            draw = ImageDraw.Draw(image)
            results = self["result"][i]
            for result in results:
                bbox, score, class_name = result
                if class_name not in label2color:
                    random_index = random.randint(0, len(color_list) - 1)
                    label2color[class_name] = color_list[random_index]
                    catid2fontcolor[class_name] = self._get_font_colormap(random_index)
                color = tuple(label2color[class_name])
                font_color = tuple(catid2fontcolor[class_name])
                xmin, ymin, xmax, ymax = bbox
                rectangle = [
                    (xmin, ymin),
                    (xmin, ymax),
                    (xmax, ymax),
                    (xmax, ymin),
                    (xmin, ymin),
                ]
                draw.line(
                    rectangle,
                    width=draw_thickness,
                    fill=color,
                )
                text = "{} {:.2f}".format(class_name, score)
                if tuple(map(int, PIL.__version__.split("."))) <= (10, 0, 0):
                    tw, th = draw.textsize(text, font=font)
                else:
                    left, top, right, bottom = draw.textbbox((0, 0), text, font)
                    tw, th = right - left, bottom - top + 4
                if ymin < th:
                    draw.rectangle(
                        [(xmin, ymin), (xmin + tw + 4, ymin + th + 1)], fill=color
                    )
                    draw.text((xmin + 2, ymin - 2), text, fill=font_color, font=font)
                else:
                    draw.rectangle(
                        [(xmin, ymin - th), (xmin + tw + 4, ymin + 1)], fill=color
                    )
                    draw.text(
                        (xmin + 2, ymin - th - 2), text, fill=font_color, font=font
                    )

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
