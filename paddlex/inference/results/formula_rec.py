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

import math
import random
import numpy as np
import cv2
import PIL
from PIL import Image, ImageDraw, ImageFont

from ...utils.fonts import PINGFANG_FONT_FILE_PATH
from .base import CVResult


class FormulaRecResult(CVResult):
    _HARD_FLAG = False

    def _to_str(self):
        rec_formula_str = ", ".join([str(formula) for formula in self['rec_formula']])
        return str(self).replace("\\\\","\\")


    def get_minarea_rect(self, points):
        bounding_box = cv2.minAreaRect(points)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_a, index_b, index_c, index_d = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_a = 0
            index_d = 1
        else:
            index_a = 1
            index_d = 0
        if points[3][1] > points[2][1]:
            index_b = 2
            index_c = 3
        else:
            index_b = 3
            index_c = 2

        box = np.array(
            [points[index_a], points[index_b], points[index_c], points[index_d]]
        ).astype(np.int32)

        return box
    
   
    def _to_img(
        self,
    ):
        """draw ocr result"""
        # TODO(gaotingquan): mv to postprocess
        drop_score = 0.5

        boxes = self["dt_polys"]
        formula = self["rec_formula"]
        image = self._img_reader.read(self["input_path"])
        if self._HARD_FLAG:
            image_np = np.array(image)
            image = Image.fromarray(image_np[:, :, ::-1])
        h, w = image.height, image.width
        img_left = image.copy()
        random.seed(0)
        draw_left = ImageDraw.Draw(img_left)
        if formula is None or len(formula) != len(boxes):
            formula = [None] * len(boxes)
        for idx, (box, txt) in enumerate(zip(boxes, formula)):
            try:
                color = (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                )
                box = np.array(box)

                if len(box) > 4:
                    pts = [(x, y) for x, y in box.tolist()]
                    draw_left.polygon(pts, outline=color, width=8)
                    box = self.get_minarea_rect(box)
                    height = int(0.5 * (max(box[:, 1]) - min(box[:, 1])))
                    box[:2, 1] = np.mean(box[:, 1])
                    box[2:, 1] = np.mean(box[:, 1]) + min(20, height)
                draw_left.polygon(box, fill=color)
            except:
                continue

        img_left = Image.blend(image, img_left, 0.5)
        img_show = Image.new("RGB", (w, h), (255, 255, 255))
        img_show.paste(img_left, (0, 0, w, h))
        return img_show


def create_font(txt, sz, font_path):
    """create font"""
    font_size = int(sz[1] * 0.8)
    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
    if int(PIL.__version__.split(".")[0]) < 10:
        length = font.getsize(txt)[0]
    else:
        length = font.getlength(txt)

    if length > sz[0]:
        font_size = int(font_size * sz[0] / length)
        font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
    return font
