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

import random
import numpy as np
from PIL import Image, ImageDraw

from .base import CVResult


class FormulaRecResult(CVResult):
    _HARD_FLAG = False

    def _to_str(self):
        rec_formula_str = ", ".join([str(formula) for formula in self["rec_formula"]])
        return str(self).replace("\\\\", "\\")

    def _to_img(
        self,
    ):
        """draw formula result"""
        boxes = self["dt_polys"]
        formula = self["rec_formula"]
        image = self._img_reader.read(self["input_path"])
        if self._HARD_FLAG:
            image_np = np.array(image)
            image = Image.fromarray(image_np[:, :, ::-1])
        img = image.copy()
        random.seed(0)
        draw_img = ImageDraw.Draw(img)
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
                pts = [(x, y) for x, y in box.tolist()]
                draw_img.polygon(pts, outline=color, width=8)
                draw_img.polygon(box, fill=color)
            except:
                continue

        img = Image.blend(image, img, 0.5)
        return img
