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
import cv2

from ..utils.io import ImageReader
from .base import BaseResult


class TextDetResult(BaseResult):

    def _get_res_img(self):
        """draw rectangle"""
        boxes = np.array(self["dt_polys"])
        img = self._img_reader.read(self["img_path"])
        res_img = img.copy()
        for box in boxes.astype(int):
            box = np.reshape(np.array(box), [-1, 1, 2]).astype(np.int64)
            cv2.polylines(res_img, [box], True, (0, 0, 255), 2)
        return res_img
