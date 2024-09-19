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

from pathlib import Path
import json
import numpy as np
import cv2

from ...utils import logging
from ..utils.io import JsonWriter, ImageWriter, ImageReader
from .base import BaseResult


class TableRecResult(BaseResult):
    """SaveTableResults"""

    def __init__(self, data):
        super().__init__(data)
        self._img_writer.set_backend("pillow")

    def _get_res_img(self):
        image = self._img_reader.read(self["img_path"])
        bbox_res = self["bbox"]
        if len(bbox_res) > 0 and len(bbox_res[0]) == 4:
            vis_img = self.draw_rectangle(image, bbox_res)
        else:
            vis_img = self.draw_bbox(image, bbox_res)
        return vis_img

    def draw_rectangle(self, image, boxes):
        """draw_rectangle"""
        boxes = np.array(boxes)
        img_show = image.copy()
        for box in boxes.astype(int):
            x1, y1, x2, y2 = box
            cv2.rectangle(img_show, (x1, y1), (x2, y2), (255, 0, 0), 2)
        return img_show

    def draw_bbox(self, image, boxes):
        """draw_bbox"""
        for box in boxes:
            box = np.reshape(np.array(box), [-1, 1, 2]).astype(np.int64)
            image = cv2.polylines(np.array(image), [box], True, (255, 0, 0), 2)
        return image


class StructureResult(BaseResult):
    """StructureResult"""

    def __init__(self, data):
        super().__init__(data)
        self._img_writer.set_backend("pillow")

    def _get_res_img(self):
        return self._img_reader.read(self["img_path"])
