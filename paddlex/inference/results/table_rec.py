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

import cv2
import numpy as np
from pathlib import Path

from .utils.mixin import HtmlMixin, XlsxMixin
from .base import BaseResult, CVResult


class TableRecResult(CVResult, HtmlMixin):
    """SaveTableResults"""

    def __init__(self, data):
        super().__init__(data)
        HtmlMixin.__init__(self)
        self._show_func_register("save_to_html")(self.save_to_html)

    def _to_html(self):
        return self["html"]

    def _to_img(self):
        image = self._img_reader.read(self["input_path"])
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


class StructureTableResult(TableRecResult, XlsxMixin):
    """StructureTableResult"""

    def __init__(self, data):
        super().__init__(data)
        XlsxMixin.__init__(self)


class TableResult(BaseResult):
    """TableResult"""

    def save_to_img(self, save_path):
        if not save_path.lower().endswith((".jpg", ".png")):
            input_path = self["input_path"]
            save_path = Path(save_path) / f"{Path(input_path).stem}"
        else:
            save_path = Path(save_path).stem
        layout_save_path = f"{save_path}_layout.jpg"
        ocr_save_path = f"{save_path}_ocr.jpg"
        table_save_path = f"{save_path}_table.jpg"
        layout_result = self["layout_result"]
        layout_result.save_to_img(layout_save_path)
        ocr_result = self["ocr_result"]
        ocr_result.save_to_img(ocr_save_path)
        for table_result in self["table_result"]:
            table_result.save_to_img(table_save_path)
