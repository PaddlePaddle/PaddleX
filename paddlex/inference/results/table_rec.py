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

from .base import BaseResult
from ...utils import logging
from ..utils.io import HtmlWriter, XlsxWriter


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


class StructureTableResult(TableRecResult):
    """StructureTableResult"""

    def __init__(self, data):
        """__init__"""
        super().__init__(data)
        self._img_writer.set_backend("pillow")
        self._html_writer = HtmlWriter()
        self._xlsx_writer = XlsxWriter()

    def save_to_html(self, save_path):
        """save_to_html"""
        img_idx = self["img_idx"]
        if not save_path.endswith(".html"):
            if img_idx > 0:
                save_path = (
                    Path(save_path) / f"{Path(self['img_path']).stem}_{img_idx}.html"
                )
            else:
                save_path = Path(save_path) / f"{Path(self['img_path']).stem}.html"
        elif img_idx > 0:
            save_path = Path(save_path).stem / f"_{img_idx}.html"
        self._html_writer.write(save_path.as_posix(), self["html"])
        logging.info(f"The result has been saved in {save_path}.")

    def save_to_excel(self, save_path):
        """save_to_excel"""
        img_idx = self["img_idx"]
        if not save_path.endswith(".xlsx"):
            if img_idx > 0:
                save_path = (
                    Path(save_path) / f"{Path(self['img_path']).stem}_{img_idx}.xlsx"
                )
            else:
                save_path = Path(save_path) / f"{Path(self['img_path']).stem}.xlsx"
        elif img_idx > 0:
            save_path = Path(save_path).stem / f"_{img_idx}.xlsx"
        self._xlsx_writer.write(save_path.as_posix(), self["html"])
        logging.info(f"The result has been saved in {save_path}.")

    def save_to_img(self, save_path):
        img_idx = self["img_idx"]
        if not save_path.endswith((".jpg", ".png")):
            if img_idx > 0:
                save_path = (
                    Path(save_path) / f"{Path(self['img_path']).stem}_{img_idx}.jpg"
                )
            else:
                save_path = Path(save_path) / f"{Path(self['img_path']).stem}.jpg"
        elif img_idx > 0:
            save_path = Path(save_path).stem / f"_{img_idx}.jpg"
        else:
            save_path = Path(save_path)
        res_img = self._get_res_img()
        if res_img is not None:
            self._img_writer.write(save_path.as_posix(), res_img)
            logging.info(f"The result has been saved in {save_path}.")


class TableResult(BaseResult):
    """TableResult"""

    def __init__(self, data):
        """__init__"""
        super().__init__(data)

    def save_to_img(self, save_path):
        if not save_path.lower().endswith((".jpg", ".png")):
            img_path = self["img_path"]
            save_path = Path(save_path) / f"{Path(img_path).stem}"
        else:
            save_path = Path(save_path).stem
        layout_save_path = f"{save_path}_layout.jpg"
        ocr_save_path = f"{save_path}_ocr.jpg"
        table_save_path = f"{save_path}_table.jpg"
        layout_result = self["layout_result"]
        layout_result.save_to_img(layout_save_path)
        ocr_result = self["ocr_result"]
        ocr_result.save_to_img(ocr_save_path)
        for batch_table_result in self["table_result"]:
            for table_result in batch_table_result:
                table_result.save_to_img(table_save_path)
