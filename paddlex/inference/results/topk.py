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
import PIL
from PIL import ImageDraw, ImageFont
import numpy as np

from ...utils.fonts import PINGFANG_FONT_FILE_PATH
from ...utils import logging
from ..utils.io import JsonWriter, ImageWriter, ImageReader
from ..utils.color_map import get_colormap


class TopkResult(dict):
    def __init__(self, data):
        super().__init__(data)
        self._json_writer = JsonWriter()
        self._img_reader = ImageReader(backend="pil")
        self._img_writer = ImageWriter(backend="pillow")

    def save_json(self, save_path, indent=4, ensure_ascii=False):
        if not save_path.endswith(".json"):
            save_path = Path(save_path) / f"{Path(self['img_path']).stem}.json"
        self._json_writer.write(save_path, self, indent=4, ensure_ascii=False)

    def save_img(self, save_path):
        if not save_path.lower().endswith((".jpg", ".png")):
            save_path = Path(save_path) / f"{Path(self['img_path']).stem}.jpg"
        labels = self.get("label_names", self["class_ids"])
        res_img = self._draw_label(self["img_path"], self["scores"], labels)
        self._img_writer.write(save_path, res_img)

    def print(self, json_format=True, indent=4, ensure_ascii=False):
        str_ = self
        if json_format:
            str_ = json.dumps(str_, indent=indent, ensure_ascii=ensure_ascii)
        logging.info(str_)

    def _draw_label(self, img_path, scores, class_ids):
        """Draw label on image"""
        label_str = f"{class_ids[0]} {scores[0]:.2f}"

        image = self._img_reader.read(img_path)
        image = image.convert("RGB")
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

        draw.rectangle([(rect_left, rect_top), (rect_right, rect_bottom)], fill=color)

        text_x = rect_left + 3
        text_y = rect_top
        draw.text((text_x, text_y), label_str, fill=font_color, font=font)
        return image

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


# class SaveClsResults(BaseComponent):

#     INPUT_KEYS = ["img_path", "cls_pred"]
#     OUTPUT_KEYS = None
#     DEAULT_INPUTS = {"img_path": "img_path", "cls_pred": "cls_pred"}
#     DEAULT_OUTPUTS = {}

#     def __init__(self, save_dir, class_ids=None):
#         super().__init__()
#         self.save_dir = save_dir
#         self.class_id_map = _parse_class_id_map(class_ids)
#         self._json_writer = ImageWriter(backend="pillow")


#     def _write_image(self, path, image):
#         """write image"""
#         if os.path.exists(path):
#             logging.warning(f"{path} already exists. Overwriting it.")
#         self._json_writer.write(path, image)
