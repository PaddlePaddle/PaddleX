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

import os
import json
from pathlib import Path
import numpy as np
import PIL
from PIL import ImageDraw, ImageFont, Image

from .keys import ClsKeys as K
from ...base import BaseTransform
from ...base.predictor.io import ImageWriter, ImageReader
from ....utils.fonts import PINGFANG_FONT_FILE_PATH
from ....utils import logging

__all__ = [
    "Topk",
    "NormalizeFeatures",
    "PrintResult",
    "SaveClsResults",
    "MultiLabelThreshOutput",
]


def _parse_class_id_map(class_ids):
    """parse class id to label map file"""
    if class_ids is None:
        return None
    class_id_map = {id: str(lb) for id, lb in enumerate(class_ids)}
    return class_id_map


class Topk(BaseTransform):
    """Topk Transform"""

    def __init__(self, topk, class_ids=None):
        super().__init__()
        assert isinstance(topk, (int,))
        self.topk = topk
        self.class_id_map = _parse_class_id_map(class_ids)

    def apply(self, data):
        """apply"""
        x = data[K.CLS_PRED]
        class_id_map = self.class_id_map
        y = []
        index = x.argsort(axis=0)[-self.topk :][::-1].astype("int32")
        clas_id_list = []
        score_list = []
        label_name_list = []
        for i in index:
            clas_id_list.append(i.item())
            score_list.append(x[i].item())
            if class_id_map is not None:
                label_name_list.append(class_id_map[i.item()])
        result = {
            "class_ids": clas_id_list,
            "scores": np.around(score_list, decimals=5).tolist(),
        }
        if label_name_list is not None:
            result["label_names"] = label_name_list
        y.append(result)
        data[K.CLS_RESULT] = y
        return data

    @classmethod
    def get_input_keys(cls):
        """get input keys"""
        return [K.IM_PATH, K.CLS_PRED]

    @classmethod
    def get_output_keys(cls):
        """get output keys"""
        return [K.CLS_RESULT]


class NormalizeFeatures(BaseTransform):
    """Normalize Features Transform"""

    def apply(self, data):
        """apply"""
        x = data[K.CLS_PRED]
        feas_norm = np.sqrt(np.sum(np.square(x), axis=0, keepdims=True))
        x = np.divide(x, feas_norm)
        data[K.CLS_RESULT] = x
        return data

    @classmethod
    def get_input_keys(cls):
        """get input keys"""
        return [K.IM_PATH, K.CLS_PRED]

    @classmethod
    def get_output_keys(cls):
        """get output keys"""
        return [K.CLS_RESULT]


class PrintResult(BaseTransform):
    """Print Result Transform"""

    def apply(self, data):
        """apply"""
        logging.info("The prediction result is:")
        logging.info(data[K.CLS_RESULT])
        return data

    @classmethod
    def get_input_keys(cls):
        """get input keys"""
        return [K.CLS_RESULT]

    @classmethod
    def get_output_keys(cls):
        """get output keys"""
        return []


class SaveClsResults(BaseTransform):
    def __init__(self, save_dir, class_ids=None):
        super().__init__()
        self.save_dir = save_dir
        self.class_id_map = _parse_class_id_map(class_ids)
        self._writer = ImageWriter(backend="pillow")

    def _get_colormap(self, rgb=False):
        """
        Get colormap
        """
        color_list = np.array(
            [
                0xFF,
                0x00,
                0x00,
                0xCC,
                0xFF,
                0x00,
                0x00,
                0xFF,
                0x66,
                0x00,
                0x66,
                0xFF,
                0xCC,
                0x00,
                0xFF,
                0xFF,
                0x4D,
                0x00,
                0x80,
                0xFF,
                0x00,
                0x00,
                0xFF,
                0xB2,
                0x00,
                0x1A,
                0xFF,
                0xFF,
                0x00,
                0xE5,
                0xFF,
                0x99,
                0x00,
                0x33,
                0xFF,
                0x00,
                0x00,
                0xFF,
                0xFF,
                0x33,
                0x00,
                0xFF,
                0xFF,
                0x00,
                0x99,
                0xFF,
                0xE5,
                0x00,
                0x00,
                0xFF,
                0x1A,
                0x00,
                0xB2,
                0xFF,
                0x80,
                0x00,
                0xFF,
                0xFF,
                0x00,
                0x4D,
            ]
        ).astype(np.float32)
        color_list = color_list.reshape((-1, 3))
        if not rgb:
            color_list = color_list[:, ::-1]
        return color_list.astype("int32")

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

    def apply(self, data):
        """Draw label on image"""
        ori_path = data[K.IM_PATH]
        pred = data[K.CLS_PRED]
        index = pred.argsort(axis=0)[-1].astype("int32")
        score = pred[index].item()
        label = self.class_id_map[int(index)] if self.class_id_map else ""
        label_str = f"{label} {score:.2f}"
        file_name = os.path.basename(ori_path)
        save_path = os.path.join(self.save_dir, file_name)

        image = ImageReader(backend="pil").read(ori_path)
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
        color_list = self._get_colormap(rgb=True)
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
        self._write_image(save_path, image)

        return data

    def _write_image(self, path, image):
        """write image"""
        if os.path.exists(path):
            logging.warning(f"{path} already exists. Overwriting it.")
        self._writer.write(path, image)

    @classmethod
    def get_input_keys(cls):
        """get input keys"""
        return [K.IM_PATH, K.CLS_PRED]

    @classmethod
    def get_output_keys(cls):
        """get output keys"""
        return []


class MultiLabelThreshOutput(BaseTransform):
    def __init__(self, threshold=0.5, class_ids=None, delimiter=None):
        super().__init__()
        assert isinstance(threshold, (float,))
        self.threshold = threshold
        self.delimiter = delimiter if delimiter is not None else " "
        self.class_id_map = _parse_class_id_map(class_ids)

    def apply(self, data):
        """apply"""
        y = []
        x = data[K.CLS_PRED]
        pred_index = np.where(x >= self.threshold)[0].astype("int32")
        index = pred_index[np.argsort(x[pred_index])][::-1]
        clas_id_list = []
        score_list = []
        label_name_list = []
        for i in index:
            clas_id_list.append(i.item())
            score_list.append(x[i].item())
            if self.class_id_map is not None:
                label_name_list.append(self.class_id_map[i.item()])
        result = {
            "class_ids": clas_id_list,
            "scores": np.around(score_list, decimals=5).tolist(),
            "label_names": label_name_list,
        }
        y.append(result)
        data[K.CLS_RESULT] = y
        return data

    @classmethod
    def get_input_keys(cls):
        """get input keys"""
        return [K.IM_PATH, K.CLS_PRED]

    @classmethod
    def get_output_keys(cls):
        """get output keys"""
        return [K.CLS_RESULT]


class SaveMLClsResults(SaveClsResults, BaseTransform):
    def __init__(self, save_dir, class_ids=None):
        super().__init__(save_dir=save_dir)
        self.save_dir = save_dir
        self.class_id_map = _parse_class_id_map(class_ids)
        self._writer = ImageWriter(backend="pillow")

    def apply(self, data):
        """Draw label on image"""
        ori_path = data[K.IM_PATH]
        results = data[K.CLS_RESULT]
        scores = results[0]["scores"]
        label_names = results[0]["label_names"]
        file_name = os.path.basename(ori_path)
        save_path = os.path.join(self.save_dir, file_name)
        image = ImageReader(backend="pil").read(ori_path)
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
            text_width, row_height = font.getsize(text)
            if row_width + text_width <= image_width:
                row_text += text
                row_width += text_width
            else:
                text_lines.append(row_text)
                row_text = "\t" + text
                row_width = text_width
        text_lines.append(row_text)
        color_list = self._get_colormap(rgb=True)
        color = tuple(color_list[0])
        new_image_height = image_height + len(text_lines) * int(row_height * 1.2)
        new_image = Image.new("RGB", (image_width, new_image_height), color)
        new_image.paste(image, (0, 0))

        draw = ImageDraw.Draw(new_image)
        font_color = tuple(self._get_font_colormap(3))
        for i, text in enumerate(text_lines):
            text_width, _ = font.getsize(text)
            draw.text(
                (0, image_height + i * int(row_height * 1.2)),
                text,
                fill=font_color,
                font=font,
            )
        self._write_image(save_path, new_image)
        return data
