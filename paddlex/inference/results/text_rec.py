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


class TextRecResult(dict):
    def __init__(self, data):
        super().__init__(data)
        self._json_writer = JsonWriter()
        self._img_reader = ImageReader(backend="opencv")
        self._img_writer = ImageWriter(backend="opencv")

    def save_json(self, save_path, indent=4, ensure_ascii=False):
        if not save_path.endswith(".json"):
            save_path = Path(save_path) / f"{Path(self['img_path']).stem}.json"
        self._json_writer.write(save_path, self, indent=4, ensure_ascii=False)

    def save_img(self, save_path):
        raise Exception()

    def print(self, json_format=True, indent=4, ensure_ascii=False):
        str_ = self
        if json_format:
            str_ = json.dumps(str_, indent=indent, ensure_ascii=ensure_ascii)
        logging.info(str_)
