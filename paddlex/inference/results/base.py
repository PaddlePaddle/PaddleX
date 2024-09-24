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

from abc import abstractmethod
from pathlib import Path
import numpy as np
import json

from ...utils import logging
import numpy as np
from ..utils.io import JsonWriter, ImageReader, ImageWriter


def format_data(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return [format_data(item) for item in obj.tolist()]
    elif isinstance(obj, dict):
        return type(obj)({k: format_data(v) for k, v in obj.items()})
    elif isinstance(obj, list):
        return [format_data(i) for i in obj]
    else:
        return obj


class BaseResult(dict):
    def __init__(self, data):
        super().__init__(format_data(data))
        self._check_res()
        self._json_writer = JsonWriter()
        self._img_reader = ImageReader(backend="opencv")
        self._img_writer = ImageWriter(backend="opencv")

    def save_to_json(self, save_path, indent=4, ensure_ascii=False):
        if not save_path.endswith(".json"):
            save_path = Path(save_path) / f"{Path(self['img_path']).stem}.json"
        self._json_writer.write(save_path, self, indent=4, ensure_ascii=False)

    def save_to_img(self, save_path):
        if not save_path.lower().endswith((".jpg", ".png")):
            save_path = Path(save_path) / f"{Path(self['img_path']).stem}.jpg"
        else:
            save_path = Path(save_path)
        res_img = self._get_res_img()
        if res_img is not None:
            self._img_writer.write(save_path.as_posix(), res_img)
            logging.info(f"The result has been saved in {save_path}.")

    def print(self, json_format=True, indent=4, ensure_ascii=False):
        str_ = self
        if json_format:
            str_ = json.dumps(str_, indent=indent, ensure_ascii=ensure_ascii)
        logging.info(str_)

    def _check_res(self):
        pass

    @abstractmethod
    def _get_res_img(self):
        raise NotImplementedError
