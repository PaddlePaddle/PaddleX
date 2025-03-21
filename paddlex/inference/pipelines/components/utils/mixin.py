# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import json
from abc import abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .....utils import logging
from ....utils.io import (
    CSVWriter,
    HtmlWriter,
    ImageWriter,
    JsonWriter,
    TextWriter,
    XlsxWriter,
)

#### [TODO] need tingquan to add explanatory notes


def _save_list_data(save_func, save_path, data, *args, **kwargs):
    save_path = Path(save_path)
    if data is None:
        return
    if isinstance(data, list):
        for idx, single in enumerate(data):
            save_func(
                (
                    save_path.parent / f"{save_path.stem}_{idx}{save_path.suffix}"
                ).as_posix(),
                single,
                *args,
                **kwargs,
            )
    save_func(save_path.as_posix(), data, *args, **kwargs)
    logging.info(f"The result has been saved in {save_path}.")


class StrMixin:
    @property
    def str(self):
        return self._to_str()

    def _to_str(self, data, json_format=False, indent=4, ensure_ascii=False):
        if json_format:
            return json.dumps(data.json, indent=indent, ensure_ascii=ensure_ascii)
        else:
            return str(data)

    def print(self, json_format=False, indent=4, ensure_ascii=False):
        str_ = self._to_str(
            self, json_format=json_format, indent=indent, ensure_ascii=ensure_ascii
        )
        logging.info(str_)


class JsonMixin:
    def __init__(self):
        self._json_writer = JsonWriter()
        self._show_funcs.append(self.save_to_json)

    def _to_json(self):
        def _format_data(obj):
            if isinstance(obj, np.float32):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return [_format_data(item) for item in obj.tolist()]
            elif isinstance(obj, pd.DataFrame):
                return obj.to_json(orient="records", force_ascii=False)
            elif isinstance(obj, Path):
                return obj.as_posix()
            elif isinstance(obj, dict):
                return type(obj)({k: _format_data(v) for k, v in obj.items()})
            elif isinstance(obj, (list, tuple)):
                return [_format_data(i) for i in obj]
            else:
                return obj

        return _format_data(self)

    @property
    def json(self):
        return self._to_json()

    def save_to_json(self, save_path, indent=4, ensure_ascii=False, *args, **kwargs):
        if not str(save_path).endswith(".json"):
            save_path = Path(save_path) / f"{Path(self['input_path']).stem}.json"
        _save_list_data(
            self._json_writer.write,
            save_path,
            self.json,
            indent=indent,
            ensure_ascii=ensure_ascii,
            *args,
            **kwargs,
        )


class Base64Mixin:
    def __init__(self, *args, **kwargs):
        self._base64_writer = TextWriter(*args, **kwargs)
        self._show_funcs.append(self.save_to_base64)

    @abstractmethod
    def _to_base64(self):
        raise NotImplementedError

    @property
    def base64(self):
        return self._to_base64()

    def save_to_base64(self, save_path, *args, **kwargs):
        if not str(save_path).lower().endswith((".b64")):
            fp = Path(self["input_path"])
            save_path = Path(save_path) / f"{fp.stem}{fp.suffix}"
        _save_list_data(
            self._base64_writer.write, save_path, self.base64, *args, **kwargs
        )


class ImgMixin:
    def __init__(self, backend="pillow", *args, **kwargs):
        self._img_writer = ImageWriter(backend=backend, *args, **kwargs)
        self._show_funcs.append(self.save_to_img)

    @abstractmethod
    def _to_img(self):
        raise NotImplementedError

    @property
    def img(self):
        image = self._to_img()
        # The img must be a PIL.Image obj
        if isinstance(image, np.ndarray):
            return Image.fromarray(image)
        return image

    def save_to_img(self, save_path, *args, **kwargs):
        if not str(save_path).lower().endswith((".jpg", ".png")):
            fp = Path(self["input_path"])
            save_path = Path(save_path) / f"{fp.stem}{fp.suffix}"
        _save_list_data(self._img_writer.write, save_path, self.img, *args, **kwargs)


class CSVMixin:
    def __init__(self, backend="pandas", *args, **kwargs):
        self._csv_writer = CSVWriter(backend=backend, *args, **kwargs)
        self._show_funcs.append(self.save_to_csv)

    @abstractmethod
    def _to_csv(self):
        raise NotImplementedError

    def save_to_csv(self, save_path, *args, **kwargs):
        if not str(save_path).endswith(".csv"):
            save_path = Path(save_path) / f"{Path(self['input_path']).stem}.csv"
        _save_list_data(
            self._csv_writer.write, save_path, self._to_csv(), *args, **kwargs
        )


class HtmlMixin:
    def __init__(self, *args, **kwargs):
        self._html_writer = HtmlWriter(*args, **kwargs)
        self._show_funcs.append(self.save_to_html)

    @property
    def html(self):
        return self._to_html()

    def _to_html(self):
        return self["html"]

    def save_to_html(self, save_path, *args, **kwargs):
        if not str(save_path).endswith(".html"):
            save_path = Path(save_path) / f"{Path(self['input_path']).stem}.html"
        _save_list_data(self._html_writer.write, save_path, self.html, *args, **kwargs)


class XlsxMixin:
    def __init__(self, *args, **kwargs):
        self._xlsx_writer = XlsxWriter(*args, **kwargs)
        self._show_funcs.append(self.save_to_xlsx)

    def _to_xlsx(self):
        return self["html"]

    def save_to_xlsx(self, save_path, *args, **kwargs):
        if not str(save_path).endswith(".xlsx"):
            save_path = Path(save_path) / f"{Path(self['input_path']).stem}.xlsx"
        _save_list_data(self._xlsx_writer.write, save_path, self.html, *args, **kwargs)
