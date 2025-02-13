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
import enum
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pandas as pd
import yaml
from .tablepyxl import document_to_xl


__all__ = [
    "WriterType",
    "ImageWriter",
    "TextWriter",
    "JsonWriter",
    "CSVWriter",
    "HtmlWriter",
    "XlsxWriter",
    "YAMLWriter",
    "VideoWriter",
    "MarkdownWriter",
]


class WriterType(enum.Enum):
    """WriterType"""

    IMAGE = 1
    VIDEO = 2
    TEXT = 3
    JSON = 4
    HTML = 5
    XLSX = 6
    CSV = 7
    YAML = 8


class _BaseWriter(object):
    """_BaseWriter"""

    def __init__(self, backend, **bk_args):
        super().__init__()
        if len(bk_args) == 0:
            bk_args = self.get_default_backend_args()
        self.bk_type = backend
        self.bk_args = bk_args
        self._backend = self.get_backend()

    def write(self, out_path, obj):
        """write"""
        raise NotImplementedError

    def get_backend(self, bk_args=None):
        """get backend"""
        if bk_args is None:
            bk_args = self.bk_args
        return self._init_backend(self.bk_type, bk_args)

    def set_backend(self, backend, **bk_args):
        self.bk_type = backend
        self.bk_args = bk_args
        self._backend = self.get_backend()

    def _init_backend(self, bk_type, bk_args):
        """init backend"""
        raise NotImplementedError

    def get_type(self):
        """get type"""
        raise NotImplementedError

    def get_default_backend_args(self):
        """get default backend arguments"""
        return {}


class ImageWriter(_BaseWriter):
    """ImageWriter"""

    def __init__(self, backend="opencv", **bk_args):
        super().__init__(backend=backend, **bk_args)

    def write(self, out_path, obj):
        """write"""
        return self._backend.write_obj(str(out_path), obj)

    def _init_backend(self, bk_type, bk_args):
        """init backend"""
        if bk_type == "opencv":
            return OpenCVImageWriterBackend(**bk_args)
        elif bk_type == "pil" or bk_type == "pillow":
            return PILImageWriterBackend(**bk_args)
        else:
            raise ValueError("Unsupported backend type")

    def get_type(self):
        """get type"""
        return WriterType.IMAGE


class VideoWriter(_BaseWriter):
    """VideoWriter"""

    def __init__(self, backend="opencv", **bk_args):
        super().__init__(backend=backend, **bk_args)

    def write(self, out_path, obj):
        """write"""
        return self._backend.write_obj(str(out_path), obj)

    def _init_backend(self, bk_type, bk_args):
        """init backend"""
        if bk_type == "opencv":
            return OpenCVVideoWriterBackend(**bk_args)
        else:
            raise ValueError("Unsupported backend type")

    def get_type(self):
        """get type"""
        return WriterType.VIDEO


class TextWriter(_BaseWriter):
    """TextWriter"""

    def __init__(self, backend="python", **bk_args):
        super().__init__(backend=backend, **bk_args)

    def write(self, out_path, obj):
        """write"""
        return self._backend.write_obj(str(out_path), obj)

    def _init_backend(self, bk_type, bk_args):
        """init backend"""
        if bk_type == "python":
            return TextWriterBackend(**bk_args)
        else:
            raise ValueError("Unsupported backend type")

    def get_type(self):
        """get type"""
        return WriterType.TEXT


class JsonWriter(_BaseWriter):
    def __init__(self, backend="json", **bk_args):
        super().__init__(backend=backend, **bk_args)

    def write(self, out_path, obj, **bk_args):
        return self._backend.write_obj(str(out_path), obj, **bk_args)

    def _init_backend(self, bk_type, bk_args):
        if bk_type == "json":
            return JsonWriterBackend(**bk_args)
        elif bk_type == "ujson":
            return UJsonWriterBackend(**bk_args)
        else:
            raise ValueError("Unsupported backend type")

    def get_type(self):
        """get type"""
        return WriterType.JSON


class HtmlWriter(_BaseWriter):
    def __init__(self, backend="html", **bk_args):
        super().__init__(backend=backend, **bk_args)

    def write(self, out_path, obj, **bk_args):
        return self._backend.write_obj(str(out_path), obj, **bk_args)

    def _init_backend(self, bk_type, bk_args):
        if bk_type == "html":
            return HtmlWriterBackend(**bk_args)
        else:
            raise ValueError("Unsupported backend type")

    def get_type(self):
        """get type"""
        return WriterType.HTML


class XlsxWriter(_BaseWriter):
    def __init__(self, backend="xlsx", **bk_args):
        super().__init__(backend=backend, **bk_args)

    def write(self, out_path, obj, **bk_args):
        return self._backend.write_obj(str(out_path), obj, **bk_args)

    def _init_backend(self, bk_type, bk_args):
        if bk_type == "xlsx":
            return XlsxWriterBackend(**bk_args)
        else:
            raise ValueError("Unsupported backend type")

    def get_type(self):
        """get type"""
        return WriterType.XLSX


class YAMLWriter(_BaseWriter):
    def __init__(self, backend="PyYAML", **bk_args):
        super().__init__(backend=backend, **bk_args)

    def write(self, out_path, obj, **bk_args):
        return self._backend.write_obj(str(out_path), obj, **bk_args)

    def _init_backend(self, bk_type, bk_args):
        if bk_type == "PyYAML":
            return YAMLWriterBackend(**bk_args)
        else:
            raise ValueError("Unsupported backend type")

    def get_type(self):
        """get type"""
        return WriterType.YAML


class MarkdownWriter(_BaseWriter):
    """MarkdownWriter"""

    def __init__(self, backend="markdown", **bk_args):
        super().__init__(backend=backend, **bk_args)

    def write(self, out_path, obj):
        """write"""
        return self._backend.write_obj(str(out_path), obj)

    def _init_backend(self, bk_type, bk_args):
        """init backend"""
        if bk_type == "markdown":
            return MarkdownWriterBackend(**bk_args)
        else:
            raise ValueError("Unsupported backend type")

    def get_type(self):
        """get type"""
        return WriterType.MARKDOWN


class _BaseWriterBackend(object):
    """_BaseWriterBackend"""

    def write_obj(self, out_path, obj, **bk_args):
        """write object"""
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        return self._write_obj(out_path, obj, **bk_args)

    def _write_obj(self, out_path, obj, **bk_args):
        """write object"""
        raise NotImplementedError


class TextWriterBackend(_BaseWriterBackend):
    """TextWriterBackend"""

    def __init__(self, mode="w", encoding="utf-8"):
        super().__init__()
        self.mode = mode
        self.encoding = encoding

    def _write_obj(self, out_path, obj):
        """write text object"""
        with open(out_path, mode=self.mode, encoding=self.encoding) as f:
            f.write(obj)


class HtmlWriterBackend(_BaseWriterBackend):

    def __init__(self, mode="w", encoding="utf-8"):
        super().__init__()
        self.mode = mode
        self.encoding = encoding

    def _write_obj(self, out_path, obj, **bk_args):
        with open(out_path, mode=self.mode, encoding=self.encoding) as f:
            f.write(obj)


class XlsxWriterBackend(_BaseWriterBackend):
    def _write_obj(self, out_path, obj, **bk_args):
        document_to_xl(obj, out_path)


class _ImageWriterBackend(_BaseWriterBackend):
    """_ImageWriterBackend"""

    pass


class OpenCVImageWriterBackend(_ImageWriterBackend):
    """OpenCVImageWriterBackend"""

    def _write_obj(self, out_path, obj):
        """write image object by OpenCV"""
        if isinstance(obj, Image.Image):
            # Assuming the channel order is RGB.
            arr = np.asarray(obj)[:, :, ::-1]
        elif isinstance(obj, np.ndarray):
            arr = obj
        else:
            raise TypeError("Unsupported object type")
        return cv2.imwrite(out_path, arr)


class PILImageWriterBackend(_ImageWriterBackend):
    """PILImageWriterBackend"""

    def __init__(self, format_=None):
        super().__init__()
        self.format = format_

    def _write_obj(self, out_path, obj):
        """write image object by PIL"""
        if isinstance(obj, Image.Image):
            img = obj
        elif isinstance(obj, np.ndarray):
            img = Image.fromarray(obj)
        else:
            raise TypeError("Unsupported object type")
        if len(img.getbands()) == 4:
            self.format = "PNG"
        return img.save(out_path, format=self.format)


class _VideoWriterBackend(_BaseWriterBackend):
    """_VideoWriterBackend"""

    pass


class OpenCVVideoWriterBackend(_VideoWriterBackend):
    """OpenCVImageWriterBackend"""

    def _write_obj(self, out_path, obj):
        """write video object by OpenCV"""
        obj, fps = obj
        if isinstance(obj, np.ndarray):
            vr = obj
            width, height = vr[0].shape[1], vr[0].shape[0]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Alternatively, use 'XVID'
            out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
            for frame in vr:
                out.write(frame)
            out.release()
        else:
            raise TypeError("Unsupported object type")


class _BaseJsonWriterBackend(object):
    def __init__(self, indent=4, ensure_ascii=False):
        super().__init__()
        self.indent = indent
        self.ensure_ascii = ensure_ascii

    def write_obj(self, out_path, obj, **bk_args):
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        return self._write_obj(out_path, obj, **bk_args)

    def _write_obj(self, out_path, obj):
        raise NotImplementedError


class JsonWriterBackend(_BaseJsonWriterBackend):
    def _write_obj(self, out_path, obj, **bk_args):
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, **bk_args)


class UJsonWriterBackend(_BaseJsonWriterBackend):
    # TODO
    def _write_obj(self, out_path, obj, **bk_args):
        raise NotImplementedError


class YAMLWriterBackend(_BaseWriterBackend):

    def __init__(self, mode="w", encoding="utf-8"):
        super().__init__()
        self.mode = mode
        self.encoding = encoding

    def _write_obj(self, out_path, obj, **bk_args):
        """write text object"""
        with open(out_path, mode=self.mode, encoding=self.encoding) as f:
            yaml.dump(obj, f, **bk_args)


class CSVWriter(_BaseWriter):
    """CSVWriter"""

    def __init__(self, backend="pandas", **bk_args):
        super().__init__(backend=backend, **bk_args)

    def write(self, out_path, obj):
        """write"""
        return self._backend.write_obj(str(out_path), obj)

    def _init_backend(self, bk_type, bk_args):
        """init backend"""
        if bk_type == "pandas":
            return PandasCSVWriterBackend(**bk_args)
        else:
            raise ValueError("Unsupported backend type")

    def get_type(self):
        """get type"""
        return WriterType.CSV


class _CSVWriterBackend(_BaseWriterBackend):
    """_CSVWriterBackend"""

    pass


class PandasCSVWriterBackend(_CSVWriterBackend):
    """PILImageWriterBackend"""

    def __init__(self):
        super().__init__()

    def _write_obj(self, out_path, obj):
        """write image object by PIL"""
        if isinstance(obj, pd.DataFrame):
            ts = obj
        else:
            raise TypeError("Unsupported object type")
        return ts.to_csv(out_path)


class MarkdownWriterBackend(_BaseWriterBackend):
    """MarkdownWriterBackend"""

    def __init__(self):
        super().__init__()

    def _write_obj(self, out_path, obj):
        """write markdown obj"""
        with open(out_path, mode="w", encoding="utf-8", errors="replace") as f:
            f.write(obj)
