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


import enum
import itertools
import random

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageOps

from ....utils.deps import class_requires_deps, is_dep_available

if is_dep_available("opencv-contrib-python"):
    import cv2
if is_dep_available("pypdfium2"):
    import pypdfium2 as pdfium
if is_dep_available("soundfile"):
    import soundfile

__all__ = [
    "ReaderType",
    "ImageReader",
    "VideoReader",
    "CSVReader",
    "PDFReader",
    "YAMLReader",
    "AudioReader",
]


class ReaderType(enum.Enum):
    """ReaderType"""

    IMAGE = 1
    GENERATIVE = 2
    POINT_CLOUD = 3
    JSON = 4
    TS = 5
    PDF = 6
    YAML = 8


class _BaseReader(object):
    """_BaseReader"""

    def __init__(self, backend, **bk_args):
        super().__init__()
        if len(bk_args) == 0:
            bk_args = self.get_default_backend_args()
        self.bk_type = backend
        self.bk_args = bk_args
        self._backend = self.get_backend()

    def read(self, in_path):
        """read file from path"""
        raise NotImplementedError

    def get_backend(self, bk_args=None):
        """get the backend"""
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


class PDFReader(_BaseReader):
    """PDFReader"""

    def __init__(self, backend="pypdfium2", **bk_args):
        super().__init__(backend, **bk_args)

    def read(self, in_path):
        yield from self._backend.read_file(str(in_path))

    def _init_backend(self, bk_type, bk_args):
        return PDFReaderBackend(**bk_args)

    def get_type(self):
        return ReaderType.PDF


class ImageReader(_BaseReader):
    """ImageReader"""

    def __init__(self, backend="opencv", **bk_args):
        super().__init__(backend=backend, **bk_args)

    def read(self, in_path):
        """read the image file from path"""
        arr = self._backend.read_file(str(in_path))
        return arr

    def _init_backend(self, bk_type, bk_args):
        """init backend"""
        if bk_type == "opencv":
            return OpenCVImageReaderBackend(**bk_args)
        elif bk_type == "pil" or bk_type == "pillow":
            return PILImageReaderBackend(**bk_args)
        else:
            raise ValueError("Unsupported backend type")

    def get_type(self):
        """get type"""
        return ReaderType.IMAGE


class _GenerativeReader(_BaseReader):
    """_GenerativeReader"""

    def get_type(self):
        """get type"""
        return ReaderType.GENERATIVE


def is_generative_reader(reader):
    """is_generative_reader"""
    return isinstance(reader, _GenerativeReader)


class VideoReader(_GenerativeReader):
    """VideoReader"""

    def __init__(
        self,
        backend="opencv",
        st_frame_id=0,
        max_num_frames=None,
        auto_close=True,
        **bk_args,
    ):
        super().__init__(backend=backend, **bk_args)
        self.st_frame_id = st_frame_id
        self.max_num_frames = max_num_frames
        self.auto_close = auto_close
        self._fps = 0

    def read(self, in_path):
        """read vide file from path"""
        self._backend.set_pos(self.st_frame_id)
        gen = self._backend.read_file(str(in_path))
        if self.max_num_frames is not None:
            gen = itertools.islice(gen, self.num_frames)
        yield from gen
        if self.auto_close:
            self._backend.close()

    def get_fps(self):
        """get fps"""
        return self._backend.get_fps()

    def _init_backend(self, bk_type, bk_args):
        """init backend"""
        if bk_type == "opencv":
            return OpenCVVideoReaderBackend(**bk_args)
        elif bk_type == "decord":
            return DecordVideoReaderBackend(**bk_args)
        else:
            raise ValueError("Unsupported backend type")


class YAMLReader(_BaseReader):

    def __init__(self, backend="PyYAML", **bk_args):
        super().__init__(backend, **bk_args)

    def read(self, in_path):
        return self._backend.read_file(str(in_path))

    def _init_backend(self, bk_type, bk_args):
        if bk_type == "PyYAML":
            return YAMLReaderBackend(**bk_args)
        else:
            raise ValueError("Unsupported backend type")

    def get_type(self):
        return ReaderType.YAML


class _BaseReaderBackend(object):
    """_BaseReaderBackend"""

    def read_file(self, in_path):
        """read file from path"""
        raise NotImplementedError


class _ImageReaderBackend(_BaseReaderBackend):
    """_ImageReaderBackend"""


@class_requires_deps("opencv-contrib-python")
class OpenCVImageReaderBackend(_ImageReaderBackend):
    """OpenCVImageReaderBackend"""

    def __init__(self, flags=None):
        super().__init__()
        if flags is None:
            flags = cv2.IMREAD_COLOR
        self.flags = flags

    def read_file(self, in_path):
        """read image file from path by OpenCV"""
        return cv2.imread(in_path, flags=self.flags)


class PILImageReaderBackend(_ImageReaderBackend):
    """PILImageReaderBackend"""

    def __init__(self):
        super().__init__()

    def read_file(self, in_path):
        """read image file from path by PIL"""
        return ImageOps.exif_transpose(Image.open(in_path))


@class_requires_deps("pypdfium2", "opencv-contrib-python")
class PDFReaderBackend(_BaseReaderBackend):

    def __init__(self, rotate=0, zoom=2.0):
        super().__init__()
        self._rotation = rotate
        self._scale = zoom

    def read_file(self, in_path):
        for page in pdfium.PdfDocument(in_path):
            image = page.render(scale=self._scale, rotation=self._rotation).to_pil()
            image = image.convert("RGB")
            img_cv = np.array(image)
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
            yield img_cv


class _VideoReaderBackend(_BaseReaderBackend):
    """_VideoReaderBackend"""

    def set_pos(self, pos):
        """set pos"""
        raise NotImplementedError

    def close(self):
        """close io"""
        raise NotImplementedError


@class_requires_deps("opencv-contrib-python")
class OpenCVVideoReaderBackend(_VideoReaderBackend):
    """OpenCVVideoReaderBackend"""

    def __init__(self, **bk_args):
        super().__init__()
        self.cap_init_args = bk_args
        self.num_seg = bk_args.get("num_seg", None)
        self._cap = None
        self._pos = 0
        self._max_num_frames = None

    def get_fps(self):
        return self._cap.get(cv2.CAP_PROP_FPS)

    def read_file(self, in_path):
        """read video file from path"""
        if self._cap is not None:
            self._cap_release()
        self._cap = self._cap_open(in_path)
        if self._pos is not None:
            self._cap_set_pos()
        return self._read_frames(self._cap)

    def _read_frames(self, cap):
        """read frames"""
        if self.num_seg:
            queue = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                queue = []
                if (
                    len(queue) <= 0
                ):  # At initialization, populate queue with initial frame
                    for i in range(self.num_seg):
                        queue.append(frame)
                queue.append(frame)
                queue.pop(0)
                yield queue.copy()
        else:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame
        self._cap_release()

    def _cap_open(self, video_path):
        self.cap_init_args.pop("num_seg", None)
        self.cap_init_args.pop("seg_len", None)
        self.cap_init_args.pop("sample_type", None)
        self._cap = cv2.VideoCapture(video_path, **self.cap_init_args)
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open {video_path}")
        return self._cap

    def _cap_release(self):
        self._cap.release()

    def _cap_set_pos(self):
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, self._pos)

    def set_pos(self, pos):
        self._pos = pos

    def close(self):
        if self._cap is not None:
            self._cap_release()
            self._cap = None


class DecordVideoReaderBackend(_VideoReaderBackend):
    """DecordVideoReaderBackend"""

    def __init__(self, **bk_args):
        super().__init__()
        self.cap_init_args = bk_args
        self._cap = None
        self._pos = 0
        self._max_num_frames = None
        self.num_seg = bk_args.get("num_seg", 8)
        self.seg_len = bk_args.get("seg_len", 1)
        self.sample_type = bk_args.get("sample_type", 1)
        self.valid_mode = True
        self._fps = 0

        # XXX(gaotingquan): There is a conflict with `paddle` when import `decord` globally.
        try:
            import decord

            self.decord_module = decord
        except ModuleNotFoundError():
            raise Exception(
                "Please install `decord` manually, otherwise, the related model cannot work. It can be automatically installed only on `x86_64`. Refers: `https://github.com/dmlc/decord`."
            )

    def set_pos(self, pos):
        self._pos = pos

    def sample(self, frames_len, video_object):
        frames_idx = []
        average_dur = int(frames_len / self.num_seg)
        for i in range(self.num_seg):
            idx = 0
            if not self.valid_mode:
                if average_dur >= self.seg_len:
                    idx = random.randint(0, average_dur - self.seg_len)
                    idx += i * average_dur
                elif average_dur >= 1:
                    idx += i * average_dur
                else:
                    idx = i
            else:
                if average_dur >= self.seg_len:
                    idx = (average_dur - 1) // 2
                    idx += i * average_dur
                elif average_dur >= 1:
                    idx += i * average_dur
                else:
                    idx = i
            for jj in range(idx, idx + self.seg_len):
                frames_idx.append(int(jj % frames_len))
        frames_select = video_object.get_batch(frames_idx)
        # dearray_to_img
        np_frames = frames_select.asnumpy()
        imgs = []
        for i in range(np_frames.shape[0]):
            imgbuf = np_frames[i]
            imgs.append(imgbuf)
        return imgs

    def get_fps(self):
        return self._cap.get_avg_fps()

    def read_file(self, in_path):
        """read video file from path"""
        self._cap = self.decord_module.VideoReader(in_path)
        frame_len = len(self._cap)
        if self.sample_type == "uniform":
            sample_video = self.sample(frame_len, self._cap)
            return sample_video
        else:
            return self._cap

    def close(self):
        pass


class CSVReader(_BaseReader):
    """CSVReader"""

    def __init__(self, backend="pandas", **bk_args):
        super().__init__(backend=backend, **bk_args)

    def read(self, in_path):
        """read the image file from path"""
        arr = self._backend.read_file(str(in_path))
        return arr

    def _init_backend(self, bk_type, bk_args):
        """init backend"""
        if bk_type == "pandas":
            return PandasCSVReaderBackend(**bk_args)
        else:
            raise ValueError("Unsupported backend type")

    def get_type(self):
        """get type"""
        return ReaderType.TS


class _CSVReaderBackend(_BaseReaderBackend):
    """_CSVReaderBackend"""


class PandasCSVReaderBackend(_CSVReaderBackend):
    """PandasCSVReaderBackend"""

    def __init__(self):
        super().__init__()

    def read_file(self, in_path):
        """read image file from path by OpenCV"""
        return pd.read_csv(in_path)


class YAMLReaderBackend(_BaseReaderBackend):

    def read_file(self, in_path, **kwargs):
        with open(in_path, "r", encoding="utf-8", **kwargs) as yaml_file:
            data = yaml.load(yaml_file, Loader=yaml.FullLoader)
        return data


class AudioReader(_BaseReader):
    def __init__(self, backend="wav", **bk_args):
        super().__init__(backend="wav", **bk_args)

    def _init_backend(self, bk_type, bk_args):
        """init backend"""
        if bk_type == "wav":
            return WAVReaderBackend(**bk_args)
        else:
            raise ValueError("Unsupported backend type")

    def read(self, in_path):
        audio, audio_sample_rate = self._backend.read_file(str(in_path))
        return audio, audio_sample_rate


class _AudioReaderBackend(_BaseReaderBackend):
    """_AudioReaderBackend"""


@class_requires_deps("soundfile")
class WAVReaderBackend(_AudioReaderBackend):
    """PandasCSVReaderBackend"""

    def __init__(self):
        super().__init__()

    def read_file(self, in_path):
        """read wav file from path"""
        audio, audio_sample_rate = soundfile.read(
            in_path, dtype="float32", always_2d=True
        )
        return audio, audio_sample_rate
