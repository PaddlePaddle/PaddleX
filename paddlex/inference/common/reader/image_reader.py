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

import numpy as np
import cv2

from ...utils.io import ImageReader, PDFReader
from ...utils.benchmark import benchmark


@benchmark.timeit
class ReadImage:
    """Load image from the file."""

    _FLAGS_DICT = {
        "BGR": cv2.IMREAD_COLOR,
        "RGB": cv2.IMREAD_COLOR,
        "GRAY": cv2.IMREAD_GRAYSCALE,
    }

    def __init__(self, format="BGR"):
        """
        Initialize the instance.

        Args:
            format (str, optional): Target color format to convert the image to.
                Choices are 'BGR', 'RGB', and 'GRAY'. Default: 'BGR'.
        """
        super().__init__()
        self.format = format
        flags = self._FLAGS_DICT[self.format]
        self._img_reader = ImageReader(backend="opencv", flags=flags)

    def __call__(self, imgs):
        """apply"""
        return [self.read(img) for img in imgs]

    def read(self, img):
        if isinstance(img, np.ndarray):
            if self.format == "RGB":
                img = img[:, :, ::-1]
            return img
        elif isinstance(img, str):
            blob = self._img_reader.read(img)
            if blob is None:
                raise Exception(f"Image read Error: {img}")

            if self.format == "RGB":
                if blob.ndim != 3:
                    raise RuntimeError("Array is not 3-dimensional.")
                # BGR to RGB
                blob = blob[..., ::-1]
            return blob
        else:
            raise TypeError(
                f"ReadImage only supports the following types:\n"
                f"1. str, indicating a image file path or a directory containing image files.\n"
                f"2. numpy.ndarray.\n"
                f"However, got type: {type(img).__name__}."
            )
