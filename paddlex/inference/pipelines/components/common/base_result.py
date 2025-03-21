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

import inspect
from typing import Dict

from ....utils.io import ImageReader, ImageWriter
from ..utils.mixin import ImgMixin, JsonMixin, StrMixin


class BaseResult(dict, StrMixin, JsonMixin):
    """Base Result"""

    def __init__(self, data: Dict) -> None:
        """Initializes the instance with the provided data.

        Args:
            data (Dict): The data to initialize the instance with.
        """
        super().__init__(data)
        self._show_funcs = []
        StrMixin.__init__(self)
        JsonMixin.__init__(self)

    def save_all(self, save_path: str) -> None:
        """
        Save all show functions to the specified path if they accept a save_path argument.

        Args:
            save_path (str): The path to save the functions' output.

        Returns:
            None
        """
        for func in self._show_funcs:
            signature = inspect.signature(func)
            if "save_path" in signature.parameters:
                func(save_path=save_path)
            else:
                func()


class CVResult(BaseResult, ImgMixin):
    """Result For Computer Vision Tasks"""

    def __init__(self, data: Dict) -> None:
        """Initializes the instance with the given data and sets up image processing with the 'pillow' backend.

        Args:
            data (Dict): The data to initialize the instance with.
        """
        super().__init__(data)
        ImgMixin.__init__(self, "pillow")
        self._img_reader = ImageReader(backend="pillow")
        self._img_writer = ImageWriter(backend="pillow")
