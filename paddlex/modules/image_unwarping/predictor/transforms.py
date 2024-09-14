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
from pathlib import Path
import numpy as np

from .keys import WarpKeys as K
from ...base import BaseTransform
from ...base.predictor.io import ImageWriter, ImageReader
from ....utils import logging


_all__ = ['DocTrPostProcess',  'SaveDocTrResults']


class DocTrPostProcess(BaseTransform):
    """ normalize image such as substract mean, divide std
    """

    def __init__(self, scale=None, **kwargs):
        if isinstance(scale, str):
            scale = eval(scale)
        self.scale = np.float32(scale if scale is not None else 255.0)

    def apply(self, data):
        im = data[K.DOCTR_IMG]
        assert isinstance(im,
                          np.ndarray), "invalid input 'im' in DocTrPostProcess"

        im = im.squeeze()
        im = im.transpose(1, 2, 0)
        im *= self.scale
        im = im[:, :, ::-1]
        im = im.astype("uint8", copy=False) 
        data[K.DOCTR_IMG] = im
        return data

    @classmethod
    def get_input_keys(cls):
        return [K.DOCTR_IMG]

    @classmethod
    def get_output_keys(cls):
        return [K.DOCTR_IMG]


class SaveDocTrResults(BaseTransform):

    _FILE_EXT = '.png'

    def __init__(self, save_dir, file_name=None):
        super().__init__()
        self.save_dir = save_dir
        self._writer = ImageWriter(backend='opencv')

    @staticmethod
    def _replace_ext(path, new_ext):
        """replace ext"""
        stem, _ = os.path.splitext(path)
        return stem + new_ext
    
    def apply(self, data):
        ori_path = data[K.IM_PATH]
        file_name = os.path.basename(ori_path)
        file_name = self._replace_ext(file_name, self._FILE_EXT)
        save_path = os.path.join(self.save_dir, file_name)
        doctr_img = data[K.DOCTR_IMG]
        self._writer.write(save_path, doctr_img)
        return data

    @classmethod
    def get_input_keys(cls):
        return [K.DOCTR_IMG]

    @classmethod
    def get_output_keys(cls):
        return []
