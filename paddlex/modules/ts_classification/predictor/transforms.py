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

import numpy as np
import pandas as pd

from ....utils import logging
from ...base import BaseTransform
from ...base.predictor.io.writers import TSWriter
from .keys import TSFCKeys as K

__all__ = ["SaveTSClsResults"]


class SaveTSClsResults(BaseTransform):
    """SaveSegResults"""

    def __init__(self, save_dir):
        super().__init__()
        self.save_dir = save_dir
        self._writer = TSWriter(backend="pandas")

    def apply(self, data):
        """apply"""
        pred_ts = data[K.PRED]
        pred_ts -= np.max(pred_ts, axis=-1, keepdims=True)
        pred_ts = np.exp(pred_ts) / np.sum(np.exp(pred_ts), axis=-1, keepdims=True)
        classid = np.argmax(pred_ts, axis=-1)
        pred_score = pred_ts[classid]
        result = {"classid": [classid], "score": [pred_score]}
        result = pd.DataFrame.from_dict(result)
        result.index.name = "sample"
        file_name = os.path.basename(data[K.TS_PATH])
        ts_save_path = os.path.join(self.save_dir, file_name)
        self._write_ts(ts_save_path, result)

        return data

    @classmethod
    def get_input_keys(cls):
        """get input keys"""
        return [K.PRED]

    @classmethod
    def get_output_keys(cls):
        """get output keys"""
        return []

    def _write_ts(self, path, ts):
        """write ts"""
        if os.path.exists(path):
            logging.warning(f"{path} already exists. Overwriting it.")
        self._writer.write(path, ts)

    @staticmethod
    def _add_suffix(path, suffix):
        """add suffix"""
        stem, ext = os.path.splitext(path)
        return stem + suffix + ext

    @staticmethod
    def _replace_ext(path, new_ext):
        """replace ext"""
        stem, _ = os.path.splitext(path)
        return stem + new_ext
