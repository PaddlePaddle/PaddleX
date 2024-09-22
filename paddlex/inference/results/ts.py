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
import numpy as np
import pandas as pd

from ..utils.io import TSWriter
from .base import BaseResult


class TSFcResult(BaseResult):

    def __init__(self, data):
        super().__init__(data)
        self._writer = TSWriter(backend="pandas")

    def save_to_csv(self, save_path):
        """write ts"""
        if not save_path.endswith(".csv"):
            save_path = Path(save_path) / f"{Path(self['ts_path']).stem}.csv"
        self._writer.write(save_path, self["forecast"])


class TSClsResult(BaseResult):

    def __init__(self, data):
        super().__init__(
            {"ts_path": data["ts_path"], "classification": self.process_data(data)}
        )
        self._writer = TSWriter(backend="pandas")

    def process_data(self, data):
        """apply"""
        pred_ts = data["forecast"][0]
        pred_ts -= np.max(pred_ts, axis=-1, keepdims=True)
        pred_ts = np.exp(pred_ts) / np.sum(np.exp(pred_ts), axis=-1, keepdims=True)
        classid = np.argmax(pred_ts, axis=-1)
        pred_score = pred_ts[classid]
        result = {"classid": [classid], "score": [pred_score]}
        result = pd.DataFrame.from_dict(result)
        result.index.name = "sample"
        return result

    def save_to_csv(self, save_path):
        """write ts"""
        if not save_path.endswith(".csv"):
            save_path = Path(save_path) / f"{Path(self['ts_path']).stem}.csv"
        self._writer.write(save_path, self["classification"])
