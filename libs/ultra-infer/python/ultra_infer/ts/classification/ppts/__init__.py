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

from __future__ import absolute_import

import os
from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from .... import ModelFormat
from ....py_only import PyOnlyProcessorChain
from ....py_only.ts import PyOnlyTSModel
from ....py_only.ts import processors as P
from ....utils.misc import load_config


class PyOnlyClassificationModel(PyOnlyTSModel):
    def __init__(
        self,
        model_file,
        params_file,
        config_file,
        scaler_file=None,
        runtime_option=None,
        model_format=ModelFormat.PADDLE,
    ):
        self._model_file = model_file
        self._params_file = params_file
        self._model_format = model_format
        super().__init__(runtime_option)
        if scaler_file is None:
            config_dir = os.path.dirname(config_file)
            scaler_file = os.path.join(config_dir, "scaler.pkl")
        self._config = load_config(config_file)
        self._preprocessor = _PyOnlyClassificationPreprocessor(
            self._config, scaler_file
        )
        self._postprocessor = _PyOnlyClassificationPostprocessor()

    def model_name():
        return "PyOnlyClassificationModel"

    def batch_predict(self, ts_list):
        data_list = []
        for csv_data in ts_list:
            data = {"ori_ts": deepcopy(csv_data), "ts": csv_data}
            data = self._preprocessor.run(data)
            data_list.append(data)

        input_data = {}
        input_num = self._runtime.num_inputs()
        for idx in range(input_num):
            input_name = self._runtime.get_input_info(idx).name
            ts_data = np.stack(
                [data["ts"][idx] for data in data_list], axis=0, dtype=np.float32
            )
            ts_data = np.ascontiguousarray(ts_data)
            input_data[input_name] = ts_data

        output_arrs = self._runtime.infer(input_data)

        results = []
        for data in output_arrs[0]:
            data = {"pred": data}
            result = self._postprocessor.run(data)
            results.append(result)
        return results

    def _update_option(self):
        self._option.set_model_path(
            self._model_file, self._params_file, self._model_format
        )


class _PyOnlyClassificationPreprocessor(object):
    def __init__(self, config, scaler_file):
        super().__init__()
        self.scaler_file = scaler_file
        processors = self._build_processors(config)
        self._processor_chain = PyOnlyProcessorChain(processors)

    def run(self, data):
        return self._processor_chain(data)

    def _build_processors(self, config):
        processors = []

        if config.get("scale", None):
            if not os.path.exists(self.scaler_file):
                raise Exception(f"Cannot find scaler file: {self.scaler_file}")
            processors.append(P.Normalize(self.scaler_file, config["info_params"]))

        processors.append(P.BuildTSDataset(config["info_params"]))
        processors.append(P.BuildPaddedMask(config["input_data"]))
        processors.append(P.DataFrame2Arrays(config["input_data"]))
        return processors


class _PyOnlyClassificationPostprocessor(object):
    def __init__(self):
        super().__init__()

    def run(self, data):
        pred_ts = data["pred"]
        pred_ts -= np.max(pred_ts, axis=-1, keepdims=True)
        pred_ts = np.exp(pred_ts) / np.sum(np.exp(pred_ts), axis=-1, keepdims=True)
        class_id = np.argmax(pred_ts, axis=-1)
        pred_score = pred_ts[class_id]
        result = _PyOnlyClassificationResult(class_id=class_id, score=pred_score)
        return result


@dataclass
class _PyOnlyClassificationResult(object):
    class_id: int
    score: float
