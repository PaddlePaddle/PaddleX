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
import numpy as np
import pandas as pd
from typing import List
from dataclasses import dataclass

from .... import UltraInferModel, ModelFormat
from ....py_only.ts import PyOnlyTSModel
from ....utils.misc import load_config
from ....py_only import PyOnlyProcessorChain
from ....py_only.ts import PyOnlyTSModel, processors as P


class PyOnlyForecastingModel(PyOnlyTSModel):
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
        self._preprocessor = _PyOnlyForecastingPreprocessor(self._config, scaler_file)
        self._postprocessor = _PyOnlyForecastingPostprocessor(self._config, scaler_file)

    def model_name():
        return "PyOnlyForecastingModel"

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
        for idx, data in enumerate(output_arrs[0]):
            data = {"ori_ts": data_list[idx]["ori_ts"], "pred": data}
            result = self._postprocessor.run(data)
            results.append(result)
        return results

    def _update_option(self):
        self._option.set_model_path(
            self._model_file, self._params_file, self._model_format
        )


class _PyOnlyForecastingPreprocessor(object):
    def __init__(self, config, scaler_file):
        super().__init__()
        self.scaler_file = scaler_file
        processors = self._build_processors(config)
        self._processor_chain = PyOnlyProcessorChain(processors)

    def run(self, data):
        return self._processor_chain(data)

    def _build_processors(self, config):
        processors = []
        processors.append(P.CutOff(config["size"]))

        if config.get("scale", None):
            if not os.path.exists(self.scaler_file):
                raise Exception(f"Cannot find scaler file: {self.scaler_file}")
            processors.append(P.Normalize(self.scaler_file, config["info_params"]))

        processors.append(P.BuildTSDataset(config["info_params"]))

        if config.get("time_feat", None):
            processors.append(
                P.CalcTimeFeatures(
                    config["info_params"],
                    config["size"],
                    config["holiday"],
                )
            )

        processors.append(P.DataFrame2Arrays(config["input_data"]))
        return processors


class _PyOnlyForecastingPostprocessor(object):
    def __init__(self, config, scaler_file):
        super().__init__()
        self.scaler_file = scaler_file
        self.info_params = config["info_params"]
        processors = self._build_processors(config)
        self._processor_chain = PyOnlyProcessorChain(processors)

    def run(self, data):
        ori_ts = data["ori_ts"]
        pred = data["pred"]
        if ori_ts.get("past_target", None) is not None:
            ts = ori_ts["past_target"]
        elif ori_ts.get("observed_cov_numeric", None) is not None:
            ts = ori_ts["observed_cov_numeric"]
        elif ori_ts.get("known_cov_numeric", None) is not None:
            ts = ori_ts["known_cov_numeric"]
        elif ori_ts.get("static_cov_numeric", None) is not None:
            ts = ori_ts["static_cov_numeric"]
        else:
            raise ValueError("No value in ori_ts")

        column_name = (
            self.info_params["target_cols"]
            if "target_cols" in self.info_params
            else self.info_params["feature_cols"]
        )
        if isinstance(self.info_params["freq"], str):
            past_target_index = ts.index
            if past_target_index.freq is None:
                past_target_index.freq = pd.infer_freq(ts.index)
            future_target_index = pd.date_range(
                past_target_index[-1] + past_target_index.freq,
                periods=pred.shape[0],
                freq=self.info_params["freq"],
                name=self.info_params["time_col"],
            )
        elif isinstance(self.info_params["freq"], int):
            start_idx = max(ts.index) + 1
            stop_idx = start_idx + pred.shape[0]
            future_target_index = pd.RangeIndex(
                start=start_idx,
                stop=stop_idx,
                step=self.info_params["freq"],
                name=self.info_params["time_col"],
            )

        future_target = pd.DataFrame(
            np.reshape(pred, newshape=[pred.shape[0], -1]),
            index=future_target_index,
            columns=column_name,
        )
        data = {"pred": future_target}
        forecast_dataframe = self._processor_chain(data)
        forecast = forecast_dataframe["pred"]
        col_names = forecast.columns.tolist()
        data = [forecast[col_name].tolist() for col_name in col_names]
        dates = [int(i.timestamp()) for i in forecast.index]
        result = _PyOnlyForecastingResult(dates=dates, col_names=col_names, data=data)
        return result

    def _build_processors(self, config):
        processors = []
        if config.get("scale", None):
            if not os.path.exists(self.scaler_file):
                raise Exception(f"Cannot find scaler file: {self.scaler_file}")
            processors.append(P.Denormalize(self.scaler_file, config["info_params"]))
        return processors


@dataclass
class _PyOnlyForecastingResult(object):
    dates: List[int]
    col_names: List[str]
    data: List[List[float]]
