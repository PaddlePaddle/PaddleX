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

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ....utils.deps import class_requires_deps, is_dep_available
from ...utils.benchmark import benchmark

if is_dep_available("joblib"):
    import joblib


@benchmark.timeit
@class_requires_deps("joblib")
class TSDeNormalize:
    """A class to de-normalize time series prediction data using a pre-fitted scaler."""

    def __init__(self, scale_path: str, params_info: dict):
        """
        Initializes the TSDeNormalize class with a scaler and parameters information.

        Args:
            scale_path (str): The file path to the serialized scaler object.
            params_info (dict): Additional parameters information.
        """
        super().__init__()
        self.scaler = joblib.load(scale_path)
        self.params_info = params_info

    def __call__(self, preds_list: List[pd.DataFrame]) -> List[pd.DataFrame]:
        """
        Applies de-normalization to a list of prediction DataFrames.

        Args:
            preds_list (List[pd.DataFrame]): A list of DataFrames containing normalized prediction data.

        Returns:
            List[pd.DataFrame]: A list of DataFrames with de-normalized prediction data.
        """
        return [self.tsdenorm(pred) for pred in preds_list]

    def tsdenorm(self, pred: pd.DataFrame) -> pd.DataFrame:
        """
        De-normalizes a single prediction DataFrame.

        Args:
            pred (pd.DataFrame): A DataFrame containing normalized prediction data.

        Returns:
            pd.DataFrame: A DataFrame with de-normalized prediction data.
        """
        scale_cols = pred.columns.values.tolist()
        pred[scale_cols] = self.scaler.inverse_transform(pred[scale_cols])
        return pred


@benchmark.timeit
class ArraytoTS:
    """A class to convert arrays of predictions into time series format."""

    def __init__(self, info_params: Dict[str, Any]):
        """
        Initializes the ArraytoTS class with the given parameters.

        Args:
            info_params (Dict[str, Any]): Configuration parameters including target columns, frequency, and time column name.
        """
        super().__init__()
        self.info_params = info_params

    def __call__(
        self, ori_ts_list: List[Dict[str, Any]], pred_list: List[np.ndarray]
    ) -> List[pd.DataFrame]:
        """
        Converts a list of arrays to a list of time series DataFrames.

        Args:
            ori_ts_list (List[Dict[str, Any]]): Original time series data for each prediction, including past and covariate information.
            pred_list (List[np.ndarray]): List of prediction arrays corresponding to each time series in ori_ts_list.

        Returns:
            List[pd.DataFrame]: A list of DataFrames, each representing the forecasted time series.
        """
        return [
            self.arraytots(ori_ts, pred) for ori_ts, pred in zip(ori_ts_list, pred_list)
        ]

    def arraytots(self, ori_ts: Dict[str, Any], pred: np.ndarray) -> pd.DataFrame:
        """
        Converts a single array prediction to a time series DataFrame.

        Args:
            ori_ts (Dict[str, Any]): Original time series data for a single time series.
            pred (np.ndarray): Prediction array for the given time series.

        Returns:
            pd.DataFrame: A DataFrame representing the forecasted time series.

        Raises:
            ValueError: If none of the expected keys are found in ori_ts.
        """
        pred = pred[0]
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
        return future_target
