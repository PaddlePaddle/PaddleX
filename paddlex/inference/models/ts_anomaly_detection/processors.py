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

from ...utils.benchmark import benchmark


@benchmark.timeit
class GetAnomaly:
    """A class to detect anomalies in time series data based on a model threshold."""

    def __init__(self, model_threshold: float, info_params: Dict[str, Any]):
        """
        Initializes the GetAnomaly class with a model threshold and parameters information.

        Args:
            model_threshold (float): The threshold for determining anomalies.
            info_params (Dict[str, Any]): Configuration parameters including target columns and time column name.
        """
        super().__init__()
        self.model_threshold = model_threshold
        self.info_params = info_params

    def __call__(
        self, ori_ts_list: List[Dict[str, Any]], pred_list: List[np.ndarray]
    ) -> List[pd.DataFrame]:
        """
        Detects anomalies for a list of time series predictions.

        Args:
            ori_ts_list (List[Dict[str, Any]]): Original time series data for each prediction, including past and covariate information.
            pred_list (List[np.ndarray]): List of prediction arrays corresponding to each time series in ori_ts_list.

        Returns:
            List[pd.DataFrame]: A list of DataFrames, each containing anomaly labels for the time series.
        """
        return [
            self.getanomaly(ori_ts, pred)
            for ori_ts, pred in zip(ori_ts_list, pred_list)
        ]

    def getanomaly(self, ori_ts: Dict[str, Any], pred: np.ndarray) -> pd.DataFrame:
        """
        Detects anomalies in a single time series prediction.

        Args:
            ori_ts (Dict[str, Any]): Original time series data for a single time series.
            pred (np.ndarray): Prediction array for the given time series.

        Returns:
            pd.DataFrame: A DataFrame containing anomaly labels for the time series.

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

        anomaly_score = np.mean(np.square(pred - np.array(ts)), axis=-1)
        anomaly_label = (anomaly_score >= self.model_threshold) + 0

        past_target_index = ts.index
        past_target_index.name = self.info_params["time_col"]
        anomaly_label_df = pd.DataFrame(
            np.reshape(anomaly_label, newshape=[pred.shape[0], -1]),
            index=past_target_index,
            columns=["label"],
        )
        return anomaly_label_df
