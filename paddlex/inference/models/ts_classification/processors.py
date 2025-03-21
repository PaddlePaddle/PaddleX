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
class GetCls:
    """A class to process prediction outputs and return class IDs and scores."""

    def __init__(self):
        """Initializes the GetCls instance."""
        super().__init__()

    def __call__(self, pred_list: List[Any]) -> List[pd.DataFrame]:
        """
        Processes a list of predictions and returns a list of DataFrames with class IDs and scores.

        Args:
            pred_list (List[Any]): A list of predictions, where each prediction is expected to be an iterable of arrays.

        Returns:
            List[pd.DataFrame]: A list of DataFrames, each containing the class ID and score for the corresponding prediction.
        """
        return [self.getcls(pred) for pred in pred_list]

    def getcls(self, pred: Any) -> pd.DataFrame:
        """
        Computes the class ID and score from a single prediction.

        Args:
            pred (Any): A prediction, expected to be an iterable where the first element is an array representing logits or probabilities.

        Returns:
            pd.DataFrame: A DataFrame containing the class ID and score for the prediction.
        """
        pred_ts = pred[0]
        pred_ts -= np.max(pred_ts, axis=-1, keepdims=True)
        pred_ts = np.exp(pred_ts) / np.sum(np.exp(pred_ts), axis=-1, keepdims=True)
        classid = np.argmax(pred_ts, axis=-1)
        pred_score = pred_ts[classid]
        result = pd.DataFrame.from_dict({"classid": [classid], "score": [pred_score]})
        result.index.name = "sample"
        return result


@benchmark.timeit
class BuildPadMask:
    """A class to build padding masks for time series data."""

    def __init__(self, input_data: Dict[str, Any]):
        """
        Initializes the BuildPadMask instance.

        Args:
            input_data (Dict[str, Any]): A dictionary containing configuration data, including 'features'
                                         and 'pad_mask' keys that influence how padding is applied.
        """
        super().__init__()
        self.input_data = input_data

    def __call__(self, ts_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Applies padding mask to a list of time series data.

        Args:
            ts_list (List[Dict[str, Any]]): A list of dictionaries, each representing a time series instance
                                            with keys like 'features' and 'past_target'.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries with updated 'features' and 'pad_mask' keys.
        """
        return [self.padmask(ts) for ts in ts_list]

    def padmask(self, ts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Builds a padding mask for a single time series instance.

        Args:
            ts (Dict[str, Any]): A dictionary representing a time series instance, expected to have keys
                                 like 'features' and 'past_target'.

        Returns:
            Dict[str, Any]: The input dictionary with potentially updated 'features' and 'pad_mask' keys.
        """
        if "features" in self.input_data:
            ts["features"] = ts["past_target"]

        if "pad_mask" in self.input_data:
            target_dim = len(ts["features"])
            max_length = self.input_data["pad_mask"][-1]
            if max_length > 0:
                ones = np.ones(max_length, dtype=np.int32)
                if max_length != target_dim:
                    target_ndarray = np.array(ts["features"]).astype(np.float32)
                    target_ndarray_final = np.zeros(
                        [max_length, target_dim], dtype=np.int32
                    )
                    end = min(target_dim, max_length)
                    target_ndarray_final[:end, :] = target_ndarray
                    ts["features"] = target_ndarray_final
                    ones[end:] = 0.0
                    ts["pad_mask"] = ones
                else:
                    ts["pad_mask"] = ones
        return ts
