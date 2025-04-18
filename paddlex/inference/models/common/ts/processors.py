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

from .....utils.deps import class_requires_deps, is_dep_available
from ....utils.benchmark import benchmark
from .funcs import load_from_dataframe, time_feature

if is_dep_available("joblib"):
    import joblib

__all__ = [
    "BuildTSDataset",
    "TSCutOff",
    "TSNormalize",
    "TimeFeature",
    "TStoArray",
    "TStoBatch",
]


@benchmark.timeit
class TSCutOff:
    """Truncates time series data to a specified length for training.

    This class provides a method to truncate or cut off time series data
    to a specified input length, optionally skipping some initial data
    points. This is useful for preparing data for training models that
    require a fixed input size.
    """

    def __init__(self, size: Dict[str, int]):
        """Initializes the TSCutOff with size configurations.

        Args:
            size (Dict[str, int]): Dictionary containing size configurations,
                including 'in_chunk_len' for the input chunk length and
                optionally 'skip_chunk_len' for the number of initial data
                points to skip.
        """
        super().__init__()
        self.size = size

    def __call__(self, ts_list: List) -> List:
        """Applies the cut off operation to a list of time series.

        Args:
            ts_list (List): List of time series data frames to be truncated.

        Returns:
            List: List of truncated time series data frames.
        """
        return [self.cutoff(ts) for ts in ts_list]

    def cutoff(self, ts: Any) -> Any:
        """Truncates a single time series data frame to the specified length.

        This method truncates the time series data to the specified input
        chunk length, optionally skipping some initial data points. It raises
        a ValueError if the time series is too short.

        Args:
            ts: A single time series data frame to be truncated.

        Returns:
            Any: The truncated time series data frame.

        Raises:
            ValueError: If the time series length is less than the required
            minimum length (input chunk length plus any skip chunk length).
        """
        skip_len = self.size.get("skip_chunk_len", 0)
        if len(ts) < self.size["in_chunk_len"] + skip_len:
            raise ValueError(
                f"The length of the input data is {len(ts)}, but it should be at least {self.size['in_chunk_len'] + self.size['skip_chunk_len']} for training."
            )
        ts_data = ts[-(self.size["in_chunk_len"] + skip_len) :]
        return ts_data


@benchmark.timeit
@class_requires_deps("joblib")
class TSNormalize:
    """Normalizes time series data using a pre-fitted scaler.

    This class normalizes specified columns of time series data using a
    pre-fitted scaler loaded from a specified path. It supports normalization
    of both target and feature columns as specified in the parameters.
    """

    def __init__(self, scale_path: str, params_info: Dict[str, Any]):
        """Initializes the TSNormalize with a scaler and normalization parameters.

        Args:
            scale_path (str): Path to the pre-fitted scaler object file.
            params_info (Dict[str, Any]): Dictionary containing information
                about which columns to normalize, including 'target_cols'
                and 'feature_cols'.
        """
        super().__init__()
        self.scaler = joblib.load(scale_path)
        self.params_info = params_info

    def __call__(self, ts_list: List[pd.DataFrame]) -> List[pd.DataFrame]:
        """Applies normalization to a list of time series data frames.

        Args:
            ts_list (List[pd.DataFrame]): List of time series data frames to be normalized.

        Returns:
            List[pd.DataFrame]: List of normalized time series data frames.
        """
        return [self.tsnorm(ts) for ts in ts_list]

    def tsnorm(self, ts: pd.DataFrame) -> pd.DataFrame:
        """Normalizes specified columns of a single time series data frame.

        This method applies the scaler to normalize the specified target
        and feature columns of the time series.

        Args:
            ts (pd.DataFrame): A single time series data frame to be normalized.

        Returns:
            pd.DataFrame: The normalized time series data frame.
        """
        if self.params_info.get("target_cols", None) is not None:
            ts[self.params_info["target_cols"]] = self.scaler.transform(
                ts[self.params_info["target_cols"]]
            )
        if self.params_info.get("feature_cols", None) is not None:
            ts[self.params_info["feature_cols"]] = self.scaler.transform(
                ts[self.params_info["feature_cols"]]
            )
        return ts


@benchmark.timeit
class BuildTSDataset:
    """Constructs a time series dataset from a list of time series data frames."""

    def __init__(self, params_info: Dict[str, Any]):
        """Initializes the BuildTSDataset with parameters for dataset construction.

        Args:
            params_info (Dict[str, Any]): Dictionary containing parameters for
                constructing the time series dataset.
        """
        super().__init__()
        self.params_info = params_info

    def __call__(self, ts_list: List) -> List:
        """Applies the dataset construction to a list of time series.

        Args:
            ts_list (List): List of time series data frames.

        Returns:
            List: List of constructed time series datasets.
        """
        return [self.buildtsdata(ts) for ts in ts_list]

    def buildtsdata(self, ts) -> Any:
        """Builds a time series dataset from a single time series data frame.

        Args:
            ts: A single time series data frame.

        Returns:
            Any: A constructed time series dataset.
        """
        ts_data = load_from_dataframe(ts, **self.params_info)
        return ts_data


@benchmark.timeit
class TimeFeature:
    """Extracts time features from time series data for forecasting."""

    def __init__(
        self, params_info: Dict[str, Any], size: Dict[str, int], holiday: bool = False
    ):
        """Initializes the TimeFeature extractor.

        Args:
            params_info (Dict[str, Any]): Dictionary containing frequency information.
            size (Dict[str, int]): Dictionary containing the output chunk length.
            holiday (bool, optional): Whether to include holiday features. Defaults to False.
        """
        super().__init__()
        self.freq = params_info["freq"]
        self.size = size
        self.holiday = holiday

    def __call__(self, ts_list: List) -> List:
        """Applies time feature extraction to a list of time series.

        Args:
            ts_list (List): List of time series data frames.

        Returns:
            List: List of time series with extracted time features.
        """
        return [self.timefeat(ts) for ts in ts_list]

    def timefeat(self, ts: Dict[str, Any]) -> Any:
        """Extracts time features from a single time series data frame.

        Args:
            ts: A single time series data frame.

        Returns:
            Any: The time series with added time features.
        """
        if not self.holiday:
            ts = time_feature(
                ts,
                self.freq,
                ["hourofday", "dayofmonth", "dayofweek", "dayofyear"],
                self.size["out_chunk_len"],
            )
        else:
            ts = time_feature(
                ts,
                self.freq,
                [
                    "minuteofhour",
                    "hourofday",
                    "dayofmonth",
                    "dayofweek",
                    "dayofyear",
                    "monthofyear",
                    "weekofyear",
                    "holidays",
                ],
                self.size["out_chunk_len"],
            )
        return ts


@benchmark.timeit
class TStoArray:
    """Converts time series data into arrays for model input."""

    def __init__(self, input_data: Dict[str, Any]):
        """Initializes the TStoArray converter.

        Args:
            input_data (Dict[str, Any]): Dictionary specifying the input data format.
        """
        super().__init__()
        self.input_data = input_data

    def __call__(self, ts_list: List[Dict[str, Any]]) -> List[List[np.ndarray]]:
        """Converts a list of time series data frames into arrays.

        Args:
            ts_list (List[Dict[str, Any]]): List of time series data frames.

        Returns:
            List[List[np.ndarray]]: List of lists of arrays for each time series.
        """
        return [self.tstoarray(ts) for ts in ts_list]

    def tstoarray(self, ts: Dict[str, Any]) -> List[np.ndarray]:
        """Converts a single time series data frame into arrays.

        Args:
            ts (Dict[str, Any]): A single time series data frame.

        Returns:
            List[np.ndarray]: List of arrays representing the time series data.
        """
        ts_list = []
        input_name = list(self.input_data.keys())
        input_name.sort()
        for key in input_name:
            ts_list.append(np.array(ts[key]).astype("float32"))

        return ts_list


@benchmark.timeit
class TStoBatch:
    """Convert a list of time series into batches for processing.

    This class provides a method to convert a list of time series data into
    batches. Each time series in the list is assumed to be a sequence of
    equal-length arrays or DataFrames.
    """

    def __call__(self, ts_list: List[np.ndarray]) -> List[np.ndarray]:
        """Convert a list of time series into batches.

        This method stacks time series data along a new axis to create batches.
        It assumes that each time series in the list has the same length.

        Args:
            ts_list (List[np.ndarray]): A list of time series, where each
                time series is represented as a list or array of equal length.

        Returns:
            List[np.ndarray]: A list of batches, where each batch is a stacked
            array of time series data at the same index across all series.
        """
        n = len(ts_list[0])
        return [np.stack([ts[i] for ts in ts_list], axis=0) for i in range(n)]
