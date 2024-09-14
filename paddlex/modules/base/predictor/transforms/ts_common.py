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
import joblib
import numpy as np
import pandas as pd

from .....utils.download import download
from .....utils.cache import CACHE_DIR
from ..transform import BaseTransform
from ..io.readers import TSReader
from ..io.writers import TSWriter
from .ts_functions import load_from_dataframe, time_feature


__all__ = [
    "ReadTS",
    "BuildTSDataset",
    "TSCutOff",
    "TSNormalize",
    "TimeFeature",
    "TStoArray",
    "BuildPadMask",
]


class ReadTS(BaseTransform):
    """Load image from the file."""

    def __init__(self):
        """
        Initialize the instance.

        Args:
            format (str, optional): Target color format to convert the image to.
                Choices are 'BGR', 'RGB', and 'GRAY'. Default: 'BGR'.
        """
        super().__init__()
        self._reader = TSReader(backend="pandas")
        self._writer = TSWriter(backend="pandas")

    def apply(self, data):
        """apply"""
        if "ts" in data:
            ts = data["ts"]
            ts_path = (Path(CACHE_DIR) / "predict_input" / "tmp_ts.csv").as_posix()
            self._writer.write(ts_path, ts)
            data["input_path"] = ts_path
            data["original_ts"] = ts
            return data

        elif "input_path" not in data:
            raise KeyError(f"Key {repr('input_path')} is required, but not found.")

        ts_path = data["input_path"]
        # XXX: auto download for url
        ts_path = self._download_from_url(ts_path)
        blob = self._reader.read(ts_path)

        data["input_path"] = ts_path
        data["ts"] = blob
        data["original_ts"] = blob
        return data

    def _download_from_url(self, in_path):
        if in_path.startswith("http"):
            file_name = Path(in_path).name
            save_path = Path(CACHE_DIR) / "predict_input" / file_name
            download(in_path, save_path, overwrite=True)
            return save_path.as_posix()
        return in_path

    @classmethod
    def get_input_keys(cls):
        """get input keys"""
        # input_path: Path of the image.
        return [["input_path"], ["ts"]]

    @classmethod
    def get_output_keys(cls):
        """get output keys"""
        # image: Image in hw or hwc format.
        # original_image: Original image in hw or hwc format.
        # original_image_size: Width and height of the original image.
        return ["ts", "original_ts"]


class TSCutOff(BaseTransform):
    """Reorder the dimensions of the image from HWC to CHW."""

    def __init__(self, size):
        super().__init__()
        self.size = size

    def apply(self, data):
        df = data["ts"].copy()
        skip_len = self.size.get("skip_chunk_len", 0)
        if len(df) < self.size["in_chunk_len"] + skip_len:
            raise ValueError(
                f"The length of the input data is {len(df)}, but it should be at least {self.size['in_chunk_len'] + self.size['skip_chunk_len']} for training."
            )

        df = df[-(self.size["in_chunk_len"] + skip_len) :]
        data["ts"] = df
        data["original_ts"] = df
        return data

    @classmethod
    def get_input_keys(cls):
        """get input keys"""
        # image: Image in hwc format.
        return ["ts"]

    @classmethod
    def get_output_keys(cls):
        """get output keys"""
        # image: Image in chw format.
        return ["ts"]


class TSNormalize(BaseTransform):
    """Flip the image vertically or horizontally."""

    def __init__(self, scale_path, params_info):
        """
        Initialize the instance.

        Args:
            mode (str, optional): 'H' for horizontal flipping and 'V' for vertical
                flipping. Default: 'H'.
        """
        super().__init__()
        self.scaler = joblib.load(scale_path)
        self.params_info = params_info

    def apply(self, data):
        """apply"""
        df = data["ts"].copy()
        if self.params_info.get("target_cols", None) is not None:
            df[self.params_info["target_cols"]] = self.scaler.transform(
                df[self.params_info["target_cols"]]
            )
        if self.params_info.get("feature_cols", None) is not None:
            df[self.params_info["feature_cols"]] = self.scaler.transform(
                df[self.params_info["feature_cols"]]
            )

        data["ts"] = df
        return data

    @classmethod
    def get_input_keys(cls):
        """get input keys"""
        # image: Image in hw or hwc format.
        return ["ts"]

    @classmethod
    def get_output_keys(cls):
        """get output keys"""
        # image: Image in hw or hwc format.
        return ["ts"]


class TSDeNormalize(BaseTransform):
    """Flip the image vertically or horizontally."""

    def __init__(self, scale_path, params_info):
        """
        Initialize the instance.

        Args:
            mode (str, optional): 'H' for horizontal flipping and 'V' for vertical
                flipping. Default: 'H'.
        """
        super().__init__()
        self.scaler = joblib.load(scale_path)
        self.params_info = params_info

    def apply(self, data):
        """apply"""
        future_target = data["pred_ts"].copy()
        scale_cols = future_target.columns.values.tolist()
        future_target[scale_cols] = self.scaler.inverse_transform(
            future_target[scale_cols]
        )
        data["pred_ts"] = future_target
        return data

    @classmethod
    def get_input_keys(cls):
        """get input keys"""
        # image: Image in hw or hwc format.
        return ["pred_ts"]

    @classmethod
    def get_output_keys(cls):
        """get output keys"""
        # image: Image in hw or hwc format.
        return ["pred_ts"]


class BuildTSDataset(BaseTransform):
    """bulid the ts."""

    def __init__(self, params_info):
        """
        Initialize the instance.

        Args:
            mode (str, optional): 'H' for horizontal flipping and 'V' for vertical
                flipping. Default: 'H'.
        """
        super().__init__()
        self.params_info = params_info

    def apply(self, data):
        """apply"""
        df = data["ts"].copy()
        tsdata = load_from_dataframe(df, **self.params_info)
        data["ts"] = tsdata
        data["original_ts"] = tsdata
        return data

    @classmethod
    def get_input_keys(cls):
        """get input keys"""
        # image: Image in hw or hwc format.
        return ["ts"]

    @classmethod
    def get_output_keys(cls):
        """get output keys"""
        # image: Image in hw or hwc format.
        return ["ts"]


class TimeFeature(BaseTransform):
    """Normalize the image."""

    def __init__(self, params_info, size, holiday=False):
        """
        Initialize the instance.
        """
        super().__init__()
        self.freq = params_info["freq"]
        self.size = size
        self.holiday = holiday

    def apply(self, data):
        """apply"""
        ts = data["ts"].copy()
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
        data["ts"] = ts
        return data

    @classmethod
    def get_input_keys(cls):
        """get input keys"""
        # image: Image in hw or hwc format.
        return ["ts"]

    @classmethod
    def get_output_keys(cls):
        """get output keys"""
        # image: Image in hw or hwc format.
        return ["ts"]


class BuildPadMask(BaseTransform):

    def __init__(self, input_data):

        super().__init__()
        self.input_data = input_data

    def apply(self, data):
        """apply"""
        df = data["ts"].copy()

        if "features" in self.input_data:
            df["features"] = df["past_target"]

        if "pad_mask" in self.input_data:
            target_dim = len(df["features"])
            max_length = self.input_data["pad_mask"][-1]
            if max_length > 0:
                ones = np.ones(max_length, dtype=np.int32)
                if max_length != target_dim:
                    target_ndarray = np.array(df["features"]).astype(np.float32)
                    target_ndarray_final = np.zeros(
                        [max_length, target_dim], dtype=np.int32
                    )
                    end = min(target_dim, max_length)
                    target_ndarray_final[:end, :] = target_ndarray
                    df["features"] = target_ndarray_final
                    ones[end:] = 0.0
                    df["pad_mask"] = ones
                else:
                    df["pad_mask"] = ones
        data["ts"] = df
        return data

    @classmethod
    def get_input_keys(cls):
        """get input keys"""
        # image: Image in hw or hwc format.
        return ["ts"]

    @classmethod
    def get_output_keys(cls):
        """get output keys"""
        # image: Image in hw or hwc format.
        return ["ts"]


class TStoArray(BaseTransform):

    def __init__(self, input_data):

        super().__init__()
        self.input_data = input_data

    def apply(self, data):
        """apply"""
        df = data["ts"].copy()
        ts_list = []
        input_name = list(self.input_data.keys())
        input_name.sort()
        for key in input_name:
            ts_list.append(np.array(df[key]).astype("float32"))

        data["ts"] = ts_list
        return data

    @classmethod
    def get_input_keys(cls):
        """get input keys"""
        # image: Image in hw or hwc format.
        return ["ts"]

    @classmethod
    def get_output_keys(cls):
        """get output keys"""
        # image: Image in hw or hwc format.
        return ["ts"]


class ArraytoTS(BaseTransform):

    def __init__(self, info_params):

        super().__init__()
        self.info_params = info_params

    def apply(self, data):
        """apply"""
        output_data = data["pred_ts"].copy()
        if data["original_ts"].get("past_target", None) is not None:
            ts = data["original_ts"]["past_target"]
        elif data["original_ts"].get("observed_cov_numeric", None) is not None:
            ts = data["original_ts"]["observed_cov_numeric"]
        elif data["original_ts"].get("known_cov_numeric", None) is not None:
            ts = data["original_ts"]["known_cov_numeric"]
        elif data["original_ts"].get("static_cov_numeric", None) is not None:
            ts = data["original_ts"]["static_cov_numeric"]
        else:
            raise ValueError("No value in original_ts")

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
                periods=output_data.shape[0],
                freq=self.info_params["freq"],
                name=self.info_params["time_col"],
            )
        elif isinstance(self.info_params["freq"], int):
            start_idx = max(ts.index) + 1
            stop_idx = start_idx + output_data.shape[0]
            future_target_index = pd.RangeIndex(
                start=start_idx,
                stop=stop_idx,
                step=self.info_params["freq"],
                name=self.info_params["time_col"],
            )

        future_target = pd.DataFrame(
            np.reshape(output_data, newshape=[output_data.shape[0], -1]),
            index=future_target_index,
            columns=column_name,
        )
        data["pred_ts"] = future_target
        return data

    @classmethod
    def get_input_keys(cls):
        """get input keys"""
        # image: Image in hw or hwc format.
        return ["pred_ts"]

    @classmethod
    def get_output_keys(cls):
        """get output keys"""
        # image: Image in hw or hwc format.
        return ["pred_ts"]


class GetAnomaly(BaseTransform):

    def __init__(self, model_threshold, info_params):

        super().__init__()
        self.model_threshold = model_threshold
        self.info_params = info_params

    def apply(self, data):
        """apply"""
        output_data = data["pred_ts"].copy()
        if data["original_ts"].get("past_target", None) is not None:
            ts = data["original_ts"]["past_target"]
        elif data["original_ts"].get("observed_cov_numeric", None) is not None:
            ts = data["original_ts"]["observed_cov_numeric"]
        elif data["original_ts"].get("known_cov_numeric", None) is not None:
            ts = data["original_ts"]["known_cov_numeric"]
        elif data["original_ts"].get("static_cov_numeric", None) is not None:
            ts = data["original_ts"]["static_cov_numeric"]
        else:
            raise ValueError("No value in original_ts")
        column_name = (
            self.info_params["target_cols"]
            if "target_cols" in self.info_params
            else self.info_params["feature_cols"]
        )

        anomaly_score = np.mean(np.square(output_data - np.array(ts)), axis=-1)
        anomaly_label = (anomaly_score >= self.model_threshold) + 0

        past_target_index = ts.index
        past_target_index.name = self.info_params["time_col"]
        anomaly_label = pd.DataFrame(
            np.reshape(anomaly_label, newshape=[output_data.shape[0], -1]),
            index=past_target_index,
            columns=["label"],
        )
        data["pred_ts"] = anomaly_label
        return data

    @classmethod
    def get_input_keys(cls):
        """get input keys"""
        # image: Image in hw or hwc format.
        return ["pred_ts"]

    @classmethod
    def get_output_keys(cls):
        """get output keys"""
        # image: Image in hw or hwc format.
        return ["pred_ts"]
