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
from copy import deepcopy
import joblib
import numpy as np
import pandas as pd

from .....utils.download import download
from .....utils.cache import CACHE_DIR
from ....utils.io.readers import TSReader
from ....utils.io.writers import TSWriter
from ...base import BaseComponent
from ..read_data import _BaseRead
from .funcs import load_from_dataframe, time_feature


__all__ = [
    "ReadTS",
    "BuildTSDataset",
    "TSCutOff",
    "TSNormalize",
    "TimeFeature",
    "TStoArray",
    "BuildPadMask",
    "ArraytoTS",
    "TSDeNormalize",
    "GetAnomaly",
    "GetCls",
]


class ReadTS(_BaseRead):

    INPUT_KEYS = ["ts"]
    OUTPUT_KEYS = ["ts_path", "ts", "ori_ts"]
    DEAULT_INPUTS = {"ts": "ts"}
    DEAULT_OUTPUTS = {"ts_path": "ts_path", "ts": "ts", "ori_ts": "ori_ts"}

    SUFFIX = ["csv"]

    def __init__(self, batch_size=1):
        super().__init__(batch_size)
        self._reader = TSReader(backend="pandas")
        self._writer = TSWriter(backend="pandas")

    def apply(self, ts):
        if not isinstance(ts, str):
            ts_path = (Path(CACHE_DIR) / "predict_input" / "tmp_ts.csv").as_posix()
            self._writer.write(ts_path, ts)
            return {"ts_path": ts_path, "ts": ts, "ori_ts": deepcopy(ts)}

        ts_path = ts
        ts_path = self._download_from_url(ts_path)
        file_list = self._get_files_list(ts_path)
        batch = []
        for ts_path in file_list:
            ts_data = self._reader.read(ts_path)
            batch.append(
                {"ts_path": ts_path, "ts": ts_data, "ori_ts": deepcopy(ts_data)}
            )
            if len(batch) >= self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0:
            yield batch


class TSCutOff(BaseComponent):

    INPUT_KEYS = ["ts", "ori_ts"]
    OUTPUT_KEYS = ["ts", "ori_ts"]
    DEAULT_INPUTS = {"ts": "ts", "ori_ts": "ori_ts"}
    DEAULT_OUTPUTS = {"ts": "ts", "ori_ts": "ori_ts"}

    def __init__(self, size):
        super().__init__()
        self.size = size

    def apply(self, ts, ori_ts):
        skip_len = self.size.get("skip_chunk_len", 0)
        if len(ts) < self.size["in_chunk_len"] + skip_len:
            raise ValueError(
                f"The length of the input data is {len(ts)}, but it should be at least {self.size['in_chunk_len'] + self.size['skip_chunk_len']} for training."
            )
        ts_data = ts[-(self.size["in_chunk_len"] + skip_len) :]
        return {"ts": ts_data, "ori_ts": ts_data}


class TSNormalize(BaseComponent):

    INPUT_KEYS = ["ts"]
    OUTPUT_KEYS = ["ts"]
    DEAULT_INPUTS = {"ts": "ts"}
    DEAULT_OUTPUTS = {"ts": "ts"}

    def __init__(self, scale_path, params_info):
        super().__init__()
        self.scaler = joblib.load(scale_path)
        self.params_info = params_info

    def apply(self, ts):
        """apply"""
        if self.params_info.get("target_cols", None) is not None:
            ts[self.params_info["target_cols"]] = self.scaler.transform(
                ts[self.params_info["target_cols"]]
            )
        if self.params_info.get("feature_cols", None) is not None:
            ts[self.params_info["feature_cols"]] = self.scaler.transform(
                ts[self.params_info["feature_cols"]]
            )

        return {"ts": ts}


class TSDeNormalize(BaseComponent):

    INPUT_KEYS = ["pred"]
    OUTPUT_KEYS = ["pred"]
    DEAULT_INPUTS = {"pred": "pred"}
    DEAULT_OUTPUTS = {"pred": "pred"}

    def __init__(self, scale_path, params_info):
        super().__init__()
        self.scaler = joblib.load(scale_path)
        self.params_info = params_info

    def apply(self, pred):
        """apply"""
        scale_cols = pred.columns.values.tolist()
        pred[scale_cols] = self.scaler.inverse_transform(pred[scale_cols])
        return {"pred": pred}


class BuildTSDataset(BaseComponent):

    INPUT_KEYS = ["ts", "ori_ts"]
    OUTPUT_KEYS = ["ts", "ori_ts"]
    DEAULT_INPUTS = {"ts": "ts", "ori_ts": "ori_ts"}
    DEAULT_OUTPUTS = {"ts": "ts", "ori_ts": "ori_ts"}

    def __init__(self, params_info):
        super().__init__()
        self.params_info = params_info

    def apply(self, ts, ori_ts):
        """apply"""
        ts_data = load_from_dataframe(ts, **self.params_info)
        return {"ts": ts_data, "ori_ts": ts_data}


class TimeFeature(BaseComponent):

    INPUT_KEYS = ["ts"]
    OUTPUT_KEYS = ["ts"]
    DEAULT_INPUTS = {"ts": "ts"}
    DEAULT_OUTPUTS = {"ts": "ts"}

    def __init__(self, params_info, size, holiday=False):
        super().__init__()
        self.freq = params_info["freq"]
        self.size = size
        self.holiday = holiday

    def apply(self, ts):
        """apply"""
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
        return {"ts": ts}


class BuildPadMask(BaseComponent):

    INPUT_KEYS = ["ts"]
    OUTPUT_KEYS = ["ts"]
    DEAULT_INPUTS = {"ts": "ts"}
    DEAULT_OUTPUTS = {"ts": "ts"}

    def __init__(self, input_data):
        super().__init__()
        self.input_data = input_data

    def apply(self, ts):
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
        return {"ts": ts}


class TStoArray(BaseComponent):

    INPUT_KEYS = ["ts"]
    OUTPUT_KEYS = ["ts"]
    DEAULT_INPUTS = {"ts": "ts"}
    DEAULT_OUTPUTS = {"ts": "ts"}

    def __init__(self, input_data):
        super().__init__()
        self.input_data = input_data

    def apply(self, ts):
        ts_list = []
        input_name = list(self.input_data.keys())
        input_name.sort()
        for key in input_name:
            ts_list.append(np.array(ts[key]).astype("float32"))

        return {"ts": ts_list}


class ArraytoTS(BaseComponent):

    INPUT_KEYS = ["ori_ts", "pred"]
    OUTPUT_KEYS = ["pred"]
    DEAULT_INPUTS = {"ori_ts": "ori_ts", "pred": "pred"}
    DEAULT_OUTPUTS = {"pred": "pred"}

    def __init__(self, info_params):
        super().__init__()
        self.info_params = info_params

    def apply(self, ori_ts, pred):
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
        return {"pred": future_target}


class GetAnomaly(BaseComponent):

    INPUT_KEYS = ["ori_ts", "pred"]
    OUTPUT_KEYS = ["anomaly"]
    DEAULT_INPUTS = {"ori_ts": "ori_ts", "pred": "pred"}
    DEAULT_OUTPUTS = {"anomaly": "anomaly"}

    def __init__(self, model_threshold, info_params):
        super().__init__()
        self.model_threshold = model_threshold
        self.info_params = info_params

    def apply(self, ori_ts, pred):
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
        anomaly_label = pd.DataFrame(
            np.reshape(anomaly_label, newshape=[pred.shape[0], -1]),
            index=past_target_index,
            columns=["label"],
        )
        return {"anomaly": anomaly_label}


class GetCls(BaseComponent):

    INPUT_KEYS = ["pred"]
    OUTPUT_KEYS = ["classification"]
    DEAULT_INPUTS = {"pred": "pred"}
    DEAULT_OUTPUTS = {"classification": "classification"}

    def __init__(self):
        super().__init__()

    def apply(self, pred):
        pred_ts = pred[0]
        pred_ts -= np.max(pred_ts, axis=-1, keepdims=True)
        pred_ts = np.exp(pred_ts) / np.sum(np.exp(pred_ts), axis=-1, keepdims=True)
        classid = np.argmax(pred_ts, axis=-1)
        pred_score = pred_ts[classid]
        result = pd.DataFrame.from_dict({"classid": [classid], "score": [pred_score]})
        result.index.name = "sample"
        return {"classification": result}
