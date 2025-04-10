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

from typing import List, Optional, Union, Dict

import chinese_calendar
import joblib
import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset, Easter, Day
from pandas.tseries import holiday as hd
from sklearn.preprocessing import StandardScaler

from ..base import PyOnlyProcessor

__all__ = [
    "CutOff",
    "Normalize",
    "Denormalize",
    "BuildTSDataset",
    "CalcTimeFeatures",
    "BuildPaddedMask",
    "DataFrame2Arrays",
]

_MAX_WINDOW = 183 + 17
_EASTER_SUNDAY = hd.Holiday("Easter Sunday", month=1, day=1, offset=[Easter(), Day(0)])
_NEW_YEARS_DAY = hd.Holiday("New Years Day", month=1, day=1)
_SUPER_BOWL = hd.Holiday(
    "Superbowl", month=2, day=1, offset=DateOffset(weekday=hd.SU(1))
)
_MOTHERS_DAY = hd.Holiday(
    "Mothers Day", month=5, day=1, offset=DateOffset(weekday=hd.SU(2))
)
_INDEPENDENCE_DAY = hd.Holiday("Independence Day", month=7, day=4)
_CHRISTMAS_EVE = hd.Holiday("Christmas", month=12, day=24)
_CHRISTMAS_DAY = hd.Holiday("Christmas", month=12, day=25)
_NEW_YEARS_EVE = hd.Holiday("New Years Eve", month=12, day=31)
_BLACK_FRIDAY = hd.Holiday(
    "Black Friday",
    month=11,
    day=1,
    offset=[pd.DateOffset(weekday=hd.TH(4)), Day(1)],
)
_CYBER_MONDAY = hd.Holiday(
    "Cyber Monday",
    month=11,
    day=1,
    offset=[pd.DateOffset(weekday=hd.TH(4)), Day(4)],
)

_HOLYDAYS = [
    hd.EasterMonday,
    hd.GoodFriday,
    hd.USColumbusDay,
    hd.USLaborDay,
    hd.USMartinLutherKingJr,
    hd.USMemorialDay,
    hd.USPresidentsDay,
    hd.USThanksgivingDay,
    _EASTER_SUNDAY,
    _NEW_YEARS_DAY,
    _SUPER_BOWL,
    _MOTHERS_DAY,
    _INDEPENDENCE_DAY,
    _CHRISTMAS_EVE,
    _CHRISTMAS_DAY,
    _NEW_YEARS_EVE,
    _BLACK_FRIDAY,
    _CYBER_MONDAY,
]


def _cal_year(
    x: np.datetime64,
):
    return x.year


def _cal_month(
    x: np.datetime64,
):
    return x.month


def _cal_day(
    x: np.datetime64,
):
    return x.day


def _cal_hour(
    x: np.datetime64,
):
    return x.hour


def _cal_weekday(
    x: np.datetime64,
):
    return x.dayofweek


def _cal_quarter(
    x: np.datetime64,
):
    return x.quarter


def _cal_hourofday(
    x: np.datetime64,
):
    return x.hour / 23.0 - 0.5


def _cal_dayofweek(
    x: np.datetime64,
):
    return x.dayofweek / 6.0 - 0.5


def _cal_dayofmonth(
    x: np.datetime64,
):
    return x.day / 30.0 - 0.5


def _cal_dayofyear(
    x: np.datetime64,
):
    return x.dayofyear / 364.0 - 0.5


def _cal_weekofyear(
    x: np.datetime64,
):
    return x.weekofyear / 51.0 - 0.5


def _cal_holiday(
    x: np.datetime64,
):
    return float(chinese_calendar.is_holiday(x))


def _cal_workday(
    x: np.datetime64,
):
    return float(chinese_calendar.is_workday(x))


def _cal_minuteofhour(
    x: np.datetime64,
):
    return x.minute / 59 - 0.5


def _cal_monthofyear(
    x: np.datetime64,
):
    return x.month / 11.0 - 0.5


_CAL_DATE_METHOD = {
    "year": _cal_year,
    "month": _cal_month,
    "day": _cal_day,
    "hour": _cal_hour,
    "weekday": _cal_weekday,
    "quarter": _cal_quarter,
    "minuteofhour": _cal_minuteofhour,
    "monthofyear": _cal_monthofyear,
    "hourofday": _cal_hourofday,
    "dayofweek": _cal_dayofweek,
    "dayofmonth": _cal_dayofmonth,
    "dayofyear": _cal_dayofyear,
    "weekofyear": _cal_weekofyear,
    "is_holiday": _cal_holiday,
    "is_workday": _cal_workday,
}


def _load_from_one_dataframe(
    data: Union[pd.DataFrame, pd.Series],
    time_col: Optional[str] = None,
    value_cols: Optional[Union[List[str], str]] = None,
    freq: Optional[Union[str, int]] = None,
    drop_tail_nan: bool = False,
    dtype: Optional[Union[type, Dict[str, type]]] = None,
):
    series_data = None
    if value_cols is None:
        if isinstance(data, pd.Series):
            series_data = data.copy()
        else:
            series_data = data.loc[:, data.columns != time_col].copy()
    else:
        series_data = data.loc[:, value_cols].copy()

    if time_col:
        if time_col not in data.columns:
            raise ValueError(
                "The time column: {} doesn't exist in the `data`!".format(time_col)
            )
        time_col_vals = data.loc[:, time_col]
    else:
        time_col_vals = data.index

    if np.issubdtype(time_col_vals.dtype, np.integer) and isinstance(freq, str):
        time_col_vals = time_col_vals.astype(str)

    if np.issubdtype(time_col_vals.dtype, np.integer):
        if freq:
            if not isinstance(freq, int) or freq < 1:
                raise ValueError(
                    "The type of `freq` should be `int` when the type of `time_col` is `RangeIndex`."
                )
        else:
            freq = 1
        start_idx, stop_idx = min(time_col_vals), max(time_col_vals) + freq
        if (stop_idx - start_idx) / freq != len(data):
            raise ValueError("The number of rows doesn't match with the RangeIndex!")
        time_index = pd.RangeIndex(start=start_idx, stop=stop_idx, step=freq)
    elif np.issubdtype(time_col_vals.dtype, np.object_) or np.issubdtype(
        time_col_vals.dtype, np.datetime64
    ):
        time_col_vals = pd.to_datetime(time_col_vals, infer_datetime_format=True)
        time_index = pd.DatetimeIndex(time_col_vals)
        if freq:
            if not isinstance(freq, str):
                raise ValueError(
                    "The type of `freq` should be `str` when the type of `time_col` is `DatetimeIndex`."
                )
        else:
            # If freq is not provided and automatic inference fail, throw exception
            freq = pd.infer_freq(time_index)
            if freq is None:
                raise ValueError(
                    "Failed to infer the `freq`. A valid `freq` is required."
                )
            if freq[0] == "-":
                freq = freq[1:]
    else:
        raise ValueError("The type of `time_col` is invalid.")
    if isinstance(series_data, pd.Series):
        series_data = series_data.to_frame()
    series_data.set_index(time_index, inplace=True)
    series_data.sort_index(inplace=True)
    return series_data


def _load_from_dataframe(
    df: pd.DataFrame,
    group_id: str = None,
    time_col: Optional[str] = None,
    target_cols: Optional[Union[List[str], str]] = None,
    label_col: Optional[Union[List[str], str]] = None,
    observed_cov_cols: Optional[Union[List[str], str]] = None,
    feature_cols: Optional[Union[List[str], str]] = None,
    known_cov_cols: Optional[Union[List[str], str]] = None,
    static_cov_cols: Optional[Union[List[str], str]] = None,
    freq: Optional[Union[str, int]] = None,
    fill_missing_dates: bool = False,
    fillna_method: str = "pre",
    fillna_window_size: int = 10,
    **kwargs,
):
    dfs = []  # seperate multiple group
    if group_id is not None:
        group_unique = df[group_id].unique()
        for column in group_unique:
            dfs.append(df[df[group_id].isin([column])])
    else:
        dfs = [df]
    res = []
    if label_col:
        if isinstance(label_col, str) and len(label_col) > 1:
            raise ValueError("The length of label_col must be 1.")
        target_cols = label_col
    if feature_cols:
        observed_cov_cols = feature_cols
    for df in dfs:
        target = None
        observed_cov = None
        known_cov = None
        static_cov = dict()
        if not any([target_cols, observed_cov_cols, known_cov_cols, static_cov_cols]):
            target = _load_from_one_dataframe(
                df,
                time_col,
                [a for a in df.columns if a != time_col],
                freq,
            )

        else:
            if target_cols:
                target = _load_from_one_dataframe(
                    df,
                    time_col,
                    target_cols,
                    freq,
                )

            if observed_cov_cols:
                observed_cov = _load_from_one_dataframe(
                    df,
                    time_col,
                    observed_cov_cols,
                    freq,
                )

            if known_cov_cols:
                known_cov = _load_from_one_dataframe(
                    df,
                    time_col,
                    known_cov_cols,
                    freq,
                )

            if static_cov_cols:
                if isinstance(static_cov_cols, str):
                    static_cov_cols = [static_cov_cols]
                for col in static_cov_cols:
                    if col not in df.columns or len(np.unique(df[col])) != 1:
                        raise ValueError(
                            "static cov cals data is not in columns or schema is not right!"
                        )
                    static_cov[col] = df[col].iloc[0]
        res.append(
            {
                "past_target": target,
                "observed_cov_numeric": observed_cov,
                "known_cov_numeric": known_cov,
                "static_cov_numeric": static_cov,
            }
        )
    return res[0]


def _distance_to_holiday(holiday):
    def _distance_to_day(index):
        holiday_date = holiday.dates(
            index - pd.Timedelta(days=_MAX_WINDOW),
            index + pd.Timedelta(days=_MAX_WINDOW),
        )
        assert (
            len(holiday_date) != 0
        ), f"No closest holiday for the date index {index} found."
        # It sometimes returns two dates if it is exactly half a year after the
        # holiday. In this case, the smaller distance (182 days) is returned.
        return float((index - holiday_date[0]).days)

    return _distance_to_day


def _to_time_features(
    dataset, freq, feature_cols, extend_points, inplace: bool = False
):
    new_ts = dataset
    if not inplace:
        new_ts = dataset.copy()
    # Get known_cov
    kcov = new_ts["known_cov_numeric"]
    if not kcov:
        tf_kcov = new_ts["past_target"].index.to_frame()
    else:
        tf_kcov = kcov.index.to_frame()
    time_col = tf_kcov.columns[0]
    if np.issubdtype(tf_kcov[time_col].dtype, np.integer):
        raise ValueError(
            "The time_col can't be the type of numpy.integer, and it must be the type of numpy.datetime64"
        )
    if not kcov:
        freq = freq if freq is not None else pd.infer_freq(tf_kcov[time_col])
        extend_time = pd.date_range(
            start=tf_kcov[time_col][-1],
            freq=freq,
            periods=extend_points + 1,
            closed="right",
            name=time_col,
        ).to_frame()
        tf_kcov = pd.concat([tf_kcov, extend_time])

    for k in feature_cols:
        if k != "holidays":
            v = tf_kcov[time_col].apply(lambda x: _CAL_DATE_METHOD[k](x))
            v.index = tf_kcov[time_col]

            if new_ts["known_cov_numeric"] is None:
                new_ts["known_cov_numeric"] = pd.DataFrame(v.rename(k), index=v.index)
            else:
                new_ts["known_cov_numeric"][k] = v.rename(k).reindex(
                    new_ts["known_cov_numeric"].index
                )

        else:
            holidays_col = []
            for i, H in enumerate(_HOLYDAYS):
                v = tf_kcov[time_col].apply(_distance_to_holiday(H))
                v.index = tf_kcov[time_col]
                holidays_col.append(k + "_" + str(i))
                if new_ts["known_cov_numeric"] is None:
                    new_ts["known_cov_numeric"] = pd.DataFrame(
                        v.rename(k + "_" + str(i)), index=v.index
                    )
                else:
                    new_ts["known_cov_numeric"][k + "_" + str(i)] = v.rename(k).reindex(
                        new_ts["known_cov_numeric"].index
                    )

            scaler = StandardScaler()
            scaler.fit(new_ts["known_cov_numeric"][holidays_col])
            new_ts["known_cov_numeric"][holidays_col] = scaler.transform(
                new_ts["known_cov_numeric"][holidays_col]
            )
    return new_ts


class CutOff(PyOnlyProcessor):
    def __init__(self, size):
        super().__init__()
        self._size = size

    def __call__(self, data):
        ts = data["ts"]
        ori_ts = data["ori_ts"]

        skip_len = self._size.get("skip_chunk_len", 0)
        if len(ts) < self._size["in_chunk_len"] + skip_len:
            raise ValueError(
                f"The length of the input data is {len(ts)}, but it should be at least {self._size['in_chunk_len'] + self._size['skip_chunk_len']} for training."
            )
        ts_data = ts[-(self._size["in_chunk_len"] + skip_len) :]

        return {**data, "ts": ts_data, "ori_ts": ts_data}


class Normalize(PyOnlyProcessor):
    def __init__(self, scale_path, params_info):
        super().__init__()
        self._scaler = joblib.load(scale_path)
        self._params_info = params_info

    def __call__(self, data):
        ts = data["ts"]

        if self._params_info.get("target_cols", None) is not None:
            ts[self._params_info["target_cols"]] = self._scaler.transform(
                ts[self._params_info["target_cols"]]
            )
        if self._params_info.get("feature_cols", None) is not None:
            ts[self._params_info["feature_cols"]] = self._scaler.transform(
                ts[self._params_info["feature_cols"]]
            )

        return {**data, "ts": ts}


class Denormalize(PyOnlyProcessor):
    def __init__(self, scale_path, params_info):
        super().__init__()
        self._scaler = joblib.load(scale_path)
        self._params_info = params_info

    def __call__(self, data):
        pred = data["pred"]

        scale_cols = pred.columns.values.tolist()
        pred[scale_cols] = self._scaler.inverse_transform(pred[scale_cols])

        return {**data, "pred": pred}


class BuildTSDataset(PyOnlyProcessor):
    def __init__(self, params_info):
        super().__init__()
        self._params_info = params_info

    def __call__(self, data):
        ts = data["ts"]
        ori_ts = data["ori_ts"]

        ts_data = _load_from_dataframe(ts, **self._params_info)

        return {**data, "ts": ts_data, "ori_ts": ts_data}


class CalcTimeFeatures(PyOnlyProcessor):
    def __init__(self, params_info, size, holiday=False):
        super().__init__()
        self._freq = params_info["freq"]
        self._size = size
        self._holiday = holiday

    def __call__(self, data):
        ts = data["ts"]

        if not self._holiday:
            ts = _to_time_features(
                ts,
                self._freq,
                ["hourofday", "dayofmonth", "dayofweek", "dayofyear"],
                self._size["out_chunk_len"],
            )
        else:
            ts = _to_time_features(
                ts,
                self._freq,
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
                self._size["out_chunk_len"],
            )

        return {**data, "ts": ts}


class BuildPaddedMask(PyOnlyProcessor):
    def __init__(self, input_data):
        super().__init__()
        self._input_data = input_data

    def __call__(self, data):
        ts = data["ts"]

        if "features" in self._input_data:
            ts["features"] = ts["past_target"]

        if "pad_mask" in self._input_data:
            target_dim = len(ts["features"])
            max_length = self._input_data["pad_mask"][-1]
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

        return {**data, "ts": ts}


class DataFrame2Arrays(PyOnlyProcessor):
    def __init__(self, input_data):
        super().__init__()
        self._input_data = input_data

    def __call__(self, data):
        ts = data["ts"]

        ts_list = []
        input_name = list(self._input_data.keys())
        input_name.sort()
        for key in input_name:
            ts_list.append(np.array(ts[key]).astype("float32"))

        return {**data, "ts": ts_list}
