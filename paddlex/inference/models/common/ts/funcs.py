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


from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from packaging.version import Version
from pandas.tseries import holiday as hd
from pandas.tseries.offsets import DateOffset, Day, Easter

from .....utils.deps import function_requires_deps, get_dep_version, is_dep_available

if is_dep_available("chinese-calendar"):
    import chinese_calendar
if is_dep_available("scikit-learn"):
    from sklearn.preprocessing import StandardScaler

MAX_WINDOW = 183 + 17
EasterSunday = hd.Holiday("Easter Sunday", month=1, day=1, offset=[Easter(), Day(0)])
NewYearsDay = hd.Holiday("New Years Day", month=1, day=1)
SuperBowl = hd.Holiday("Superbowl", month=2, day=1, offset=DateOffset(weekday=hd.SU(1)))
MothersDay = hd.Holiday(
    "Mothers Day", month=5, day=1, offset=DateOffset(weekday=hd.SU(2))
)
IndependenceDay = hd.Holiday("Independence Day", month=7, day=4)
ChristmasEve = hd.Holiday("Christmas", month=12, day=24)
ChristmasDay = hd.Holiday("Christmas", month=12, day=25)
NewYearsEve = hd.Holiday("New Years Eve", month=12, day=31)
BlackFriday = hd.Holiday(
    "Black Friday",
    month=11,
    day=1,
    offset=[pd.DateOffset(weekday=hd.TH(4)), Day(1)],
)
CyberMonday = hd.Holiday(
    "Cyber Monday",
    month=11,
    day=1,
    offset=[pd.DateOffset(weekday=hd.TH(4)), Day(4)],
)

HOLIDAYS = [
    hd.EasterMonday,
    hd.GoodFriday,
    hd.USColumbusDay,
    hd.USLaborDay,
    hd.USMartinLutherKingJr,
    hd.USMemorialDay,
    hd.USPresidentsDay,
    hd.USThanksgivingDay,
    EasterSunday,
    NewYearsDay,
    SuperBowl,
    MothersDay,
    IndependenceDay,
    ChristmasEve,
    ChristmasDay,
    NewYearsEve,
    BlackFriday,
    CyberMonday,
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


@function_requires_deps("chinese-calendar")
def _cal_holiday(
    x: np.datetime64,
):
    return float(chinese_calendar.is_holiday(x))


@function_requires_deps("chinese-calendar")
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


CAL_DATE_METHOD = {
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


def load_from_one_dataframe(
    data: Union[pd.DataFrame, pd.Series],
    time_col: Optional[str] = None,
    value_cols: Optional[Union[List[str], str]] = None,
    freq: Optional[Union[str, int]] = None,
    drop_tail_nan: bool = False,
    dtype: Optional[Union[type, Dict[str, type]]] = None,
) -> pd.DataFrame:
    """Transforms a DataFrame or Series into a time-indexed DataFrame.

    Args:
        data (Union[pd.DataFrame, pd.Series]): The input data containing time series information.
        time_col (Optional[str]): The column name representing time information. If None, uses the index.
        value_cols (Optional[Union[List[str], str]]): Columns to extract as values. If None, uses all except time_col.
        freq (Optional[Union[str, int]]): The frequency of the time series data.
        drop_tail_nan (bool): If True, drop trailing NaN values from the data.
        dtype (Optional[Union[type, Dict[str, type]]]): Enforce a specific data type on the resulting DataFrame.

    Returns:
        pd.DataFrame: A DataFrame with time as the index and specified value columns.

    Raises:
        ValueError: If the time column doesn't exist, or if frequency cannot be inferred.

    """
    # Initialize series_data with specified value columns or all except time_col
    series_data = None
    if value_cols is None:
        if isinstance(data, pd.Series):
            series_data = data.copy()
        else:
            series_data = data.loc[:, data.columns != time_col].copy()
    else:
        series_data = data.loc[:, value_cols].copy()

    # Determine the time column values
    if time_col:
        if time_col not in data.columns:
            raise ValueError(
                "The time column: {} doesn't exist in the `data`!".format(time_col)
            )
        time_col_vals = data.loc[:, time_col]
    else:
        time_col_vals = data.index

    # Handle integer-based time column values when frequency is a string
    if np.issubdtype(time_col_vals.dtype, np.integer) and isinstance(freq, str):
        time_col_vals = time_col_vals.astype(str)

    # Process integer-based time column values
    if np.issubdtype(time_col_vals.dtype, np.integer):
        if freq:
            if not isinstance(freq, int) or freq < 1:
                raise ValueError(
                    "The type of `freq` should be `int` when the type of `time_col` is `RangeIndex`."
                )
        else:
            freq = 1  # Default frequency for integer index
        start_idx, stop_idx = min(time_col_vals), max(time_col_vals) + freq
        if (stop_idx - start_idx) / freq != len(data):
            raise ValueError("The number of rows doesn't match with the RangeIndex!")
        time_index = pd.RangeIndex(start=start_idx, stop=stop_idx, step=freq)

    # Process datetime-like time column values
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
            # Attempt to infer frequency if not provided
            freq = pd.infer_freq(time_index)
            if freq is None:
                raise ValueError(
                    "Failed to infer the `freq`. A valid `freq` is required."
                )
            if freq[0] == "-":
                freq = freq[1:]

    # Raise error for unsupported time column types
    else:
        raise ValueError("The type of `time_col` is invalid.")

    # Ensure series_data is a DataFrame
    if isinstance(series_data, pd.Series):
        series_data = series_data.to_frame()

    # Set time index and sort data
    series_data.set_index(time_index, inplace=True)
    series_data.sort_index(inplace=True)
    return series_data


def load_from_dataframe(
    df: pd.DataFrame,
    group_id: Optional[str] = None,
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
) -> Dict[str, Optional[Union[pd.DataFrame, Dict[str, any]]]]:
    """Loads and processes time series data from a DataFrame.

    This function extracts and organizes time series data from a given DataFrame.
    It supports optional grouping and extraction of specific columns as features.

    Args:
        df (pd.DataFrame): The input DataFrame containing time series data.
        group_id (Optional[str]): Column name used for grouping the data.
        time_col (Optional[str]): Name of the time column.
        target_cols (Optional[Union[List[str], str]]): Columns to be used as target.
        label_col (Optional[Union[List[str], str]]): Columns to be used as label.
        observed_cov_cols (Optional[Union[List[str], str]]): Columns for observed covariates.
        feature_cols (Optional[Union[List[str], str]]): Columns to be used as features.
        known_cov_cols (Optional[Union[List[str], str]]): Columns for known covariates.
        static_cov_cols (Optional[Union[List[str], str]]): Columns for static covariates.
        freq (Optional[Union[str, int]]): Frequency of the time series data.
        fill_missing_dates (bool): Whether to fill missing dates in the time series.
        fillna_method (str): Method to fill missing values ('pre' or 'post').
        fillna_window_size (int): Window size for filling missing values.
        **kwargs: Additional keyword arguments.

    Returns:
        Dict[str, Optional[Union[pd.DataFrame, Dict[str, any]]]]: A dictionary containing processed time series data.
    """
    # List to store DataFrames if grouping is applied
    dfs = []

    # Separate the DataFrame into groups if group_id is provided
    if group_id is not None:
        group_unique = df[group_id].unique()
        for column in group_unique:
            dfs.append(df[df[group_id].isin([column])])
    else:
        dfs = [df]

    # Result list to store processed data from each group
    res = []

    # If label_col is provided, ensure it is a single column
    if label_col:
        if isinstance(label_col, str) and len(label_col) > 1:
            raise ValueError("The length of label_col must be 1.")
        target_cols = label_col

    # If feature_cols is provided, treat it as observed_cov_cols
    if feature_cols:
        observed_cov_cols = feature_cols

    # Process each DataFrame in the list
    for df in dfs:
        target = None
        observed_cov = None
        known_cov = None
        static_cov = dict()

        # If no specific columns are provided, use all columns except time_col
        if not any([target_cols, observed_cov_cols, known_cov_cols, static_cov_cols]):
            target = load_from_one_dataframe(
                df,
                time_col,
                [a for a in df.columns if a != time_col],
                freq,
            )
        else:
            if target_cols:
                target = load_from_one_dataframe(
                    df,
                    time_col,
                    target_cols,
                    freq,
                )

            if observed_cov_cols:
                observed_cov = load_from_one_dataframe(
                    df,
                    time_col,
                    observed_cov_cols,
                    freq,
                )

            if known_cov_cols:
                known_cov = load_from_one_dataframe(
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
                            "Static covariate columns data is not in columns or schema is not correct!"
                        )
                    static_cov[col] = df[col].iloc[0]
        # Append the processed data into the results list
        res.append(
            {
                "past_target": target,
                "observed_cov_numeric": observed_cov,
                "known_cov_numeric": known_cov,
                "static_cov_numeric": static_cov,
            }
        )
    # Return the first processed result
    return res[0]


def _distance_to_holiday(holiday) -> Callable[[pd.Timestamp], float]:
    """Creates a function to calculate the distance in days to the nearest holiday.

    This function generates a closure that computes the number of days from
    a given date index to the nearest holiday within a defined window.

    Args:
        holiday: An object that provides a `dates` method, which returns the
            dates of holidays within a specified range.

    Returns:
        Callable[[pd.Timestamp], float]: A function that takes a date index
        as input and returns the distance in days to the nearest holiday.
    """

    def _distance_to_day(index: pd.Timestamp) -> float:
        """Calculates the distance in days from a given date index to the nearest holiday.

        Args:
            index (pd.Timestamp): The date index for which the distance to the
                nearest holiday should be calculated.

        Returns:
            float: The number of days to the nearest holiday.

        Raises:
            AssertionError: If no holiday is found within the specified window.
        """
        holiday_date = holiday.dates(
            index - pd.Timedelta(days=MAX_WINDOW),
            index + pd.Timedelta(days=MAX_WINDOW),
        )
        assert (
            len(holiday_date) != 0
        ), f"No closest holiday for the date index {index} found."
        # It sometimes returns two dates if it is exactly half a year after the
        # holiday. In this case, the smaller distance (182 days) is returned.
        return float((index - holiday_date[0]).days)

    return _distance_to_day


@function_requires_deps("scikit-learn")
def time_feature(
    dataset: Dict,
    freq: Optional[Union[str, int]],
    feature_cols: List[str],
    extend_points: int,
    inplace: bool = False,
) -> Dict:
    """Transforms the time column of a dataset into time features.

    This function extracts time-related features from the time column in a
    dataset, optionally extending the time series for future points and
    normalizing holiday distances.

    Args:
        dataset (Dict): Dataset to be transformed.
        freq: Optional[Union[str, int]]: Frequency of the time series data. If not provided,
            the frequency will be inferred.
        feature_cols (List[str]): List of feature columns to be extracted.
        extend_points (int): Number of future points to extend the time series.
        inplace (bool): Whether to perform the transformation inplace. Default is False.

    Returns:
        Dict: The transformed dataset with time features added.

    Raises:
        ValueError: If the time column is of an integer type instead of datetime.
    """
    new_ts = dataset
    if not inplace:
        new_ts = dataset.copy()
    # Get known_cov_numeric or initialize with past target index
    kcov = new_ts["known_cov_numeric"]
    if not kcov:
        tf_kcov = new_ts["past_target"].index.to_frame()
    else:
        tf_kcov = kcov.index.to_frame()
    time_col = tf_kcov.columns[0]
    # Check if time column is of datetime type
    if np.issubdtype(tf_kcov[time_col].dtype, np.integer):
        raise ValueError(
            "The time_col can't be the type of numpy.integer, and it must be the type of numpy.datetime64"
        )
    # Extend the time series if no known_cov_numeric
    if not kcov:
        freq = freq if freq is not None else pd.infer_freq(tf_kcov[time_col])
        pd_version = get_dep_version("pandas")
        if Version(pd_version) >= Version("1.4"):
            extend_time = pd.date_range(
                start=tf_kcov[time_col][-1],
                freq=freq,
                periods=extend_points + 1,
                inclusive="right",
                name=time_col,
            ).to_frame()
        else:
            extend_time = pd.date_range(
                start=tf_kcov[time_col][-1],
                freq=freq,
                periods=extend_points + 1,
                closed="right",
                name=time_col,
            ).to_frame()
        tf_kcov = pd.concat([tf_kcov, extend_time])

    # Extract and add time features to known_cov_numeric
    for k in feature_cols:
        if k != "holidays":
            v = tf_kcov[time_col].apply(lambda x: CAL_DATE_METHOD[k](x))
            v.index = tf_kcov[time_col]

            if new_ts["known_cov_numeric"] is None:
                new_ts["known_cov_numeric"] = pd.DataFrame(v.rename(k), index=v.index)
            else:
                new_ts["known_cov_numeric"][k] = v.rename(k).reindex(
                    new_ts["known_cov_numeric"].index
                )
        else:
            holidays_col = []
            for i, H in enumerate(HOLIDAYS):
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
