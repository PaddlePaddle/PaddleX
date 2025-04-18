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

import io
from typing import Any

import pandas as pd
from PIL import Image

from ....utils.deps import function_requires_deps, is_dep_available
from ...common.result import BaseTSResult

if is_dep_available("matplotlib"):
    import matplotlib.pyplot as plt


@function_requires_deps("matplotlib")
def visualize(forecast: pd.DataFrame, actual_data: pd.DataFrame) -> Image.Image:
    """
    Visualizes both the time series forecast and actual results, returning them as a Pillow image.

    Args:
        forecast (pd.DataFrame): The DataFrame containing the forecast data.
        actual_data (pd.Series): The actual observed data for comparison.
        title (str): The title of the plot.

    Returns:
        Image.Image: The visualized result as a Pillow image.
    """
    plt.figure(figsize=(12, 6))
    forecast_columns = forecast.columns
    index_name = forecast.index.name
    actual_data = actual_data.set_index(index_name)

    actual_data.index = actual_data.index.astype(str)
    forecast.index = forecast.index.astype(str)

    length = min(len(forecast), len(actual_data))
    actual_data = actual_data.tail(length)

    plt.plot(
        actual_data.index,
        actual_data[forecast_columns[0]],
        label="Actual Data",
        color="blue",
        linestyle="--",
    )
    plt.plot(
        forecast.index, forecast[forecast_columns[0]], label="Forecast", color="red"
    )

    plt.title("Time Series Forecast")
    plt.xlabel("Time")
    plt.ylabel(forecast_columns[0])
    plt.legend()
    plt.grid(True)
    plt.xticks(ticks=range(0, 2 * length, 10))
    plt.xticks(rotation=45)

    buf = io.BytesIO()
    plt.savefig(buf, bbox_inches="tight")
    buf.seek(0)
    plt.close()
    image = Image.open(buf)

    return image


class TSFcResult(BaseTSResult):
    """A class representing the result of a time series forecasting task."""

    def _to_img(self) -> Image.Image:
        """apply"""
        forecast = self["forecast"]
        ts_input = self["cutoff_ts"]
        return {"res": visualize(forecast, ts_input)}

    def _to_csv(self) -> Any:
        """
        Converts the forecasting results to a CSV format.

        Returns:
            Any: The forecast data formatted for CSV output, typically a DataFrame or similar structure.
        """
        return {"res": self["forecast"]}
