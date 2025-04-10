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

from PIL import Image

from ....utils.deps import function_requires_deps, is_dep_available
from ...common.result import BaseTSResult

if is_dep_available("matplotlib"):
    import matplotlib.pyplot as plt
import pandas as pd


@function_requires_deps("matplotlib")
def visualize(forecast: pd.DataFrame) -> Image.Image:
    """
    Visualizes both the time series forecast and actual results, returning them as a Pillow image.

    Args:
        forecast (pd.DataFrame): The DataFrame containing the forecast data.

    Returns:
        Image.Image: The visualized result as a Pillow image.
    """
    plt.figure(figsize=(12, 6))
    forecast_columns = forecast.columns
    forecast.index.name
    forecast.index = forecast.index.astype(str)

    plt.step(
        forecast.index,
        forecast[forecast_columns[0]],
        where="post",
        label="Anomaly",
        color="red",
    )
    plt.title("Time Series Anomaly Detection")
    plt.xlabel("Time")
    plt.ylabel(forecast_columns[0])
    plt.legend()
    plt.grid(True)
    plt.xticks(ticks=range(0, len(forecast), 10))
    plt.xticks(rotation=45)

    buf = io.BytesIO()
    plt.savefig(buf, bbox_inches="tight")
    buf.seek(0)
    plt.close()
    image = Image.open(buf)

    return image


class TSAdResult(BaseTSResult):
    """A class representing the result of a time series anomaly detection task."""

    def _to_img(self) -> Image.Image:
        """apply"""
        anomaly = self["anomaly"]
        return {"res": visualize(anomaly)}

    def _to_csv(self) -> Any:
        """
        Converts the anomaly detection results to a CSV format.

        Returns:
            Any: The anomaly data formatted for CSV output, typically a DataFrame or similar structure.
        """
        return {"res": self["anomaly"]}
