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


@function_requires_deps("matplotlib")
def visualize(predicted_label, input_ts, target_cols):
    """
    Visualize time series data and its prediction results.

    Parameters:
    - input_ts: A DataFrame containing the input_ts.
    - predicted_label: A list of predicted class labels.

    Returns:
    - image: An image object containing the visualization result.
    """
    # 设置图形大小
    plt.figure(figsize=(12, 6))
    input_ts.columns
    input_ts.index = input_ts.index.astype(str)
    length = len(input_ts)
    value = predicted_label.loc[0, "classid"]
    plt.plot(
        input_ts.index,
        input_ts[target_cols[0]],
        label=f"Predicted classid: {value}",
        color="blue",
    )

    # 设置图形标题和标签
    plt.title("Time Series input_ts with Predicted Labels")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.xticks(ticks=range(0, length, 10))
    plt.xticks(rotation=45)

    # 保存图像到内存
    buf = io.BytesIO()
    plt.savefig(buf, bbox_inches="tight")
    buf.seek(0)
    plt.close()
    image = Image.open(buf)

    return image


class TSClsResult(BaseTSResult):
    """A class representing the result of a time series classification task."""

    def _to_img(self) -> Image.Image:
        """apply"""
        classification = self["classification"]
        ts_input = self["input_ts_data"]
        return {"res": visualize(classification, ts_input, self["target_cols"])}

    def _to_csv(self) -> Any:
        """
        Converts the classification results to a CSV format.

        Returns:
            Any: The classification data formatted for CSV output, typically a DataFrame or similar structure.
        """
        return {"res": self["classification"]}
