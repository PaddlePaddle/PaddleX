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


import os
import platform
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

from .....utils.fonts import PINGFANG_FONT_FILE_PATH


def deep_analyse(attrs, output):
    """Generate comparison histogram for dataset statistics"""
    # Extract values from attributes dictionary
    train_values = [attrs["train_num_identities"], attrs["train_samples"]]
    val_values = [attrs["val_num_identities"], attrs["val_samples"]]

    # Plot configuration
    os_system = platform.system().lower()
    if os_system == "windows":
        plt.rcParams["font.sans-serif"] = "FangSong"
    else:
        font = font_manager.FontProperties(fname=PINGFANG_FONT_FILE_PATH)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    x = np.arange(len(train_values))  # x-axis positions for the groups
    width = 0.5  # width of the bars

    # Plot bars
    ax.bar(x, train_values, width, label="train")
    ax.bar(x + width, val_values, width, label="val")

    # Configure axis and labels
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["num_identities", "num_samples"],
        fontproperties=None if os_system == "windows" else font,
    )
    ax.set_ylabel("Counts")

    # Add legend and adjust layout
    plt.legend()
    fig.tight_layout()

    # Save and return path
    fig_path = os.path.join(output, "histogram.png")
    fig.savefig(fig_path)
    plt.close(fig)
    return {"histogram": os.path.join("check_dataset", "histogram.png")}
