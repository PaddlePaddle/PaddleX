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

import os
import platform

import numpy as np

from .....utils.deps import function_requires_deps, is_dep_available
from .....utils.file_interface import custom_open
from .....utils.fonts import PINGFANG_FONT_FILE_PATH

if is_dep_available("matplotlib"):
    import matplotlib.pyplot as plt
    from matplotlib import font_manager


@function_requires_deps("matplotlib")
def deep_analyse(dataset_path, output, dataset_type="ShiTuRec"):
    """class analysis for dataset"""
    tags = ["train", "gallery", "query"]
    tags_info = dict()
    for tag in tags:
        anno_path = os.path.join(dataset_path, f"{tag}.txt")
        with custom_open(anno_path, "r") as f:
            lines = f.readlines()
            lines = [line.strip("\n").split(" ") for line in lines]
            num_images = len(lines)
            num_labels = len(set([int(line[1]) for line in lines]))
        tags_info[tag] = {
            "num_images": num_images,
            "num_labels": num_labels,
        }

    categories = list(tags_info.keys())
    num_images = [tags_info[category]["num_images"] for category in categories]
    num_labels = [tags_info[category]["num_labels"] for category in categories]

    # bar
    os_system = platform.system().lower()
    if os_system == "windows":
        plt.rcParams["font.sans-serif"] = "FangSong"
    else:
        font = font_manager.FontProperties(fname=PINGFANG_FONT_FILE_PATH, size=10)

    x = np.arange(len(categories))  # 标签位置
    width = 0.35  # 每个条形的宽度

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, num_images, width, label="Num Images")
    rects2 = ax.bar(x + width / 2, num_labels, width, label="Num Classes")

    # 添加一些文本标签
    ax.set_xlabel("集合", fontproperties=None if os_system == "windows" else font)
    ax.set_ylabel("数量", fontproperties=None if os_system == "windows" else font)
    ax.set_title(
        "不同集合的图片和类别数量",
        fontproperties=None if os_system == "windows" else font,
    )
    ax.set_xticks(x, fontproperties=None if os_system == "windows" else font)
    ax.set_xticklabels(categories)
    ax.legend()

    # 在条形图上添加数值标签
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                "{}".format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    file_path = os.path.join(output, "histogram.png")
    fig.savefig(file_path, dpi=300)

    return {"histogram": os.path.join("check_dataset", "histogram.png")}
