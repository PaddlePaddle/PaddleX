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
from collections import defaultdict

import numpy as np

from .....utils.deps import function_requires_deps, is_dep_available

if is_dep_available("matplotlib"):
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

from .....utils.file_interface import custom_open
from .....utils.fonts import PINGFANG_FONT_FILE_PATH


@function_requires_deps("matplotlib")
def deep_analyse(dataset_path, output):
    """class analysis for dataset"""
    tags = ["train", "val"]
    labels_cnt = defaultdict(str)
    label_path = os.path.join(dataset_path, "label.txt")
    with custom_open(label_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip().split()
        labels_cnt[line[0]] = " ".join(line[1:])
    for tag in tags:
        anno_path = os.path.join(dataset_path, f"{tag}.txt")
        classes_num = defaultdict(int)
        for i in range(len(labels_cnt)):
            classes_num[labels_cnt[str(i)]] = 0
        with custom_open(anno_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            classes_num[labels_cnt[line[1]]] += 1
        if tag == "train":
            cnts_train = [cat_ids for cat_name, cat_ids in classes_num.items()]
        elif tag == "val":
            cnts_val = [cat_ids for cat_name, cat_ids in classes_num.items()]

    classes = [cat_name for cat_name, cat_ids in classes_num.items()]
    sorted_id = sorted(
        range(len(cnts_train)), key=lambda k: cnts_train[k], reverse=True
    )
    cnts_train_sorted = [cnts_train[index] for index in sorted_id]
    cnts_val_sorted = [cnts_val[index] for index in sorted_id]
    classes_sorted = [classes[index] for index in sorted_id]
    x = np.arange(len(classes))
    width = 0.5

    # bar
    os_system = platform.system().lower()
    if os_system == "windows":
        plt.rcParams["font.sans-serif"] = "FangSong"
    else:
        font = font_manager.FontProperties(fname=PINGFANG_FONT_FILE_PATH, size=10)
    fig, ax = plt.subplots(figsize=(max(8, int(len(classes) / 5)), 5), dpi=300)
    ax.bar(x, cnts_train_sorted, width=0.5, label="train")
    ax.bar(x + width, cnts_val_sorted, width=0.5, label="val")
    plt.xticks(
        x + width / 2,
        classes_sorted,
        rotation=90,
        fontproperties=None if os_system == "windows" else font,
    )
    ax.set_xlabel(
        "类别名称", fontproperties=None if os_system == "windows" else font, fontsize=12
    )
    ax.set_ylabel(
        "视频数量", fontproperties=None if os_system == "windows" else font, fontsize=12
    )
    plt.legend(loc=1)
    fig.tight_layout()
    file_path = os.path.join(output, "histogram.png")
    fig.savefig(file_path, dpi=300)

    return {"histogram": os.path.join("check_dataset", "histogram.png")}
