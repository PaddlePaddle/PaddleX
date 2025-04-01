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
import pandas as pd

from .....utils.deps import function_requires_deps, is_dep_available
from .....utils.fonts import PINGFANG_FONT_FILE_PATH

if is_dep_available("matplotlib"):
    import matplotlib.pyplot as plt
    from matplotlib import font_manager


@function_requires_deps("matplotlib")
def deep_analyse(dataset_dir, output, label_col="label"):
    """class analysis for dataset"""
    tags = ["train", "val"]
    label_unique = None
    for tag in tags:
        csv_path = os.path.abspath(os.path.join(dataset_dir, tag + ".csv"))
        df = pd.read_csv(csv_path)
        if label_col not in df.columns:
            raise ValueError(f"default label_col: {label_col} not in {tag} dataset")
        if label_unique is None:
            label_unique = df[label_col].unique()
        cls_dict = {}
        for label in label_unique:
            vis_df = df[df[label_col].isin([label])]
            cls_dict[label] = len(vis_df)
        if tag == "train":
            cls_train = [label_num for label_col, label_num in cls_dict.items()]
        elif tag == "val":
            cls_val = [label_num for label_col, label_num in cls_dict.items()]
    sorted_id = sorted(range(len(cls_train)), key=lambda k: cls_train[k], reverse=True)
    cls_train_sorted = sorted(cls_train, reverse=True)
    cls_val_sorted = [cls_val[index] for index in sorted_id]
    classes_sorted = [label_unique[index] for index in sorted_id]
    x = np.arange(len(label_unique))
    width = 0.5

    # bar
    os_system = platform.system().lower()
    if os_system == "windows":
        plt.rcParams["font.sans-serif"] = "FangSong"
    else:
        font = font_manager.FontProperties(fname=PINGFANG_FONT_FILE_PATH)
    fig, ax = plt.subplots(figsize=(max(8, int(len(label_unique) / 5)), 5), dpi=120)
    ax.bar(x, cls_train_sorted, width=0.5, label="train")
    ax.bar(x + width, cls_val_sorted, width=0.5, label="val")
    plt.xticks(
        x + width / 2,
        classes_sorted,
        rotation=90,
        fontproperties=None if os_system == "windows" else font,
    )
    ax.set_ylabel("Counts")
    plt.legend()
    fig.tight_layout()
    fig_path = os.path.join(output, "histogram.png")
    fig.savefig(fig_path)
    return {"histogram": os.path.join("check_dataset", "histogram.png")}
