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
import pickle
import platform
from collections import defaultdict

import numpy as np

from paddlex.utils.deps import function_requires_deps, is_dep_available
from paddlex.utils.fonts import PINGFANG_FONT_FILE_PATH

if is_dep_available("matplotlib"):
    import matplotlib.pyplot as plt
    from matplotlib import font_manager


@function_requires_deps("matplotlib")
def deep_analyse(dataset_dir, output):
    """class analysis for dataset"""
    tags = ["train", "val"]
    class_name_train = defaultdict(int)
    class_name_val = defaultdict(int)
    for tag in tags:
        anno_file = os.path.join(dataset_dir, f"nuscenes_infos_{tag}.pkl")
        with open(anno_file, "rb") as f:
            datas = pickle.load(f)
        data_infos = datas["infos"]
        for item in data_infos:
            gts = item["gt_names"]
            for gt_name in gts:
                if tag == "train":
                    class_name_train[gt_name] = (
                        0
                        if gt_name not in class_name_train
                        else class_name_train[gt_name] + 1
                    )
                elif tag == "val":
                    class_name_val[gt_name] = (
                        0
                        if gt_name not in class_name_val
                        else class_name_val[gt_name] + 1
                    )

    classes = set()
    for key in class_name_train:
        classes.add(key)
    for key in class_name_val:
        classes.add(key)

    # set cnt to 0 if class not in cnt dict
    for key in classes:
        if key not in class_name_train:
            class_name_train[key] = 0
        if key not in class_name_val:
            class_name_val[key] = 0

    cnts_train = [cat_ids for cat_name, cat_ids in class_name_train.items()]
    cnts_val = [cat_ids for cat_name, cat_ids in class_name_val.items()]

    # sort class name
    classes = [cat_name for cat_name, cat_ids in class_name_train.items()]
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
        font = font_manager.FontProperties(fname=PINGFANG_FONT_FILE_PATH)
    fig, ax = plt.subplots(figsize=(max(8, int(len(classes) / 5)), 5), dpi=120)
    ax.bar(x, cnts_train_sorted, width=0.5, label="train")
    ax.bar(x + width, cnts_val_sorted, width=0.5, label="val")
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
