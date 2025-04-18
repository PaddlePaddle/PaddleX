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
from .....utils.fonts import PINGFANG_FONT_FILE_PATH

if is_dep_available("matplotlib"):
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
if is_dep_available("pycocotools"):
    from pycocotools.coco import COCO


@function_requires_deps("pycocotools", "matplotlib")
def deep_analyse(dataset_dir, output):
    """class analysis for dataset"""
    tags = ["train", "val"]
    all_instances = 0
    for tag in tags:
        annotations_path = os.path.abspath(
            os.path.join(dataset_dir, f"annotations/instance_{tag}.json")
        )
        labels_cnt = defaultdict(list)
        coco = COCO(annotations_path)
        cat_ids = coco.getCatIds()
        for cat_id in cat_ids:
            cat_name = coco.loadCats(ids=cat_id)[0]["name"]
            labels_cnt[cat_name] = labels_cnt[cat_name] + coco.getAnnIds(catIds=cat_id)
            all_instances += len(labels_cnt[cat_name])
        if tag == "train":
            cnts_train = [len(cat_ids) for cat_name, cat_ids in labels_cnt.items()]
        elif tag == "val":
            cnts_val = [len(cat_ids) for cat_name, cat_ids in labels_cnt.items()]
    classes = [cat_name for cat_name, cat_ids in labels_cnt.items()]
    sorted_id = sorted(
        range(len(cnts_train)), key=lambda k: cnts_train[k], reverse=True
    )
    cnts_train_sorted = sorted(cnts_train, reverse=True)
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
