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
import os.path as osp

import numpy as np
from PIL import Image, ImageOps

from .....utils.deps import function_requires_deps, is_dep_available
from .....utils.file_interface import custom_open

if is_dep_available("matplotlib"):
    import matplotlib.pyplot as plt


@function_requires_deps("matplotlib")
def anaylse_dataset(dataset_dir, output):
    """class analysis for dataset"""

    split_tags = ["train", "val"]
    label2count = {tag: dict() for tag in split_tags}
    for tag in split_tags:
        mapping_file = osp.join(dataset_dir, f"{tag}.txt")
        with custom_open(mapping_file, "r") as fp:
            lines = filter(None, (line.strip() for line in fp.readlines()))
            for i, line in enumerate(lines):
                _, ann_file = line.split(" ")
                ann_file = osp.join(dataset_dir, ann_file)
                ann = np.array(ImageOps.exif_transpose(Image.open(ann_file)), "uint8")

                for idx in set(ann.reshape([-1]).tolist()):
                    if idx == 255:
                        continue
                    if idx not in label2count[tag]:
                        label2count[tag][idx] = 1
                    else:
                        label2count[tag][idx] += 1
            if label2count[tag].get(0, None) is None:
                label2count[tag][0] = 0

    train_label_idx = np.array(list(label2count["train"].keys()))
    val_label_idx = np.array(list(label2count["val"].keys()))
    label_idx = np.array(list(set(train_label_idx) | set(val_label_idx)))
    x = np.arange(len(label_idx))
    train_list = []
    val_list = []
    for i in range(len(label_idx)):
        train_list.append(label2count["train"].get(i, 0))
        val_list.append(label2count["val"].get(i, 0))
    fig, ax = plt.subplots(figsize=(max(8, int(len(label_idx) / 5)), 5), dpi=120)

    width = (0.5,)
    ax.bar(x, train_list, width=width, label="train")
    ax.bar(x + width, val_list, width=width, label="val")

    plt.xticks(x + 0.25, label_idx)
    ax.set_xlabel("Label Index")
    ax.set_ylabel("Sample Counts")
    plt.legend()
    fig.tight_layout()
    fig_path = os.path.join(output, "histogram.png")
    fig.savefig(fig_path)
    return {"histogram": os.path.join("check_dataset", "histogram.png")}
