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


import math
import os
import platform
from collections import defaultdict

import numpy as np

from .....utils.deps import function_requires_deps, is_dep_available
from .....utils.file_interface import custom_open
from .....utils.fonts import PINGFANG_FONT_FILE_PATH
from .....utils.logging import warning

if is_dep_available("opencv-contrib-python"):
    import cv2
if is_dep_available("matplotlib"):
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.backends.backend_agg import FigureCanvasAgg


def simple_analyse(dataset_path, images_dict):
    """
    Analyse the dataset samples by return image path and label path

    Args:
        dataset_path (str): dataset path
        ds_meta (dict): dataset meta
        images_dict (dict): train, val and test image path

    Returns:
        tuple: tuple of sample number, image path and label path for train, val and text subdataset.

    """
    tags = ["train", "val", "test"]
    sample_cnts = defaultdict(int)
    img_paths = defaultdict(list)
    res = [None] * 6

    for tag in tags:
        file_list = os.path.join(dataset_path, f"{tag}.txt")
        if not os.path.exists(file_list):
            if tag in ("train", "val"):
                res.insert(0, "数据集不符合规范，请先通过数据校准")
                return res
            else:
                continue
        else:
            with custom_open(file_list, "r") as f:
                all_lines = f.readlines()

            # Each line corresponds to a sample
            sample_cnts[tag] = len(all_lines)
            img_paths[tag] = images_dict[tag]

    return (
        "完成数据分析",
        sample_cnts[tags[0]],
        sample_cnts[tags[1]],
        sample_cnts[tags[2]],
        img_paths[tags[0]],
        img_paths[tags[1]],
        img_paths[tags[2]],
    )


@function_requires_deps("matplotlib", "opencv-contrib-python")
def deep_analyse(dataset_path, output, datatype="MSTextRecDataset"):
    """class analysis for dataset"""
    tags = ["train", "val"]
    labels_cnt = {}
    x_max = []
    classes_max = []
    for tag in tags:
        image_path = os.path.join(dataset_path, f"{tag}.txt")
        str_nums = []
        with custom_open(image_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip().split("\t")
            if len(line) != 2:
                warning(f"Error in {line}.")
                continue
            str_nums.append(len(line[1]))
        if datatype == "LaTeXOCRDataset":
            max_length = min(768, max(str_nums))
            interval = 20
        else:
            max_length = min(100, max(str_nums))
            interval = 5
        start = 0
        for i in range(1, math.ceil((max_length / interval))):
            stop = i * interval
            num_str = sum(start < i <= stop for i in str_nums)
            labels_cnt[f"{start}-{stop}"] = num_str
            start = stop
        if sum(max_length < i for i in str_nums) != 0:
            labels_cnt[f"> {max_length}"] = sum(max_length < i for i in str_nums)
        if tag == "train":
            cnts_train = [cat_ids for cat_name, cat_ids in labels_cnt.items()]
            x_train = np.arange(len(cnts_train))
            if len(x_train) > len(x_max):
                x_max = x_train
                classes_max = [cat_name for cat_name, cat_ids in labels_cnt.items()]
        elif tag == "val":
            cnts_val = [cat_ids for cat_name, cat_ids in labels_cnt.items()]
            x_val = np.arange(len(cnts_val))
            if len(x_val) > len(x_max):
                x_max = x_val
                classes_max = [cat_name for cat_name, cat_ids in labels_cnt.items()]

    width = 0.3

    # bar
    os_system = platform.system().lower()
    if os_system == "windows":
        plt.rcParams["font.sans-serif"] = "FangSong"
    else:
        font = font_manager.FontProperties(fname=PINGFANG_FONT_FILE_PATH, size=15)
    if datatype == "LaTeXOCRDataset":
        fig, ax = plt.subplots(figsize=(15, 9), dpi=120)
        xlabel_name = "公式长度区间"
    else:
        fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
        xlabel_name = "文本字长度区间"
    ax.bar(x_train, cnts_train, width=0.3, label="train")
    ax.bar(x_val + width, cnts_val, width=0.3, label="val")
    plt.xticks(x_max + width / 2, classes_max, rotation=90)
    plt.legend(prop={"size": 18})
    ax.set_xlabel(
        xlabel_name,
        fontproperties=None if os_system == "windows" else font,
        fontsize=12,
    )
    ax.set_ylabel(
        "图片数量", fontproperties=None if os_system == "windows" else font, fontsize=12
    )

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    pie_array = np.frombuffer(canvas.tostring_rgb(), dtype="uint8").reshape(
        int(height), int(width), 3
    )
    fig1_path = os.path.join(output, "histogram.png")
    cv2.imwrite(fig1_path, pie_array)

    return {"histogram": os.path.join("check_dataset", "histogram.png")}
