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
from collections import defaultdict

from .....utils.file_interface import custom_open


def simple_analyse(dataset_path):
    """
    Analyse the dataset samples by return image path and label path

    Args:
        dataset_path (str): dataset path

    Returns:
        tuple: tuple of sample number, image path and label path for train, val and text subdataset.

    """
    tags = ["train", "val", "test"]
    sample_cnts = defaultdict(int)
    defaultdict(list)
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
            # img_paths[tag] = images_dict[tag]

    return f"训练数据样本数: {sample_cnts[tags[0]]}\t评估数据样本数: {sample_cnts[tags[1]]}"


def deep_analyse(dataset_path, output=None):
    """class analysis for dataset"""
    return simple_analyse(dataset_path)
