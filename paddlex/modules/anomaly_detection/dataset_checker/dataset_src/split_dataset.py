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
import random
import shutil

from .....utils import logging
from .....utils.file_interface import custom_open


def split_dataset(root_dir, train_percent, val_percent):
    """split dataset"""
    assert train_percent > 0, ValueError(
        f"The train_percent({train_percent}) must greater than 0!"
    )
    assert val_percent > 0, ValueError(
        f"The val_percent({val_percent}) must greater than 0!"
    )
    if train_percent + val_percent != 100:
        raise ValueError(
            f"The sum of train_percent({train_percent})and val_percent({val_percent}) should be 100!"
        )

    img_dir = osp.join(root_dir, "images")
    assert osp.exists(img_dir), FileNotFoundError(
        f"The dir of images ({img_dir}) doesn't exist, please check!"
    )
    ann_dir = osp.join(root_dir, "annotations")
    assert osp.exists(ann_dir), FileNotFoundError(
        f"The dir of annotations ({ann_dir}) doesn't exist, please check!"
    )

    img_file_list = [osp.join("images", img_name) for img_name in os.listdir(img_dir)]
    img_num = len(img_file_list)
    ann_file_list = [
        osp.join("annotations", ann_name) for ann_name in os.listdir(ann_dir)
    ]
    ann_num = len(ann_file_list)
    assert img_num == ann_num, ValueError(
        "The number of images and annotations must be equal!"
    )

    split_tags = ["train", "val"]
    mapping_line_list = []
    for tag in split_tags:
        mapping_file = osp.join(root_dir, f"{tag}.txt")
        if not osp.exists(mapping_file):
            logging.info(f"The mapping file ({mapping_file}) doesn't exist, ignored.")
            continue
        with custom_open(mapping_file, "r") as fp:
            lines = filter(None, (line.strip() for line in fp.readlines()))
            mapping_line_list.extend(lines)

    sample_num = len(mapping_line_list)
    random.shuffle(mapping_line_list)
    split_percents = [train_percent, val_percent]
    start_idx = 0
    for tag, percent in zip(split_tags, split_percents):
        if tag == "test" and percent == 0:
            continue
        end_idx = start_idx + round(sample_num * percent / 100)
        end_idx = min(end_idx, sample_num)
        mapping_file = osp.join(root_dir, f"{tag}.txt")
        if os.path.exists(mapping_file):
            shutil.move(mapping_file, mapping_file + ".bak")
            logging.info(
                f"The original mapping file ({mapping_file}) "
                f"has been backed up to ({mapping_file}.bak)"
            )
        with custom_open(mapping_file, "w") as fp:
            fp.write("\n".join(mapping_line_list[start_idx:end_idx]))
        start_idx = end_idx
    return root_dir
