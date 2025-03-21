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
from collections import defaultdict

from .....utils.errors import CheckFailedError, DatasetFileNotFoundError
from .....utils.file_interface import custom_open


def check(dataset_dir, output, sample_num=10):
    """check dataset"""
    dataset_dir = osp.abspath(dataset_dir)
    # Custom dataset
    if not osp.exists(dataset_dir) or not osp.isdir(dataset_dir):
        raise DatasetFileNotFoundError(file_path=dataset_dir)

    tags = ["train", "val"]
    valid_num_parts = 5

    sample_cnts = dict()
    label_map_dict = dict()
    sample_paths = defaultdict(list)
    labels = []
    image_dir = osp.join(dataset_dir, "rgb-images")
    label_dir = osp.join(dataset_dir, "labels")
    if not osp.exists(image_dir):
        raise DatasetFileNotFoundError(file_path=image_dir)
    if not osp.exists(label_dir):
        raise DatasetFileNotFoundError(file_path=label_dir)

    label_map_file = osp.join(dataset_dir, "label_map.txt")
    if not osp.exists(label_map_file):
        raise DatasetFileNotFoundError(
            file_path=label_map_file,
            solution=f"Ensure that `label_map.txt` exist in {dataset_dir}",
        )
    with open(label_map_file, "r", encoding="utf-8") as f:
        all_lines = f.readlines()
        for line in all_lines:
            substr = line.strip("\n").split(" ", 1)
            try:
                label_idx = int(substr[1])
                labels.append(label_idx)
                label_map_dict[label_idx] = str(substr[0])
            except:
                raise CheckFailedError(
                    f"Ensure that the second number in each line in {label_map_file} should be int."
                )
    if min(labels) != 1:
        raise CheckFailedError(
            f"Ensure that the index starts from 1 in `{label_map_file}`."
        )

    for tag in tags:
        file_list = osp.join(dataset_dir, f"{tag}.txt")
        if not osp.exists(file_list):
            if tag in ("train", "val"):
                # train and val file lists must exist
                raise DatasetFileNotFoundError(
                    file_path=file_list,
                    solution=f"Ensure that both `train.txt` and `val.txt` exist in {dataset_dir}",
                )
            else:
                # tag == 'test'
                continue
        else:
            with open(file_list, "r", encoding="utf-8") as f:
                all_lines = f.readlines()
                random.seed(123)
                random.shuffle(all_lines)
                sample_cnts[tag] = len(all_lines)

                for line in all_lines:
                    substr = line.strip("\n")
                    label_path = osp.join(dataset_dir, substr)
                    img_path = (
                        osp.join(dataset_dir, substr)
                        .replace("labels", "rgb-images")
                        .replace("txt", "jpg")
                    )

                    if not osp.exists(img_path):
                        raise DatasetFileNotFoundError(file_path=img_path)
                    if not osp.exists(label_path):
                        raise DatasetFileNotFoundError(file_path=label_path)
                    with custom_open(label_path, "r") as f:
                        label_lines = f.readlines()
                        for label_line in label_lines:
                            label_info = label_line.strip().split(" ")
                            try:
                                int(label_info[0])
                            except (ValueError, TypeError) as e:
                                raise CheckFailedError(
                                    f"Ensure that the first number in each line in {label_info} should be int."
                                ) from e
                                if len(label_info) != valid_num_parts:
                                    raise CheckFailedError(
                                        f"Ensure that each line in {label_line} has exactly two numbers."
                                    )

                    if len(sample_paths[tag]) < sample_num:
                        sample_path = osp.join(
                            "check_dataset", os.path.relpath(img_path, output)
                        )
                        sample_paths[tag].append(sample_path)

    num_classes = max(labels)

    attrs = {}
    attrs["label_file"] = osp.relpath(label_map_file, output)
    attrs["num_classes"] = num_classes
    attrs["train_samples"] = sample_cnts["train"]
    attrs["train_sample_paths"] = sample_paths["train"]

    attrs["val_samples"] = sample_cnts["val"]
    attrs["val_sample_paths"] = sample_paths["val"]

    return attrs
