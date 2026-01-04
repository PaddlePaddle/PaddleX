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


def check(dataset_dir, output, sample_num=10):
    """check dataset"""
    dataset_dir = osp.abspath(dataset_dir)
    # Custom dataset
    if not osp.exists(dataset_dir) or not osp.isdir(dataset_dir):
        raise DatasetFileNotFoundError(file_path=dataset_dir)

    tags = ["train", "val"]
    delim = " "
    valid_num_parts = 2

    sample_cnts = dict()
    label_map_dict = dict()
    sample_paths = defaultdict(list)
    labels = []

    label_file = osp.join(dataset_dir, "label.txt")
    if not osp.exists(label_file):
        raise DatasetFileNotFoundError(
            file_path=label_file,
            solution=f"Ensure that `label.txt` exist in {dataset_dir}",
        )

    with open(label_file, "r", encoding="utf-8") as f:
        all_lines = f.readlines()
        for line in all_lines:
            substr = line.strip("\n").split(" ", 1)
            try:
                label_idx = int(substr[0])
                labels.append(label_idx)
                label_map_dict[label_idx] = str(substr[1])
            except:
                raise CheckFailedError(
                    f"Ensure that the first number in each line in {label_file} should be int."
                )
    if min(labels) != 0:
        raise CheckFailedError(
            f"Ensure that the index starts from 0 in `{label_file}`."
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
                    substr = line.strip("\n").split(delim)
                    if len(substr) != valid_num_parts:
                        raise CheckFailedError(
                            f"The number of delimiter-separated items in each row in {file_list} \
                                    should be {valid_num_parts} (current delimiter is '{delim}')."
                        )
                    file_name = substr[0]
                    label = substr[1]

                    video_path = osp.join(dataset_dir, file_name)

                    if not osp.exists(video_path):
                        raise DatasetFileNotFoundError(file_path=video_path)

                    if len(sample_paths[tag]) < sample_num:
                        sample_path = osp.join(
                            "check_dataset", os.path.relpath(video_path, output)
                        )
                        sample_paths[tag].append(sample_path)

                    try:
                        label = int(label)
                    except (ValueError, TypeError) as e:
                        raise CheckFailedError(
                            f"Ensure that the second number in each line in {label_file} should be int."
                        ) from e

    num_classes = max(labels) + 1

    attrs = {}
    attrs["label_file"] = osp.relpath(label_file, output)
    attrs["num_classes"] = num_classes
    attrs["train_samples"] = sample_cnts["train"]
    attrs["train_sample_paths"] = sample_paths["train"]

    attrs["val_samples"] = sample_cnts["val"]
    attrs["val_sample_paths"] = sample_paths["val"]

    return attrs
