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
from collections import defaultdict

from .....utils.errors import CheckFailedError, DatasetFileNotFoundError


def check(
    dataset_dir, output, dataset_type=" FormulaRecDataset", mode="fast", sample_num=10
):
    """check dataset"""
    if dataset_type == "FormulaRecDataset":
        # Custom dataset
        if not osp.exists(dataset_dir) or not osp.isdir(dataset_dir):
            raise DatasetFileNotFoundError(file_path=dataset_dir)
        tags = ["train", "val"]
        delim = "\t"
        valid_num_parts = 2
        max_recorded_sample_cnts = 50
        sample_cnts = dict()
        sample_paths = defaultdict(list)
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
                    continue
            else:
                with open(file_list, "r", encoding="utf-8") as f:
                    all_lines = f.readlines()
                    sample_cnts[tag] = len(all_lines)
                    for line in all_lines:
                        substr = line.strip("\n").split(delim)
                        if len(line.strip("\n")) < 1:
                            continue
                        if len(substr) != valid_num_parts and len(line.strip("\n")) > 1:
                            raise CheckFailedError(
                                f"Error in {line}, The number of delimiter-separated items in each row "
                                "in {file_list} should be {valid_num_parts} (current delimiter is '{delim}')."
                            )
                        file_name = substr[0]
                        img_path = osp.join(dataset_dir, file_name)
                        if len(sample_paths[tag]) < max_recorded_sample_cnts:
                            sample_paths[tag].append(os.path.relpath(img_path, output))

                        if not os.path.exists(img_path):
                            raise DatasetFileNotFoundError(file_path=img_path)

        meta = {}
        meta["train_samples"] = sample_cnts["train"]
        meta["train_sample_paths"] = sample_paths["train"][:sample_num]

        meta["val_samples"] = sample_cnts["val"]
        meta["val_sample_paths"] = sample_paths["val"][:sample_num]

        return meta
