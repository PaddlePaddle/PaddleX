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

from PIL import Image, ImageOps

from .....utils.errors import CheckFailedError, DatasetFileNotFoundError
from .utils.visualizer import draw_label


def check_train(dataset_dir, output, sample_num=10):
    """check dataset"""
    dataset_dir = osp.abspath(dataset_dir)
    # Custom dataset
    if not osp.exists(dataset_dir) or not osp.isdir(dataset_dir):
        raise DatasetFileNotFoundError(file_path=dataset_dir)

    delim = " "
    valid_num_parts = 2

    label_map_dict = dict()
    sample_paths = []
    labels = []

    label_file = osp.join(dataset_dir, "label.txt")
    if not osp.exists(label_file):
        raise DatasetFileNotFoundError(
            file_path=label_file,
            solution=f"Ensure that `label.txt` exist in {dataset_dir}",
        )
    with open(label_file, "r", encoding="utf-8") as f:
        all_lines = f.readlines()
        random.seed(123)
        random.shuffle(all_lines)
        sample_cnts = len(all_lines)
        for line in all_lines:
            substr = line.strip("\n").split(delim)
            if len(substr) != valid_num_parts:
                raise CheckFailedError(
                    f"The number of delimiter-separated items in each row in {label_file} \
                            should be {valid_num_parts} (current delimiter is '{delim}')."
                )
            file_name = substr[0]
            label = substr[1]

            img_path = osp.join(dataset_dir, file_name)

            if not osp.exists(img_path):
                raise DatasetFileNotFoundError(file_path=img_path)

            vis_save_dir = osp.join(output, "demo_img")
            if not osp.exists(vis_save_dir):
                os.makedirs(vis_save_dir)

            try:
                label = int(label)
                label_map_dict[label] = str(label)
            except (ValueError, TypeError) as e:
                raise CheckFailedError(
                    f"Ensure that the second number in each line in {label_file} should be int."
                ) from e

            if len(sample_paths) < sample_num:
                img = Image.open(img_path)
                img = ImageOps.exif_transpose(img)
                vis_im = draw_label(img, label, label_map_dict)
                vis_path = osp.join(vis_save_dir, osp.basename(file_name))
                vis_im.save(vis_path)
                sample_path = osp.join(
                    "check_dataset", os.path.relpath(vis_path, output)
                )
                sample_paths.append(sample_path)
            labels.append(label)
    if min(labels) != 0:
        raise CheckFailedError(
            f"Ensure that the index starts from 0 in `{label_file}`."
        )
    num_classes = max(labels) + 1
    attrs = {}
    attrs["train_label_file"] = osp.relpath(label_file, output)
    attrs["train_num_classes"] = num_classes
    attrs["train_samples"] = sample_cnts
    attrs["train_sample_paths"] = sample_paths
    return attrs


def check_val(dataset_dir, output, sample_num=10):
    """check dataset"""
    dataset_dir = osp.abspath(dataset_dir)
    # Custom dataset
    if not osp.exists(dataset_dir) or not osp.isdir(dataset_dir):
        raise DatasetFileNotFoundError(file_path=dataset_dir)

    delim = " "
    valid_num_parts = 3

    labels = []
    sample_paths = []
    label_file = osp.join(dataset_dir, "pair_label.txt")
    if not osp.exists(label_file):
        raise DatasetFileNotFoundError(
            file_path=label_file,
            solution=f"Ensure that `label.txt` exist in {dataset_dir}",
        )
    with open(label_file, "r", encoding="utf-8") as f:
        all_lines = f.readlines()
        random.seed(123)
        random.shuffle(all_lines)
        sample_cnts = len(all_lines)
        for line in all_lines:
            substr = line.strip("\n").split(delim)
            if len(substr) != valid_num_parts:
                raise CheckFailedError(
                    f"The number of delimiter-separated items in each row in {label_file} \
                            should be {valid_num_parts} (current delimiter is '{delim}')."
                )
            left_file_name = substr[0]
            right_file_name = substr[1]
            label = substr[2]

            left_img_path = osp.join(dataset_dir, left_file_name)
            if not osp.exists(left_img_path):
                raise DatasetFileNotFoundError(file_path=left_img_path)

            right_img_path = osp.join(dataset_dir, right_file_name)
            if not osp.exists(right_img_path):
                raise DatasetFileNotFoundError(file_path=right_img_path)

            try:
                label = int(label)
                assert label in [0, 1], "Face eval dataset only support two classes"
            except (ValueError, TypeError) as e:
                raise CheckFailedError(
                    f"Ensure that the second number in each line in {label_file} should be int."
                ) from e

            vis_save_dir = osp.join(output, "demo_img")
            if not osp.exists(vis_save_dir):
                os.makedirs(vis_save_dir)

            if len(sample_paths) < sample_num:
                img = Image.open(left_img_path)
                img = ImageOps.exif_transpose(img)
                vis_path = osp.join(vis_save_dir, osp.basename(left_file_name))
                img.save(vis_path)
                sample_path = osp.join(
                    "check_dataset", os.path.relpath(vis_path, output)
                )
                sample_paths.append(sample_path)

            labels.append(label)
    num_classes = max(labels) + 1
    attrs = {}
    attrs["val_label_file"] = osp.relpath(label_file, output)
    attrs["val_num_classes"] = num_classes
    attrs["val_samples"] = sample_cnts
    attrs["val_sample_paths"] = sample_paths
    return attrs
