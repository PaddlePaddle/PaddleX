#!/usr/bin/env python
# coding: utf-8
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .dataset_split.coco_split import split_coco_dataset
from .dataset_split.voc_split import split_voc_dataset
from .dataset_split.imagenet_split import split_imagenet_dataset
from .dataset_split.seg_split import split_seg_dataset


def dataset_split(dataset_dir, dataset_format, val_value, test_value,
                  save_dir):
    if dataset_format == "coco":
        train_num, val_num, test_num = split_coco_dataset(
            dataset_dir, val_value, test_value, save_dir)
    elif dataset_format == "voc":
        train_num, val_num, test_num = split_voc_dataset(
            dataset_dir, val_value, test_value, save_dir)
    elif dataset_format == "seg":
        train_num, val_num, test_num = split_seg_dataset(
            dataset_dir, val_value, test_value, save_dir)
    elif dataset_format == "imagenet":
        train_num, val_num, test_num = split_imagenet_dataset(
            dataset_dir, val_value, test_value, save_dir)
    print("Dataset Split Done.")
    print("Train samples: {}".format(train_num))
    print("Eval samples: {}".format(val_num))
    print("Test samples: {}".format(test_num))
    print("Split files saved in {}".format(save_dir))
