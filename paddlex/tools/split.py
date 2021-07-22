# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from .dataset_split import *
from paddlex.utils import logging


def dataset_split(dataset_dir, dataset_format, val_value, test_value,
                  save_dir):
    logging.info("Dataset split starts...")
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
    else:
        raise Exception("Dataset format {} is not supported.".format(
            dataset_format))
    logging.info("Dataset split done.")
    logging.info("Train samples: {}".format(train_num))
    logging.info("Eval samples: {}".format(val_num))
    logging.info("Test samples: {}".format(test_num))
    logging.info("Split files saved in {}".format(save_dir))
