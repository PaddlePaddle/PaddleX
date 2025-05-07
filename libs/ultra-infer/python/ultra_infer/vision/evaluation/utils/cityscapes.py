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
import glob
from . import fd_logging as logging

# import fd_logging as logging


class Cityscapes(object):
    """
    Cityscapes dataset `https://www.cityscapes-dataset.com/`.
    The folder structure is as follow:

        cityscapes
        |
        |--leftImg8bit
        |  |--train
        |  |--val
        |  |--test
        |
        |--gtFine
        |  |--train
        |  |--val
        |  |--test

    Args:
        dataset_root (str): Cityscapes dataset directory.
    """

    NUM_CLASSES = 19

    def __init__(self, dataset_root, mode):
        self.dataset_root = dataset_root
        self.file_list = list()
        mode = mode.lower()
        self.mode = mode
        self.num_classes = self.NUM_CLASSES
        self.ignore_index = 255

        img_dir = os.path.join(self.dataset_root, "leftImg8bit")
        label_dir = os.path.join(self.dataset_root, "gtFine")
        if (
            self.dataset_root is None
            or not os.path.isdir(self.dataset_root)
            or not os.path.isdir(img_dir)
            or not os.path.isdir(label_dir)
        ):
            raise ValueError(
                "The dataset is not Found or the folder structure is nonconfoumance."
            )

        label_files = sorted(
            glob.glob(os.path.join(label_dir, mode, "*", "*_gtFine_labelTrainIds.png"))
        )
        img_files = sorted(
            glob.glob(os.path.join(img_dir, mode, "*", "*_leftImg8bit.png"))
        )

        self.file_list = [
            [img_path, label_path]
            for img_path, label_path in zip(img_files, label_files)
        ]

        self.num_samples = len(self.file_list)
        logging.info("{} samples in file {}".format(self.num_samples, img_dir))
