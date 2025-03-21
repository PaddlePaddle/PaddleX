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
from pathlib import Path

from ...base import BaseDatasetChecker
from ..model_list import MODELS
from .dataset_src import check_train, check_val


class FaceRecDatasetChecker(BaseDatasetChecker):
    """Dataset Checker for Image Classification Model"""

    entities = MODELS
    sample_num = 10

    def get_dataset_root(self, dataset_dir: str) -> str:
        """find the dataset root dir

        Args:
            dataset_dir (str): the directory that contain dataset.

        Returns:
            str: the root directory of dataset.
        """
        anno_dirs = list(Path(dataset_dir).glob("**/images"))
        assert len(anno_dirs) == 2
        dataset_dir = anno_dirs[0].parent.parent.as_posix()
        return dataset_dir

    def check_dataset(self, dataset_dir: str, sample_num: int = sample_num) -> dict:
        """check if the dataset meets the specifications and get dataset summary

        Args:
            dataset_dir (str): the root directory of dataset.
            sample_num (int): the number to be sampled.
        Returns:
            dict: dataset summary.
        """
        train_attr = check_train(os.path.join(dataset_dir, "train"), self.output)
        val_attr = check_val(os.path.join(dataset_dir, "val"), self.output)
        train_attr.update(val_attr)
        return train_attr

    def get_show_type(self) -> str:
        """get the show type of dataset

        Returns:
            str: show type
        """
        return "image"

    def get_dataset_type(self) -> str:
        """return the dataset type

        Returns:
            str: dataset type
        """
        return "ClsDataset"
