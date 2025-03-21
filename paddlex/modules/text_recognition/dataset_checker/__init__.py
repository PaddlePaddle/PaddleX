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


import json
import os
import os.path as osp
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from ...base import BaseDatasetChecker
from ..model_list import MODELS
from .dataset_src import check, convert, deep_analyse, split_dataset


class TextRecDatasetChecker(BaseDatasetChecker):
    """Dataset Checker for Text Recognition Model"""

    entities = MODELS
    sample_num = 10

    def get_dataset_root(self, dataset_dir: str) -> str:
        """find the dataset root dir

        Args:
            dataset_dir (str): the directory that contain dataset.

        Returns:
            str: the root directory of dataset.
        """
        anno_dirs = list(Path(dataset_dir).glob("**/train.txt"))
        assert len(anno_dirs) == 1
        dataset_dir = anno_dirs[0].parent.as_posix()
        return dataset_dir

    def convert_dataset(self, src_dataset_dir: str) -> str:
        """convert the dataset from other type to specified type

        Args:
            src_dataset_dir (str): the root directory of dataset.

        Returns:
            str: the root directory of converted dataset.
        """
        return convert(
            self.check_dataset_config.convert.src_dataset_type, src_dataset_dir
        )

    def split_dataset(self, src_dataset_dir: str) -> str:
        """repartition the train and validation dataset

        Args:
            src_dataset_dir (str): the root directory of dataset.

        Returns:
            str: the root directory of splited dataset.
        """
        return split_dataset(
            src_dataset_dir,
            self.check_dataset_config.split.train_percent,
            self.check_dataset_config.split.val_percent,
        )

    def check_dataset(self, dataset_dir: str, sample_num: int = sample_num) -> dict:
        """check if the dataset meets the specifications and get dataset summary

        Args:
            dataset_dir (str): the root directory of dataset.
            sample_num (int): the number to be sampled.
        Returns:
            dict: dataset summary.
        """
        return check(
            dataset_dir,
            self.output,
            sample_num=10,
            dataset_type=self.get_dataset_type(),
        )

    def analyse(self, dataset_dir: str) -> dict:
        """deep analyse dataset

        Args:
            dataset_dir (str): the root directory of dataset.

        Returns:
            dict: the deep analysis results.
        """
        if self.global_config["model"] in ["LaTeX_OCR_rec"]:
            datatype = "LaTeXOCRDataset"
        else:
            datatype = "MSTextRecDataset"
        return deep_analyse(dataset_dir, self.output, datatype=datatype)

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
        if self.global_config["model"] in ["LaTeX_OCR_rec"]:
            return "LaTeXOCRDataset"
        else:
            return "MSTextRecDataset"
