# copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.
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

from ...base.dataset_checker import BaseDatasetChecker
from .dataset_src import check_dataset, convert_dataset, split_dataset, anaylse_dataset

from ..support_models import SUPPORT_MODELS


class SegDatasetChecker(BaseDatasetChecker):
    """ Dataset Checker for Semantic Segmentation Model """
    support_models = SUPPORT_MODELS
    sample_num = 10

    def convert_dataset(self, src_dataset_dir: str) -> str:
        """convert the dataset from other type to specified type

        Args:
            src_dataset_dir (str): the root directory of dataset.

        Returns:
            str: the root directory of converted dataset.
        """
        converted_dataset_path = osp.join(
            os.getenv("TEMP_DIR"), "converted_dataset")
        return convert_dataset(self.check_dataset_config.src_dataset_type,
                               dataset_dir, converted_dataset_path)

    def split_dataset(self, src_dataset_dir: str) -> str:
        """repartition the train and validation dataset

        Args:
            src_dataset_dir (str): the root directory of dataset.

        Returns:
            str: the root directory of splited dataset.
        """
        return split_dataset(dataset_dir,
                             self.check_dataset_config.split_train_percent,
                             self.check_dataset_config.split_val_percent)

    def check_dataset(self, dataset_dir: str,
                      sample_num: int=sample_num) -> dict:
        """check if the dataset meets the specifications and get dataset summary

        Args:
            dataset_dir (str): the root directory of dataset.
            sample_num (int): the number to be sampled.
        Returns:
            dict: dataset summary.
        """
        return check_dataset(dataset_dir, self.global_config.output, sample_num)

    def analyse(self, dataset_dir: str) -> dict:
        """deep analyse dataset

        Args:
            dataset_dir (str): the root directory of dataset.

        Returns:
            dict: the deep analysis results.
        """
        return anaylse_dataset(dataset_dir, self.global_config.output)

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
        return "SegDataset"
