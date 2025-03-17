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

from pathlib import Path

from ...base import BaseDatasetChecker
from ....utils.errors import UnsupportedAPIError
from .dataset_src import check_train, check_val, deep_analyse

from ..model_list import MODELS


class JDEDatasetChecker(BaseDatasetChecker):
    """Dataset Checker for Object Detection Model"""

    entities = MODELS
    sample_num = 10
    dataset_attrs = {}

    def get_dataset_root(self, dataset_dir: str) -> str:
        """find the dataset root dir

        Args:
            dataset_dir (str): the directory that contain dataset.

        Returns:
            str: the root directory of dataset.
        """
        dataset_dir = Path(dataset_dir).as_posix()
        return dataset_dir

    def convert_dataset(self, src_dataset_dir: str) -> str:
        """convert the dataset from other type to specified type

        Args:
            src_dataset_dir (str): the root directory of dataset.

        Returns:
            str: the root directory of converted dataset.
        """
        raise UnsupportedAPIError("MOTDataset does not support convert operation")

    def split_dataset(self, src_dataset_dir: str) -> str:
        """repartition the train and validation dataset

        Args:
            src_dataset_dir (str): the root directory of dataset.

        Returns:
            str: the root directory of splited dataset.
        """
        raise UnsupportedAPIError("MOTDataset does not support split operation")

    def check_dataset(self, dataset_dir: str, sample_num: int = sample_num) -> dict:
        """check if the dataset meets the specifications and get dataset summary

        Args:
            dataset_dir (str): the root directory of dataset.
            sample_num (int): the number to be sampled.
        Returns:
            dict: dataset summary.
        """
        if self.dataset_attrs:
            return self.dataset_attrs
        train_attr = check_train(dataset_dir, self.output, sample_num)
        val_attr = check_val(dataset_dir, self.output, sample_num)
        train_attr.update(val_attr)
        self.dataset_attrs = train_attr
        return train_attr

    def analyse(self, dataset_dir: str) -> dict:
        """deep analyse dataset

        Args:
            dataset_dir (str): the root directory of dataset.

        Returns:
            dict: the deep analysis results.
        """
        if not self.dataset_attrs:
            self.check_dataset(dataset_dir)
        return deep_analyse(self.dataset_attrs, self.output)

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
        return "MOTDataset"
