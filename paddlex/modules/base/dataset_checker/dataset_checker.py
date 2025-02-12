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
from abc import ABC, abstractmethod

from .utils import build_res_dict
from ....utils.misc import AutoRegisterABCMetaClass
from ....utils.config import AttrDict
from ....utils.logging import info


def build_dataset_checker(config: AttrDict) -> "BaseDatasetChecker":
    """build dataset checker

    Args:
        config (AttrDict): PaddleX pipeline config, which is loaded from pipeline yaml file.

    Returns:
        BaseDatasetChecker: the dataset checker, which is subclass of BaseDatasetChecker.
    """
    model_name = config.Global.model
    try:
        import feature_line_modules
    except ModuleNotFoundError:
        pass

    return BaseDatasetChecker.get(model_name)(config)


class BaseDatasetChecker(ABC, metaclass=AutoRegisterABCMetaClass):
    """Base Dataset Checker"""

    __is_base = True

    def __init__(self, config):
        """Initialize the instance.

        Args:
            config (AttrDict): PaddleX pipeline config, which is loaded from pipeline yaml file.
        """
        super().__init__()
        self.global_config = config.Global
        self.check_dataset_config = config.CheckDataset
        self.output = os.path.join(self.global_config.output, "check_dataset")

    def check(self) -> dict:
        """execute dataset checking

        Returns:
            dict: the dataset checking result.
        """
        dataset_dir = self.get_dataset_root(self.global_config.dataset_dir)

        if not os.path.exists(self.output):
            os.makedirs(self.output)

        if self.check_dataset_config.get("convert", None):
            if self.check_dataset_config.convert.get("enable", False):
                self.convert_dataset(dataset_dir)
                info("Convert dataset successfully !")

        if self.check_dataset_config.get("split", None):
            if self.check_dataset_config.split.get("enable", False):
                self.split_dataset(dataset_dir)
                info("Split dataset successfully !")

        attrs = self.check_dataset(dataset_dir)
        analysis = self.analyse(dataset_dir)
        check_result = build_res_dict(True)
        check_result["attributes"] = attrs
        check_result["analysis"] = analysis
        check_result["dataset_path"] = os.path.basename(dataset_dir)
        check_result["show_type"] = self.get_show_type()
        check_result["dataset_type"] = self.get_dataset_type()
        info("Check dataset passed !")
        return check_result

    def get_dataset_root(self, dataset_dir: str) -> str:
        """find the dataset root dir

        Args:
            dataset_dir (str): the directory that contain dataset.

        Returns:
            str: the root directory of dataset.
        """
        # XXX: forward compatible
        # dataset_dir = [d for d in Path(dataset_dir).iterdir() if d.is_dir()]
        # assert len(dataset_dir) == 1
        # return dataset_dir[0].as_posix()
        return dataset_dir

    @abstractmethod
    def check_dataset(self, dataset_dir: str):
        """check if the dataset meets the specifications and get dataset summary

        Args:
            dataset_dir (str): the root directory of dataset.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError

    def convert_dataset(self, src_dataset_dir: str) -> str:
        """convert the dataset from other type to specified type

        Args:
            src_dataset_dir (str): the root directory of dataset.

        Returns:
            str: the root directory of converted dataset.
        """
        dst_dataset_dir = src_dataset_dir
        return dst_dataset_dir

    def split_dataset(self, src_dataset_dir: str) -> str:
        """repartition the train and validation dataset

        Args:
            src_dataset_dir (str): the root directory of dataset.

        Returns:
            str: the root directory of splited dataset.
        """
        dst_dataset_dir = src_dataset_dir
        return dst_dataset_dir

    def analyse(self, dataset_dir: str) -> dict:
        """deep analyse dataset

        Args:
            dataset_dir (str): the root directory of dataset.

        Returns:
            dict: the deep analysis results.
        """
        return {}

    @abstractmethod
    def get_show_type(self):
        """return the dataset show type

        Raises:
            NotImplementedError
        """
        raise NotImplementedError

    @abstractmethod
    def get_dataset_type(self):
        """return the dataset type

        Raises:
            NotImplementedError
        """
        raise NotImplementedError
