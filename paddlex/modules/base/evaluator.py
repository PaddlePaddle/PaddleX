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
from pathlib import Path
from abc import ABC, abstractmethod

from .build_model import build_model
from ...utils.device import get_device
from ...utils.misc import AutoRegisterABCMetaClass
from ...utils.config import AttrDict
from ...utils.logging import *


def build_evaluater(config: AttrDict) -> "BaseEvaluator":
    """build model evaluater

    Args:
        config (AttrDict): PaddleX pipeline config, which is loaded from pipeline yaml file.

    Returns:
        BaseEvaluator: the evaluater, which is subclass of BaseEvaluator.
    """
    model_name = config.Global.model
    return BaseEvaluator.get(model_name)(config)


class BaseEvaluator(ABC, metaclass=AutoRegisterABCMetaClass):
    """ Base Model Evaluator """

    __is_base = True

    def __init__(self, config):
        """Initialize the instance.

        Args:
            config (AttrDict):  PaddleX pipeline config, which is loaded from pipeline yaml file.
        """
        self.global_config = config.Global
        self.eval_config = config.Evaluate

        config_path = self.get_config_path(self.eval_config.weight_path)
        if not config_path.exists():
            warning(
                f"The config file(`{config_path}`) related to weight file(`{self.eval_config.weight_path}`) is not exist, use default instead."
            )
            config_path = None
        self.pdx_config, self.pdx_model = build_model(
            self.global_config.model, config_path=config_path)

    def get_config_path(self, weight_path):
        """
        get config path

        Args:
            weight_path (str): The path to the weight

        Returns:
            config_path (str): The path to the config

        """

        config_path = Path(weight_path).parent / "config.yaml"

        return config_path

    def check_return(self, metrics: dict) -> bool:
        """check evaluation metrics

        Args:
            metrics (dict): evaluation output metrics

        Returns:
            bool: whether the format of evaluation metrics is legal
        """
        if not isinstance(metrics, dict):
            return False
        for metric in metrics:
            val = metrics[metric]
            if not isinstance(val, float):
                return False
        return True

    def __call__(self) -> dict:
        """execute model training

        Returns:
            dict: the evaluation metrics
        """
        metrics = self.eval()
        assert self.check_return(
            metrics
        ), f"The return value({metrics}) of Evaluator.eval() is illegal!"
        return {"metrics": metrics}

    def dump_config(self, config_file_path=None):
        """dump the config

        Args:
            config_file_path (str, optional): the path to save dumped config.
                Defaults to None, means that save in `Global.output` as `config.yaml`.
        """
        if config_file_path is None:
            config_file_path = os.path.join(self.global_config.output,
                                            "config.yaml")
        self.pdx_config.dump(config_file_path)

    def eval(self):
        """firstly, update evaluation config, then evaluate model, finally return the evaluation result
        """
        self.update_config()
        # self.dump_config()
        evaluate_result = self.pdx_model.evaluate(**self.get_eval_kwargs())
        assert evaluate_result.returncode == 0, f"Encountered an unexpected error({evaluate_result.returncode}) in \
evaling!"

        return evaluate_result.metrics

    def get_device(self, using_device_number: int=None) -> str:
        """get device setting from config

        Args:
            using_device_number (int, optional): specify device number to use.
                Defaults to None, means that base on config setting.

        Returns:
            str: device setting, such as: `gpu:0,1`, `npu:0,1`, `cpu`.
        """
        return get_device(
            self.global_config.device, using_device_number=using_device_number)

    @abstractmethod
    def update_config(self):
        """update evalution config
        """
        raise NotImplementedError

    @abstractmethod
    def get_eval_kwargs(self):
        """get key-value arguments of model evalution function
        """
        raise NotImplementedError
