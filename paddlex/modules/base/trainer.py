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
from pathlib import Path
from .build_model import build_model
from ...utils.device import (
    update_device_num,
    set_env_for_device,
    check_supported_device,
)
from ...utils.misc import AutoRegisterABCMetaClass
from ...utils.config import AttrDict
from ...utils.logging import info


def build_trainer(config: AttrDict) -> "BaseTrainer":
    """build model trainer

    Args:
        config (AttrDict): PaddleX pipeline config, which is loaded from pipeline yaml file.

    Returns:
        BaseTrainer: the trainer, which is subclass of BaseTrainer.
    """
    model_name = config.Global.model
    try:
        import feature_line_modules
    except ModuleNotFoundError:
        pass
    return BaseTrainer.get(model_name)(config)


class BaseTrainer(ABC, metaclass=AutoRegisterABCMetaClass):
    """Base Model Trainer"""

    __is_base = True

    def __init__(self, config: AttrDict):
        """Initialize the instance.

        Args:
            config (AttrDict):  PaddleX pipeline config, which is loaded from pipeline yaml file.
        """
        super().__init__()
        self.config = config
        self.global_config = config.Global
        self.train_config = config.Train
        self.eval_config = config.Evaluate
        self.benchmark_config = config.get("Benchmark", None)
        config_path = self.train_config.get("basic_config_path", None)

        self.pdx_config, self.pdx_model = build_model(
            self.global_config.model, config_path=config_path
        )

    def train(self, *args, **kwargs):
        """execute model training"""
        os.makedirs(self.global_config.output, exist_ok=True)
        self.update_config()
        self.dump_config()
        train_args = self.get_train_kwargs()
        if self.benchmark_config is not None:
            train_args.update({"benchmark": self.benchmark_config})
        export_with_pir = self.global_config.get("export_with_pir", False) or os.getenv(
            "FLAGS_json_format_model"
        ) in ["1", "True"]
        train_args.update(
            {
                "uniform_output_enabled": self.train_config.get(
                    "uniform_output_enabled", True
                ),
                "export_with_pir": export_with_pir,
            }
        )
        train_result = self.pdx_model.train(**train_args)
        assert (
            train_result.returncode == 0
        ), f"Encountered an unexpected error({train_result.returncode}) in \
training!"

    def dump_config(self, config_file_path: str = None):
        """dump the config

        Args:
            config_file_path (str, optional): the path to save dumped config. Defaults to None,
                means that save in `Global.output` as `config.yaml`.
        """
        if config_file_path is None:
            config_file_path = os.path.join(self.global_config.output, "config.yaml")
        self.pdx_config.dump(config_file_path)

    def get_device(self, using_device_number: int = None) -> str:
        """get device setting from config

        Args:
            using_device_number (int, optional): specify device number to use. Defaults to None,
                means that base on config setting.

        Returns:
            str: device setting, such as: `gpu:0,1`, `npu:0,1` `cpu`.
        """
        check_supported_device(self.global_config.device, self.global_config.model)
        set_env_for_device(self.global_config.device)
        device_setting = (
            update_device_num(self.global_config.device, using_device_number)
            if using_device_number
            else self.global_config.device
        )
        # replace "dcu" with "gpu"
        device_setting = device_setting.replace("dcu", "gpu")
        return device_setting

    @abstractmethod
    def update_config(self):
        """update training config"""
        raise NotImplementedError

    @abstractmethod
    def get_train_kwargs(self):
        """get key-value arguments of model training function"""
        raise NotImplementedError
