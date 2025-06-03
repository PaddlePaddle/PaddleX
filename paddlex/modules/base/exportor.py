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

from abc import ABC
from pathlib import Path

from ...utils import logging
from ...utils.config import AttrDict
from ...utils.device import (
    check_supported_device,
    set_env_for_device,
    update_device_num,
)
from ...utils.flags import FLAGS_json_format_model
from ...utils.misc import AutoRegisterABCMetaClass
from .build_model import build_model


def build_exportor(config: AttrDict) -> "BaseExportor":
    """build model exportor

    Args:
        config (AttrDict): PaddleX pipeline config, which is loaded from pipeline yaml file.

    Returns:
        BaseExportor: the exportor, which is subclass of BaseExportor.
    """
    model_name = config.Global.model
    try:
        pass
    except ModuleNotFoundError:
        pass
    return BaseExportor.get(model_name)(config)


class BaseExportor(ABC, metaclass=AutoRegisterABCMetaClass):
    """Base Model Exportor"""

    __is_base = True

    def __init__(self, config):
        """Initialize the instance.

        Args:
            config (AttrDict):  PaddleX pipeline config, which is loaded from pipeline yaml file.
        """
        super().__init__()
        self.global_config = config.Global
        self.export_config = config.Export

        config_path = self.get_config_path(self.export_config.weight_path)
        if self.export_config.get("basic_config_path", None):
            config_path = self.export_config.get("basic_config_path", None)

        self.pdx_config, self.pdx_model = build_model(
            self.global_config.model, config_path=config_path
        )

    def get_config_path(self, weight_path):
        """
        get config path

        Args:
            weight_path (str): The path to the weight

        Returns:
            config_path (str): The path to the config

        """

        config_path = Path(weight_path).parent / "config.yaml"
        # `Path("https://xxx/xxx")` would cause error on Windows
        try:
            is_exists = config_path.exists()
        except Exception:
            is_exists = False
        if not is_exists:
            logging.warning(
                f"The config file(`{config_path}`) related to weight file(`{weight_path}`) is not exist, use default instead."
            )
            config_path = None

        return config_path

    def export(self) -> dict:
        """execute model exporting

        Returns:
            dict: the export metrics
        """
        self.update_config()
        export_result = self.pdx_model.export(**self.get_export_kwargs())
        assert (
            export_result.returncode == 0
        ), f"Encountered an unexpected error({export_result.returncode}) in \
exporting!"

        return None

    def get_device(self, using_device_number: int = None) -> str:
        """get device setting from config

        Args:
            using_device_number (int, optional): specify device number to use.
                Defaults to None, means that base on config setting.

        Returns:
            str: device setting, such as: `gpu:0,1`, `npu:0,1`, `cpu`.
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

    def update_config(self):
        """update export config"""

    def get_export_kwargs(self):
        """get key-value arguments of model export function"""
        export_with_pir = (
            self.global_config.get("export_with_pir", False) or FLAGS_json_format_model
        )
        return {
            "weight_path": self.export_config.weight_path,
            "save_dir": self.global_config.output,
            "device": self.get_device(1),
            "export_with_pir": export_with_pir,
        }
