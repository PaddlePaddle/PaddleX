# !/usr/bin/env python3
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2024 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Author: PaddlePaddle Authors
"""
import os
from abc import ABC, abstractmethod
from pathlib import Path
from .build_model import build_model
from .utils.device import get_device
from ...utils.misc import AutoRegisterABCMetaClass
from ...utils.config import AttrDict


def build_trainer(config: AttrDict) -> "BaseTrainer":
    """build model trainer

    Args:
        config (AttrDict): PaddleX pipeline config, which is loaded from pipeline yaml file.

    Returns:
        BaseTrainer: the trainer, which is subclass of BaseTrainer.
    """
    model_name = config.Global.model
    return BaseTrainer.get(model_name)(config)


class BaseTrainer(ABC, metaclass=AutoRegisterABCMetaClass):
    """ Base Model Trainer """
    __is_base = True

    def __init__(self, config: AttrDict):
        """Initialize the instance.

        Args:
            config (AttrDict):  PaddleX pipeline config, which is loaded from pipeline yaml file.
        """
        self.global_config = config.Global
        self.train_config = config.Train

        self.deamon = self.build_deamon(self.global_config)
        self.pdx_config, self.pdx_model = build_model(self.global_config.model)

    def __call__(self, *args, **kwargs):
        """execute model training
        """
        os.makedirs(self.global_config.output, exist_ok=True)
        self.train(*args, **kwargs)
        self.deamon.stop()

    def dump_config(self, config_file_path: str=None):
        """dump the config

        Args:
            config_file_path (str, optional): the path to save dumped config. Defaults to None,
                means that save in `Global.output` as `config.yaml`.
        """
        if config_file_path is None:
            config_file_path = os.path.join(self.global_config.output,
                                            "config.yaml")
        self.pdx_config.dump(config_file_path)

    def train(self):
        """firstly, update and dump train config, then train model
        """
        self.update_config()
        self.dump_config()
        train_result = self.pdx_model.train(**self.get_train_kwargs())
        assert train_result.returncode == 0, f"Encountered an unexpected error({train_result.returncode}) in \
training!"

    def get_device(self, using_gpu_number: int=None) -> str:
        """get device setting from config

        Args:
            using_gpu_number (int, optional): specify GPU number to use. Defaults to None,
                means that base on config setting.

        Returns:
            str: device setting, such as: `gpu:0,1`, `cpu`.
        """
        return get_device(
            self.global_config.device, using_gpu_number=using_gpu_number)

    @abstractmethod
    def build_deamon(self):
        """build deamon thread for saving training outputs timely
        """
        raise NotImplementedError

    @abstractmethod
    def update_config(self):
        """update training config
        """
        raise NotImplementedError

    @abstractmethod
    def get_train_kwargs(self):
        """get key-value arguments of model training function
        """
        raise NotImplementedError
