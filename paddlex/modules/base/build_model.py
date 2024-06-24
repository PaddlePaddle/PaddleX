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
from ...repo_apis.base import Config, PaddleModel
from ...utils.device import get_device


def build_model(model_name: str, device: str=None,
                config_path: str=None) -> tuple:
    """build Config and PaddleModel

    Args:
        model_name (str): model name
        device (str): device, such as gpu, cpu, npu, xpu, mlu
        config_path (str, optional): path to the PaddleX config yaml file.
            Defaults to None, i.e. using the default config file.

    Returns:
        tuple(Config, PaddleModel): the Config and PaddleModel
    """
    config = Config(model_name, config_path)

    if device:
        config.update_device(get_device(device))
    model = PaddleModel(config=config)
    return config, model
