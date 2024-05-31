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


def build_model(model_name: str, config_path: str=None) -> tuple:
    """build Config and PaddleModel

    Args:
        model_name (str): model name
        config_path (str, optional): path to the PaddleX config yaml file.
            Defaults to None, i.e. using the default config file.

    Returns:
        tuple(Config, PaddleModel): the Config and PaddleModel
    """
    config = Config(model_name, config_path)

    # NOTE(gaotingquan): aistudio can only support 4GB shm when single gpu can be used.
    device = os.environ.get("DEVICE", None)
    if device:
        if device.lower() == "cpu" or (
                device.lower() == "gpu" and
                os.environ.get("GPU_NUMBER", None) == "1"):
            if hasattr(config, "disable_shared_memory"):
                config.disable_shared_memory()
            if hasattr(config, "update_num_workers"):
                config.update_num_workers(2)
    model = PaddleModel(config=config)
    return config, model
