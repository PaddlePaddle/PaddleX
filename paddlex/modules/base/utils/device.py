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
from ....utils.errors import raise_unsupported_device_error

SUPPORTED_DEVICE_TYPE = ["cpu", "gpu", "xpu", "npu", "mlu"]


def get_device(device_cfg, using_gpu_number=None):
    """get running device setting
    """
    device = device_cfg.split(":")[0]
    assert device.lower() in SUPPORTED_DEVICE_TYPE
    if device.lower() in ["gpu", "xpu", "npu", "mlu"]:
        device_ids = device_cfg.split(":")[1]
        if using_gpu_number:
            device_ids = f"{device_ids[0]}"
        return "{}:{}".format(device.lower(), device_ids)
    if device.lower() == "cpu":
        return "cpu"
    else:
        raise_unsupported_device_error(device, SUPPORTED_DEVICE_TYPE)
