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
from .errors import raise_unsupported_device_error

SUPPORTED_DEVICE_TYPE = ["cpu", "gpu", "xpu", "npu", "mlu"]


def get_device(device_cfg, using_device_number=None):
    """get running device setting
    """
    device = device_cfg.split(":")[0]
    assert device.lower() in SUPPORTED_DEVICE_TYPE
    if device.lower() in ["gpu", "xpu", "npu", "mlu"]:
        if device.lower() == "npu":
            os.environ["FLAGS_npu_jit_compile"] = "0"
            os.environ["FLAGS_use_stride_kernel"] = "0"
            os.environ["FLAGS_allocator_strategy"] = "auto_growth"
        elif device.lower() == "mlu":
            os.environ["CUSTOM_DEVICE_BLACK_LIST"] = "set_value"
        if len(device_cfg.split(":")) == 2:
            device_ids = device_cfg.split(":")[1]
        else:
            device_ids = 0

        if using_device_number:
            device_ids = f"{device_ids[:using_device_number]}"
        return "{}:{}".format(device.lower(), device_ids)
    if device.lower() == "cpu":
        return "cpu"
    else:
        raise_unsupported_device_error(device, SUPPORTED_DEVICE_TYPE)
