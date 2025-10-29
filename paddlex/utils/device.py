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

import os
from contextlib import ContextDecorator

from . import logging
from .custom_device_list import (
    DCU_WHITELIST,
    GCU_WHITELIST,
    MLU_WHITELIST,
    NPU_BLACKLIST,
    XPU_WHITELIST,
)
from .flags import DISABLE_DEV_MODEL_WL

SUPPORTED_DEVICE_TYPE = ["cpu", "gpu", "xpu", "npu", "mlu", "gcu", "dcu", "iluvatar_gpu"]


def constr_device(device_type, device_ids):
    if device_type == "cpu" and device_ids is not None:
        raise ValueError("`device_ids` must be None for CPUs")
    if device_ids:
        device_ids = ",".join(map(str, device_ids))
        return f"{device_type}:{device_ids}"
    else:
        return f"{device_type}"


def get_default_device():
    import paddle

    if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
        return constr_device("gpu", [0])
    else:
        return "cpu"


def parse_device(device):
    """parse_device"""
    # According to https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/device/set_device_cn.html
    parts = device.split(":")
    if len(parts) > 2:
        raise ValueError(f"Invalid device: {device}")
    if len(parts) == 1:
        device_type, device_ids = parts[0], None
    else:
        device_type, device_ids = parts
        device_ids = device_ids.split(",")
        for device_id in device_ids:
            if not device_id.isdigit():
                raise ValueError(
                    f"Device ID must be an integer. Invalid device ID: {device_id}"
                )
        device_ids = list(map(int, device_ids))
    device_type = device_type.lower()
    # raise_unsupported_device_error(device_type, SUPPORTED_DEVICE_TYPE)
    assert device_type.lower() in SUPPORTED_DEVICE_TYPE
    if device_type == "cpu" and device_ids is not None:
        raise ValueError("No Device ID should be specified for CPUs")
    return device_type, device_ids


def update_device_num(device, num):
    device_type, device_ids = parse_device(device)
    if device_ids:
        assert len(device_ids) >= num
        return constr_device(device_type, device_ids[:num])
    else:
        return constr_device(device_type, device_ids)


def set_env_for_device(device):
    device_type, _ = parse_device(device)
    return set_env_for_device_type(device_type)


def set_env_for_device_type(device_type):
    import paddle

    def _set(envs):
        for key, val in envs.items():
            os.environ[key] = val
            logging.debug(f"{key} has been set to {val}.")

    # XXX: is_compiled_with_rocm() must be True on dcu platform ?
    if device_type.lower() == "dcu" and paddle.is_compiled_with_rocm():
        envs = {"FLAGS_conv_workspace_size_limit": "2000"}
        _set(envs)
    if device_type.lower() == "npu":
        envs = {
            "FLAGS_npu_jit_compile": "0",
            "FLAGS_use_stride_kernel": "0",
            "FLAGS_allocator_strategy": "auto_growth",
            "CUSTOM_DEVICE_BLACK_LIST": "pad3d,pad3d_grad,set_value,set_value_with_tensor",
            "FLAGS_npu_scale_aclnn": "True",
            "FLAGS_npu_split_aclnn": "True",
        }
        _set(envs)
    if device_type.lower() == "xpu":
        envs = {
            "BKCL_FORCE_SYNC": "1",
            "BKCL_TIMEOUT": "1800",
            "FLAGS_use_stride_kernel": "0",
            "XPU_BLACK_LIST": "pad3d",
        }
        _set(envs)
    if device_type.lower() == "mlu":
        envs = {
            "FLAGS_use_stride_kernel": "0",
            "FLAGS_use_stream_safe_cuda_allocator": "0",
        }
        _set(envs)
    if device_type.lower() == "gcu":
        envs = {"FLAGS_use_stride_kernel": "0"}
        _set(envs)


def check_supported_device_type(device_type, model_name):
    if DISABLE_DEV_MODEL_WL:
        logging.warning(
            "Skip checking if model is supported on device because the flag `PADDLE_PDX_DISABLE_DEV_MODEL_WL` has been set."
        )
        return
    tips = "You could set env `PADDLE_PDX_DISABLE_DEV_MODEL_WL` to `true` to disable this checking."
    if device_type == "dcu":
        assert model_name in DCU_WHITELIST, (
            f"The DCU device does not yet support `{model_name}` model!" + tips
        )
    elif device_type == "mlu":
        assert model_name in MLU_WHITELIST, (
            f"The MLU device does not yet support `{model_name}` model!" + tips
        )
    elif device_type == "npu":
        assert model_name not in NPU_BLACKLIST, (
            f"The NPU device does not yet support `{model_name}` model!" + tips
        )
    elif device_type == "xpu":
        assert model_name in XPU_WHITELIST, (
            f"The XPU device does not yet support `{model_name}` model!" + tips
        )
    elif device_type == "gcu":
        assert model_name in GCU_WHITELIST, (
            f"The GCU device does not yet support `{model_name}` model!" + tips
        )


def check_supported_device(device, model_name):
    device_type, _ = parse_device(device)
    return check_supported_device_type(device_type, model_name)


class TemporaryDeviceChanger(ContextDecorator):
    """
    A context manager to temporarily change global device
    """

    def __init__(self, new_device):
        # if new_device is None, nothing changed
        import paddle

        self.new_device = new_device
        self.original_device = paddle.device.get_device()

    def __enter__(self):
        import paddle

        if self.new_device is None:
            return self
        paddle.device.set_device(self.new_device)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import paddle

        if self.new_device is None:
            return False
        paddle.device.set_device(self.original_device)
        return False
