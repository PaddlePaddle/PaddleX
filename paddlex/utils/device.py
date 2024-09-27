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
import GPUtil
import lazy_paddle as paddle
from .errors import raise_unsupported_device_error

SUPPORTED_DEVICE_TYPE = ["cpu", "gpu", "xpu", "npu", "mlu"]


def _constr_device(device_type, device_ids):
    if device_ids:
        device_ids = ",".join(map(str, device_ids))
        return f"{device_type}:{device_ids}"
    else:
        return f"{device_type}"


def get_default_device():
    avail_gpus = GPUtil.getAvailable()
    if not avail_gpus:
        return "cpu"
    else:
        return _constr_device("gpu", [avail_gpus[0]])


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
    return device_type, device_ids


def update_device_num(device, num):
    device_type, device_ids = parse_device(device)
    if device_ids:
        assert len(device_ids) >= num
        return _constr_device(device_type, device_ids[:num])
    else:
        return _constr_device(device_type, device_ids)


def set_env_for_device(device):
    device_type, device_ids = parse_device(device)
    if device_type.lower() in ["gpu", "xpu", "npu", "mlu"]:
        if device_type.lower() == "gpu" and paddle.is_compiled_with_rocm():
            os.environ["FLAGS_conv_workspace_size_limit"] = "2000"
        if device_type.lower() == "npu":
            os.environ["FLAGS_npu_jit_compile"] = "0"
            os.environ["FLAGS_use_stride_kernel"] = "0"
            os.environ["FLAGS_allocator_strategy"] = "auto_growth"
            os.environ["CUSTOM_DEVICE_BLACK_LIST"] = (
                "pad3d,pad3d_grad,set_value,set_value_with_tensor"
            )
            os.environ["FLAGS_npu_scale_aclnn"] = "True"
            os.environ["FLAGS_npu_split_aclnn"] = "True"
        if device_type.lower() == "xpu":
            os.environ["BKCL_FORCE_SYNC"] = "1"
            os.environ["BKCL_TIMEOUT"] = "1800"
            os.environ["FLAGS_use_stride_kernel"] = "0"
        if device_type.lower() == "mlu":
            os.environ["FLAGS_use_stride_kernel"] = "0"
