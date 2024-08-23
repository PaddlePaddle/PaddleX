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
import paddle
from .errors import raise_unsupported_device_error

SUPPORTED_DEVICE_TYPE = ["cpu", "gpu", "xpu", "npu", "mlu"]


def get_device(device_cfg, using_device_number=None):
    """get running device setting
    """
    device = device_cfg.split(":")[0]
    assert device.lower() in SUPPORTED_DEVICE_TYPE
    if device.lower() in ["gpu", "xpu", "npu", "mlu"]:
        if device.lower() == "gpu" and paddle.is_compiled_with_rocm():
            os.environ['FLAGS_conv_workspace_size_limit'] = '2000'
        if device.lower() == "npu":
            os.environ["FLAGS_npu_jit_compile"] = "0"
            os.environ["FLAGS_use_stride_kernel"] = "0"
            os.environ["FLAGS_allocator_strategy"] = "auto_growth"
            os.environ[
                "CUSTOM_DEVICE_BLACK_LIST"] = "pad3d,pad3d_grad,set_value,set_value_with_tensor"
            os.environ["FLAGS_npu_scale_aclnn"] = "True"
            os.environ["FLAGS_npu_split_aclnn"] = "True"
        if device.lower() == "xpu":
            os.environ["BKCL_FORCE_SYNC"] = "1"
            os.environ["BKCL_TIMEOUT"] = "1800"
            os.environ["FLAGS_use_stride_kernel"] = "0"
        if device.lower() == "mlu":
            os.environ["FLAGS_use_stride_kernel"] = "0"

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
