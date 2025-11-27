# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from ...utils.device import get_default_device, parse_device
from ...utils.env import get_device_type


def is_mkldnn_available():
    # XXX: Not sure if this is the best way to check if MKL-DNN is available
    from paddle.inference import Config

    return hasattr(Config, "set_mkldnn_cache_capacity")


def is_bfloat16_available(device):
    import paddle.amp

    if device is None:
        device = get_default_device()
    device_type, _ = parse_device(device)
    return (
        "npu" in get_device_type() or paddle.amp.is_bfloat16_supported()
    ) and device_type in ("gpu", "npu", "xpu", "mlu")


def is_float16_available(device):
    import paddle.amp

    if device is None:
        device = get_default_device()
    device_type, _ = parse_device(device)
    return (
        "npu" in get_device_type() or paddle.amp.is_float16_supported()
    ) and device_type in ("gpu", "npu", "xpu", "mlu", "dcu")
