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

import contextlib
import os
from typing import Optional, Union

import paddle

ASYMMETRY_QUANT_SCALE_MIN = "@min_scales"
ASYMMETRY_QUANT_SCALE_MAX = "@max_scales"
SYMMETRY_QUANT_SCALE = "@scales"
CONFIG_NAME = "config.json"
LEGACY_CONFIG_NAME = "model_config.json"
PADDLE_WEIGHTS_NAME = "model_state.pdparams"
PADDLE_WEIGHTS_INDEX_NAME = "model_state.pdparams.index.json"
PYTORCH_WEIGHTS_INDEX_NAME = "pytorch_model.bin.index.json"
PYTORCH_WEIGHTS_NAME = "pytorch_model.bin"
SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
SAFE_WEIGHTS_NAME = "model.safetensors"
GENERATION_CONFIG_NAME = "generation_config.json"


def resolve_file_path(
    pretrained_model_name_or_path: str = None,
    filenames: Union[str, list] = None,
    subfolder: Optional[str] = None,
    **kwargs,
):
    """
    This is a load function, mainly called by the from_pretrained function.
    Adapt for PaddleX inference.
    """
    assert (
        pretrained_model_name_or_path is not None
    ), "pretrained_model_name_or_path cannot be None"
    assert filenames is not None, "filenames cannot be None"
    subfolder = subfolder if subfolder is not None else ""

    if isinstance(filenames, str):
        filenames = [filenames]

    if os.path.isfile(pretrained_model_name_or_path):
        return pretrained_model_name_or_path

    elif os.path.isdir(pretrained_model_name_or_path):
        for index, filename in enumerate(filenames):
            if os.path.exists(
                os.path.join(pretrained_model_name_or_path, subfolder, filename)
            ):
                if not os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, subfolder, filename)
                ):
                    raise EnvironmentError(
                        f"{pretrained_model_name_or_path} does not appear to have file named {filename}."
                    )
                return os.path.join(pretrained_model_name_or_path, subfolder, filename)
            elif index < len(filenames) - 1:
                continue
            else:
                raise FileNotFoundError(
                    f"please make sure one of the {filenames} under the dir {pretrained_model_name_or_path}"
                )

    else:
        raise ValueError(
            "please make sure `pretrained_model_name_or_path` is either a file or a directory."
        )


@contextlib.contextmanager
def device_guard(device="cpu", dev_id=0):
    origin_device = paddle.device.get_device()
    if device == "cpu":
        paddle.set_device(device)
    elif device in ["gpu", "xpu", "npu"]:
        paddle.set_device("{}:{}".format(device, dev_id))
    try:
        yield
    finally:
        paddle.set_device(origin_device)


def get_env_device():
    """
    Return the device name of running environment.
    """
    if paddle.is_compiled_with_cuda():
        return "gpu"
    elif "npu" in paddle.device.get_all_custom_device_type():
        return "npu"
    elif "gcu" in paddle.device.get_all_custom_device_type():
        return "gcu"
    elif paddle.is_compiled_with_rocm():
        return "rocm"
    elif paddle.is_compiled_with_xpu():
        return "xpu"
    return "cpu"
