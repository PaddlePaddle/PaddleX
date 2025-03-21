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

import json
import platform
from functools import lru_cache
from typing import Optional

from importlib_resources import files
from paddlex_hpi._utils.typing import DeviceType

from paddlex.utils import logging

_DB_PATH: str = "model_info_collection.json"


@lru_cache(1)
def _get_model_info_collection() -> dict:
    with files("paddlex_hpi").joinpath(_DB_PATH).open("r", encoding="utf-8") as f:
        _model_info_collection = json.load(f)
    return _model_info_collection


def get_model_info(model_name: str, device_type: DeviceType) -> Optional[dict]:
    # TODO: Typed model info and nearest referents
    model_info_collection = _get_model_info_collection()
    uname = platform.uname()
    arch = uname.machine.lower()
    if arch not in model_info_collection:
        return None
    logging.debug("Getting model information for arch: %s", arch)
    model_info_collection = model_info_collection[arch]
    os = uname.system.lower()
    if os not in model_info_collection:
        return None
    logging.debug("Getting model information for OS: %s", os)
    model_info_collection = model_info_collection[os]
    if device_type == "cpu":
        device = "cpu"
    elif device_type == "gpu":
        device = "gpu_cuda118_cudnn86"
    else:
        return None
    logging.debug("Getting model information for device: %s", device)
    model_info_collection = model_info_collection[device]
    if model_name not in model_info_collection:
        return None
    return model_info_collection[model_name]
