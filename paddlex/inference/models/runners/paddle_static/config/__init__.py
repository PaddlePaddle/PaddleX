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

"""Paddle static inference config."""

from .blocklists import MKLDNN_BLOCKLIST, NEWIR_BLOCKLIST, TRT_BLOCKLIST
from .pp_option import PaddlePredictorOption, get_default_run_mode
from .trt_config import DISABLE_TRT_HALF_OPS_CONFIG, TRT_CFG_SETTING, TRT_PRECISION_MAP

__all__ = [
    "MKLDNN_BLOCKLIST",
    "NEWIR_BLOCKLIST",
    "TRT_BLOCKLIST",
    "PaddlePredictorOption",
    "get_default_run_mode",
    "DISABLE_TRT_HALF_OPS_CONFIG",
    "TRT_CFG_SETTING",
    "TRT_PRECISION_MAP",
]
