# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

from ...common.transformers.transformers import PretrainedConfig

DEFAULT_BLOCK_CONFIGS = [
    # Stage 1 (blocks2)
    [[3, 16, 32, 1, False]],
    # Stage 2 (blocks3)
    [[3, 32, 64, 2, False], [3, 64, 64, 1, False]],
    # Stage 3 (blocks4)
    [[3, 64, 128, 2, False], [3, 128, 128, 1, False]],
    # Stage 4 (blocks5)
    [
        [3, 128, 256, 2, False],
        [5, 256, 256, 1, False],
        [5, 256, 256, 1, False],
        [5, 256, 256, 1, False],
        [5, 256, 256, 1, False],
    ],
    # Stage 5 (blocks6)
    [
        [5, 256, 512, 2, True],
        [5, 512, 512, 1, True],
        [5, 512, 512, 1, False],
        [5, 512, 512, 1, False],
    ],
]


class PPLCNetV3Config(PretrainedConfig):
    model_type = "pp_lcnet_v3"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scale = kwargs.get("scale", 1.0)
        self.block_configs = kwargs.get("block_configs", DEFAULT_BLOCK_CONFIGS)
        self.stem_channels = kwargs.get("stem_channels", 16)
        self.stem_stride = kwargs.get("stem_stride", 2)
        self.reduction = kwargs.get("reduction", 4)
        self.divisor = kwargs.get("divisor", 8)
        self.hidden_act = kwargs.get("hidden_act", "hardswish")
        self.conv_symmetric_num = kwargs.get("conv_symmetric_num", 4)
