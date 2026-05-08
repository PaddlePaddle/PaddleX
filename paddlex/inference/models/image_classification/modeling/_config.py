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

from __future__ import annotations

from ...common.transformers.transformers import PretrainedConfig


class PPLCNetConfig(PretrainedConfig):
    model_type = "pp_lcnet"
    scale: float | int = 1.0
    block_configs: list | None = None
    stem_channels: int = 16
    stem_stride: int = 2
    reduction: int = 4
    class_expand: int = 1280
    divisor: int = 8
    hidden_act: str = "hardswish"
    hidden_dropout_prob: float = 0.2

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.block_configs = (
            [
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
                    [5, 256, 256, 1, False],
                ],
                # Stage 5 (blocks6)
                [[5, 256, 512, 2, True], [5, 512, 512, 1, True]],
            ]
            if self.block_configs is None
            else self.block_configs
        )

        self.depths = [len(blocks) for blocks in self.block_configs]
        self.stage_names = ["stem"] + [
            f"stage{idx}" for idx in range(1, len(self.block_configs) + 1)
        ]
