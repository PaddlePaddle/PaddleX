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

from typing import List

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ...common.transformers.transformers import (
    BatchNormHFStateDictMixin,
    PretrainedModel,
)
from ...image_classification.modeling.pplcnetv3 import PPLCNetV3Backbone, make_divisible
from ._config_pp_ocrv5_mobile_rec import PPOCRV5MobileRecConfig
from .pp_ocrv5_server_rec import PPOCRV5ServerRecBlock, PPOCRV5ServerRecConvLayer


class PPOCRV5MobileRecEncoderWithSVTR(nn.Layer):
    def __init__(self, config):
        super().__init__()
        in_channels = make_divisible(
            config.backbone_config.block_configs[-1][-1][2]
            * config.backbone_config.scale,
            config.backbone_config.divisor,
        )
        hidden_size = config.hidden_size

        self.conv_block = nn.LayerList(
            [
                PPOCRV5ServerRecConvLayer(
                    in_channels=in_channels,
                    out_channels=in_channels // 8,
                    kernel_size=config.conv_kernel_size,
                ),
                PPOCRV5ServerRecConvLayer(
                    in_channels=in_channels // 8,
                    out_channels=hidden_size,
                    kernel_size=(1, 1),
                ),
                PPOCRV5ServerRecConvLayer(
                    in_channels=hidden_size,
                    out_channels=in_channels,
                    kernel_size=(1, 1),
                ),
                PPOCRV5ServerRecConvLayer(
                    in_channels=2 * in_channels,
                    out_channels=in_channels // 8,
                    kernel_size=config.conv_kernel_size,
                ),
                PPOCRV5ServerRecConvLayer(
                    in_channels=in_channels // 8,
                    out_channels=hidden_size,
                    kernel_size=(1, 1),
                ),
            ]
        )
        self.svtr_block = nn.LayerList(
            [PPOCRV5ServerRecBlock(config=config) for _ in range(config.depth)]
        )
        self.norm = nn.LayerNorm(hidden_size, epsilon=config.layer_norm_eps)

    def forward(self, hidden_states):
        residual = hidden_states

        hidden_states = self.conv_block[0](hidden_states)
        hidden_states = self.conv_block[1](hidden_states)

        batch_size, channels, height, width = hidden_states.shape
        hidden_states = hidden_states.flatten(2).transpose([0, 2, 1])
        for block in self.svtr_block:
            hidden_states = block(hidden_states)

        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.reshape(
            [batch_size, height, width, channels]
        ).transpose([0, 3, 1, 2])
        hidden_states = self.conv_block[2](hidden_states)
        hidden_states = self.conv_block[3](
            paddle.concat([residual, hidden_states], axis=1)
        )
        hidden_states = self.conv_block[4](hidden_states)
        hidden_states = hidden_states.squeeze(2).transpose([0, 2, 1])

        return hidden_states


class PPOCRV5MobileRecHead(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.encoder = PPOCRV5MobileRecEncoderWithSVTR(config)
        self.head = nn.Linear(config.hidden_size, config.head_out_channels)

    def forward(self, hidden_states):
        hidden_states = self.encoder(hidden_states)
        hidden_states = self.head(hidden_states)
        hidden_states = F.softmax(hidden_states, axis=2)
        return hidden_states


class PPOCRV5MobileRecModel(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.backbone = PPLCNetV3Backbone(config.backbone_config)

    def forward(self, pixel_values):
        backbone_outputs = self.backbone(pixel_values)
        hidden_state = backbone_outputs[-1]
        hidden_state = F.avg_pool2d(hidden_state, (3, 2))
        return hidden_state


class PPOCRV5MobileRec(BatchNormHFStateDictMixin, PretrainedModel):
    config_class = PPOCRV5MobileRecConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = PPOCRV5MobileRecModel(config)
        self.head = PPOCRV5MobileRecHead(config)

    def forward(self, x: List) -> List:
        x = paddle.to_tensor(x[0])
        hidden_state = self.model(x)
        output = self.head(hidden_state)
        return [output.cpu().numpy()]

    def get_transpose_weight_keys(self):
        keys = []
        for key, param in self.get_hf_state_dict().items():
            if key.endswith("weight") and len(param.shape) == 2:
                keys.append(key)
        return keys
