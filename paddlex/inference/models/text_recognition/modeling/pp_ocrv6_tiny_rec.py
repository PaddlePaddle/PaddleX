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
from ...image_classification.modeling.pplcnetv4 import PPLCNetV4Backbone
from ._config_pp_ocrv6_tiny_rec import PPOCRV6TinyRecConfig


class PPOCRV6TinyRecHead(nn.Layer):
    def __init__(self, config):
        super().__init__()
        in_channels = config.backbone_config.block_configs[-1][-1][2]
        mid_channels = config.hidden_size

        self.conv1 = nn.Conv1D(
            in_channels,
            in_channels,
            kernel_size=5,
            padding=2,
            groups=in_channels,
            bias_attr=False,
        )
        self.norm1 = nn.BatchNorm1D(in_channels)
        self.conv2 = nn.Conv1D(
            in_channels,
            in_channels,
            kernel_size=1,
            bias_attr=False,
        )
        self.norm2 = nn.BatchNorm1D(in_channels)
        self.act_fn = nn.Hardswish()
        self.fc1 = nn.Linear(in_channels, mid_channels)
        self.fc2 = nn.Linear(mid_channels, config.head_out_channels)

    def forward(self, hidden_states):
        hidden_states = hidden_states.squeeze(2)
        hidden_states = self.act_fn(self.norm1(self.conv1(hidden_states)))
        hidden_states = self.act_fn(self.norm2(self.conv2(hidden_states)))

        hidden_states = hidden_states.transpose([0, 2, 1])
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.softmax(hidden_states, axis=2)
        return hidden_states


class PPOCRV6TinyRecModel(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.backbone = PPLCNetV4Backbone(config.backbone_config)

    def forward(self, pixel_values):
        backbone_outputs = self.backbone(pixel_values)
        hidden_state = backbone_outputs[-1]
        hidden_state = F.avg_pool2d(hidden_state, (3, 2))
        return hidden_state


class PPOCRV6TinyRec(BatchNormHFStateDictMixin, PretrainedModel):
    config_class = PPOCRV6TinyRecConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = PPOCRV6TinyRecModel(config)
        self.head = PPOCRV6TinyRecHead(config)

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
