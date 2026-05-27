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

from ...common.transformers.activations import ACT2FN
from ...common.transformers.transformers import (
    BatchNormHFStateDictMixin,
    PretrainedModel,
)
from ...image_classification.modeling.pplcnetv4 import PPLCNetV4Backbone
from ._config_pp_ocrv6_small_rec import PPOCRV6SmallRecConfig


class PPOCRV6SmallRecConvLayer(nn.Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(3, 3),
        stride=1,
        activation="silu",
        groups=1,
    ):
        super().__init__()
        self.convolution = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size[0] // 2, kernel_size[1] // 2),
            bias_attr=False,
            groups=groups,
        )
        self.normalization = nn.BatchNorm2D(out_channels)
        self.activation = (
            ACT2FN[activation] if activation is not None else nn.Identity()
        )

    def forward(self, hidden_states):
        hidden_states = self.convolution(hidden_states)
        hidden_states = self.normalization(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class PPOCRV6SmallRecMLP(nn.Layer):
    def __init__(self, config, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.activation = ACT2FN[config.hidden_act]
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(0.0)

    def forward(self, hidden_state):
        hidden_state = self.fc1(hidden_state)
        hidden_state = self.activation(hidden_state)
        hidden_state = self.drop(hidden_state)
        hidden_state = self.fc2(hidden_state)
        hidden_state = self.drop(hidden_state)
        return hidden_state


class PPOCRV6SmallRecAttention(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5

        if config.qkv_bias:
            self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim)
        else:
            self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias_attr=False)

        self.projection = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states):
        bsz, tgt_len, embed_dim = hidden_states.shape

        mixed_qkv = self.qkv(hidden_states)
        mixed_qkv = mixed_qkv.reshape(
            [bsz, tgt_len, 3, self.num_heads, self.head_dim]
        )
        mixed_qkv = mixed_qkv.transpose([2, 0, 3, 1, 4])
        query_states = mixed_qkv[0]
        key_states = mixed_qkv[1]
        value_states = mixed_qkv[2]

        attn_weights = paddle.matmul(query_states, key_states, transpose_y=True)
        attn_weights = attn_weights * self.scale
        attn_weights = F.softmax(attn_weights, axis=-1)

        attn_output = paddle.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose([0, 2, 1, 3])
        attn_output = attn_output.reshape([bsz, tgt_len, embed_dim])
        attn_output = self.projection(attn_output)
        return attn_output


class PPOCRV6SmallRecBlock(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = PPOCRV6SmallRecAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, epsilon=config.layer_norm_eps)
        self.mlp = PPOCRV6SmallRecMLP(
            config=config,
            in_features=self.embed_dim,
            hidden_features=int(self.embed_dim * config.mlp_ratio),
        )
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, epsilon=config.layer_norm_eps)

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class PPOCRV6SmallRecEncoderWithSVTR(nn.Layer):
    def __init__(self, config):
        super().__init__()
        in_channels = config.backbone_config.block_configs[-1][-1][2]
        hidden_size = config.hidden_size

        self.conv_block = nn.LayerList(
            [
                PPOCRV6SmallRecConvLayer(
                    in_channels=in_channels,
                    out_channels=hidden_size,
                    kernel_size=(1, 1),
                    activation=config.hidden_act,
                ),
                PPOCRV6SmallRecConvLayer(
                    in_channels=in_channels,
                    out_channels=hidden_size,
                    kernel_size=(1, 1),
                    activation=config.hidden_act,
                ),
                PPOCRV6SmallRecConvLayer(
                    in_channels=hidden_size,
                    out_channels=hidden_size,
                    kernel_size=config.conv_kernel_size,
                    activation=config.hidden_act,
                    groups=hidden_size,
                ),
            ]
        )
        self.svtr_block = nn.LayerList(
            [PPOCRV6SmallRecBlock(config=config) for _ in range(config.depth)]
        )
        self.norm = nn.LayerNorm(hidden_size, epsilon=config.layer_norm_eps)

    def forward(self, hidden_states):
        residual = self.conv_block[0](hidden_states)

        hidden_states = self.conv_block[1](hidden_states)
        hidden_states = hidden_states + self.conv_block[2](hidden_states)

        batch_size, channels, height, width = hidden_states.shape
        hidden_states = hidden_states.flatten(2).transpose([0, 2, 1])
        for block in self.svtr_block:
            hidden_states = block(hidden_states)

        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.reshape(
            [batch_size, height, width, channels]
        ).transpose([0, 3, 1, 2])
        hidden_states = hidden_states + residual
        hidden_states = hidden_states.squeeze(2).transpose([0, 2, 1])
        return hidden_states


class PPOCRV6SmallRecHead(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.encoder = PPOCRV6SmallRecEncoderWithSVTR(config)
        self.head = nn.Linear(config.hidden_size, config.head_out_channels)

    def forward(self, hidden_states):
        hidden_states = self.encoder(hidden_states)
        hidden_states = self.head(hidden_states)
        hidden_states = F.softmax(hidden_states, axis=2)
        return hidden_states


class PPOCRV6SmallRecModel(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.backbone = PPLCNetV4Backbone(config.backbone_config)

    def forward(self, pixel_values):
        backbone_outputs = self.backbone(pixel_values)
        hidden_state = backbone_outputs[-1]
        hidden_state = F.avg_pool2d(hidden_state, (3, 2))
        return hidden_state


class PPOCRV6SmallRec(BatchNormHFStateDictMixin, PretrainedModel):
    config_class = PPOCRV6SmallRecConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = PPOCRV6SmallRecModel(config)
        self.head = PPOCRV6SmallRecHead(config)

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
