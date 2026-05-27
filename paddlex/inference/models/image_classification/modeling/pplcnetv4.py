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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import Constant

from ...common.transformers.activations import ACT2FN
from ._config_pplcnetv4 import PPLCNetV4Config


class PPLCNetV4LearnableAffineBlock(nn.Layer):
    def __init__(self, scale_value=1.0, bias_value=0.0):
        super().__init__()
        self.scale = self.create_parameter(
            shape=[1], default_initializer=Constant(value=scale_value)
        )
        self.bias = self.create_parameter(
            shape=[1], default_initializer=Constant(value=bias_value)
        )

    def forward(self, hidden_state):
        return self.scale * hidden_state + self.bias


class PPLCNetV4ConvLayer(nn.Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        groups=1,
        activation="relu",
        use_learnable_affine_block=False,
    ):
        super().__init__()
        self.convolution = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            padding=(kernel_size - 1) // 2,
            bias_attr=False,
        )
        self.normalization = nn.BatchNorm2D(out_channels)
        self.activation = (
            ACT2FN[activation] if activation is not None else nn.Identity()
        )
        if activation and use_learnable_affine_block:
            self.lab = PPLCNetV4LearnableAffineBlock()
        else:
            self.lab = nn.Identity()

    def forward(self, hidden_state):
        hidden_state = self.convolution(hidden_state)
        hidden_state = self.normalization(hidden_state)
        hidden_state = self.activation(hidden_state)
        hidden_state = self.lab(hidden_state)
        return hidden_state


class PPLCNetV4SqueezeExcitationModule(nn.Layer):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)

        self.convolutions = nn.LayerList()
        for in_ch, out_ch, activation in [
            [channel, channel // reduction, nn.ReLU()],
            [channel // reduction, channel, nn.Hardsigmoid()],
        ]:
            self.convolutions.append(
                nn.Conv2D(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.convolutions.append(activation)

    def forward(self, hidden_state):
        residual = hidden_state
        hidden_state = self.avg_pool(hidden_state)
        for layer in self.convolutions:
            hidden_state = layer(hidden_state)
        return residual * hidden_state


class PPLCNetV4LargeStem(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.stem1 = PPLCNetV4ConvLayer(
            config.stem_channels[0],
            config.stem_channels[1],
            kernel_size=3,
            stride=config.stem_strides[0],
            activation=config.hidden_act,
            use_learnable_affine_block=config.use_learnable_affine_block,
        )
        self.stem2a = PPLCNetV4ConvLayer(
            config.stem_channels[1],
            config.stem_channels[1] // 2,
            kernel_size=2,
            stride=config.stem_strides[1],
            activation=config.hidden_act,
            use_learnable_affine_block=config.use_learnable_affine_block,
        )
        self.stem2b = PPLCNetV4ConvLayer(
            config.stem_channels[1] // 2,
            config.stem_channels[1],
            kernel_size=2,
            stride=config.stem_strides[2],
            activation=config.hidden_act,
            use_learnable_affine_block=config.use_learnable_affine_block,
        )
        self.stem3 = PPLCNetV4ConvLayer(
            config.stem_channels[1] * 2,
            config.stem_channels[1],
            kernel_size=3,
            stride=config.stem_strides[3],
            activation=config.hidden_act,
            use_learnable_affine_block=config.use_learnable_affine_block,
        )
        self.stem4 = PPLCNetV4ConvLayer(
            config.stem_channels[1],
            config.stem_channels[2],
            kernel_size=1,
            stride=config.stem_strides[4],
            activation=config.hidden_act,
            use_learnable_affine_block=config.use_learnable_affine_block,
        )
        self.pool = nn.MaxPool2D(kernel_size=2, stride=1, ceil_mode=True)
        self.num_channels = config.num_channels

    def forward(self, pixel_values):
        embedding = self.stem1(pixel_values)
        embedding = F.pad(embedding, [0, 1, 0, 1])
        emb_stem_2a = self.stem2a(embedding)
        emb_stem_2a = F.pad(emb_stem_2a, [0, 1, 0, 1])
        emb_stem_2a = self.stem2b(emb_stem_2a)
        pooled = self.pool(embedding)
        embedding = paddle.concat([pooled, emb_stem_2a], axis=1)
        embedding = self.stem3(embedding)
        embedding = self.stem4(embedding)
        return embedding


class PPLCNetV4SmallStem(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.conv1 = PPLCNetV4ConvLayer(
            in_channels=config.stem_channels[0],
            out_channels=config.stem_channels[1],
            kernel_size=3,
            stride=2,
            activation=None,
        )
        self.act_fn = ACT2FN["gelu"]
        self.conv2 = PPLCNetV4ConvLayer(
            in_channels=config.stem_channels[1],
            out_channels=config.stem_channels[2],
            kernel_size=3,
            stride=2,
            activation=None,
        )

    def forward(self, hidden_states):
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.conv2(hidden_states)
        return hidden_states


class PPLCNetV4DepthwiseSeparableConvLayer(nn.Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        kernel_size,
        use_squeeze_excitation,
        config,
    ):
        super().__init__()
        self.has_residual = in_channels == out_channels and stride == 1
        self.use_rep_dw = stride == 1 and in_channels == out_channels

        if self.use_rep_dw:
            self.token_conv = nn.Conv2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=in_channels,
            )
        else:
            self.token_conv = PPLCNetV4ConvLayer(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                groups=in_channels,
                activation=None,
            )
        self.token_squeeze_excitation = (
            PPLCNetV4SqueezeExcitationModule(in_channels, config.reduction)
            if use_squeeze_excitation
            else nn.Identity()
        )
        self.channel_conv1 = PPLCNetV4ConvLayer(
            in_channels=in_channels,
            out_channels=in_channels * 2,
            kernel_size=1,
            stride=1,
            activation=None,
        )
        self.channel_act_fn = ACT2FN["gelu"]
        self.channel_conv2 = PPLCNetV4ConvLayer(
            in_channels=in_channels * 2,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            activation=None,
        )

    def forward(self, hidden_states):
        hidden_states = self.token_conv(hidden_states)
        hidden_states = self.token_squeeze_excitation(hidden_states)
        residual = hidden_states

        hidden_states = self.channel_conv1(hidden_states)
        hidden_states = self.channel_act_fn(hidden_states)
        hidden_states = self.channel_conv2(hidden_states)

        if self.has_residual:
            hidden_states = residual + hidden_states
        return hidden_states


class PPLCNetV4Block(nn.Layer):
    def __init__(self, config, stage_index):
        super().__init__()
        blocks = config.block_configs[stage_index]

        self.blocks = nn.LayerList()
        for kernel_size, in_channels, out_channels, stride, use_se in blocks:
            self.blocks.append(
                PPLCNetV4DepthwiseSeparableConvLayer(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    use_squeeze_excitation=use_se,
                    config=config,
                )
            )

    def forward(self, hidden_states):
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return hidden_states


class PPLCNetV4Encoder(nn.Layer):
    def __init__(self, config):
        super().__init__()
        if config.stem_type == "large":
            self.convolution = PPLCNetV4LargeStem(config)
        else:
            self.convolution = PPLCNetV4SmallStem(config)

        self.blocks = nn.LayerList(
            [PPLCNetV4Block(config, i) for i in range(len(config.block_configs))]
        )

    def forward(self, pixel_values):
        hidden_state = self.convolution(pixel_values)
        hidden_states = []
        for block in self.blocks:
            hidden_state = block(hidden_state)
            hidden_states.append(hidden_state)
        return hidden_state, hidden_states


class PPLCNetV4Backbone(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.encoder = PPLCNetV4Encoder(config)
        self.stage_out_channels = list(config.stage_out_channels)

    def forward(self, pixel_values):
        _, hidden_states = self.encoder(pixel_values)
        return hidden_states
