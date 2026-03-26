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
from paddle.nn.initializer import Constant

from ...common.transformers.activations import ACT2FN
from ._config_pplcnetv3 import PPLCNetV3Config


def make_divisible(value, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)


class PPLCNetV3ConvLayer(nn.Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        activation="hardswish",
        groups=1,
    ):
        super().__init__()
        self.convolution = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias_attr=False,
            groups=groups,
        )
        self.normalization = nn.BatchNorm2D(out_channels)
        self.activation = (
            ACT2FN[activation] if activation is not None else nn.Identity()
        )

    def forward(self, input):
        hidden_state = self.convolution(input)
        hidden_state = self.normalization(hidden_state)
        hidden_state = self.activation(hidden_state)
        return hidden_state


class PPLCNetV3LearnableAffineBlock(nn.Layer):
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


class PPLCNetV3ActLearnableAffineBlock(nn.Layer):
    def __init__(self, activation="hardswish"):
        super().__init__()
        self.act = ACT2FN[activation]
        self.lab = PPLCNetV3LearnableAffineBlock()

    def forward(self, hidden_state):
        return self.lab(self.act(hidden_state))


class PPLCNetV3LearnableRepLayer(nn.Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        activation,
        stride,
        num_conv_branches,
        groups=1,
    ):
        super().__init__()
        self.stride = stride

        self.identity = (
            nn.BatchNorm2D(in_channels)
            if out_channels == in_channels and stride == 1
            else None
        )

        self.conv_symmetric = nn.LayerList(
            [
                PPLCNetV3ConvLayer(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    groups=groups,
                    activation=None,
                )
                for _ in range(num_conv_branches)
            ]
        )

        self.conv_small_symmetric = (
            PPLCNetV3ConvLayer(
                in_channels, out_channels, 1, stride, groups=groups, activation=None
            )
            if kernel_size > 1
            else None
        )

        self.lab = PPLCNetV3LearnableAffineBlock()
        self.act = PPLCNetV3ActLearnableAffineBlock(activation=activation)

    def forward(self, hidden_state):
        output = None

        if self.identity is not None:
            output = self.identity(hidden_state)

        if self.conv_small_symmetric is not None:
            residual = self.conv_small_symmetric(hidden_state)
            output = residual if output is None else output + residual

        for conv in self.conv_symmetric:
            residual = conv(hidden_state)
            output = residual if output is None else output + residual

        hidden_state = self.lab(output)
        if self.stride != 2:
            hidden_state = self.act(hidden_state)
        return hidden_state


class PPLCNetV3SqueezeExcitationModule(nn.Layer):
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


class PPLCNetV3DepthwiseSeparableConvLayer(nn.Layer):
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
        self.depthwise_convolution = PPLCNetV3LearnableRepLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=in_channels,
            num_conv_branches=config.conv_symmetric_num,
            activation=config.hidden_act,
        )
        self.squeeze_excitation_module = (
            PPLCNetV3SqueezeExcitationModule(in_channels, config.reduction)
            if use_squeeze_excitation
            else nn.Identity()
        )
        self.pointwise_convolution = PPLCNetV3LearnableRepLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            num_conv_branches=config.conv_symmetric_num,
            activation=config.hidden_act,
        )

    def forward(self, hidden_state):
        hidden_state = self.depthwise_convolution(hidden_state)
        hidden_state = self.squeeze_excitation_module(hidden_state)
        hidden_state = self.pointwise_convolution(hidden_state)
        return hidden_state


class PPLCNetV3Block(nn.Layer):
    def __init__(self, config, stage_index):
        super().__init__()
        blocks = config.block_configs[stage_index]

        self.layers = nn.LayerList()
        for kernel_size, in_channels, out_channels, stride, use_se in blocks:
            scaled_in = make_divisible(in_channels * config.scale, config.divisor)
            scaled_out = make_divisible(out_channels * config.scale, config.divisor)
            self.layers.append(
                PPLCNetV3DepthwiseSeparableConvLayer(
                    in_channels=scaled_in,
                    out_channels=scaled_out,
                    kernel_size=kernel_size,
                    stride=stride,
                    use_squeeze_excitation=use_se,
                    config=config,
                )
            )

    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class PPLCNetV3Encoder(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.convolution = PPLCNetV3ConvLayer(
            in_channels=3,
            out_channels=make_divisible(
                config.stem_channels * config.scale, config.divisor
            ),
            kernel_size=3,
            stride=config.stem_stride,
            activation=None,
        )
        self.blocks = nn.LayerList(
            [
                PPLCNetV3Block(config, i)
                for i in range(len(config.block_configs))
            ]
        )

    def forward(self, pixel_values):
        hidden_state = self.convolution(pixel_values)
        hidden_states = []
        for block in self.blocks:
            hidden_state = block(hidden_state)
            hidden_states.append(hidden_state)
        return hidden_state, hidden_states


class PPLCNetV3Backbone(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.encoder = PPLCNetV3Encoder(config)
        self.out_channels = [
            make_divisible(block_cfg[-1][2] * config.scale, config.divisor)
            for block_cfg in config.block_configs
        ]

    def forward(self, pixel_values):
        _, hidden_states = self.encoder(pixel_values)
        return hidden_states
