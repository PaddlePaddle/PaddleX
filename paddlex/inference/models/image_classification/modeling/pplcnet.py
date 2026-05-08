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

from typing import List, Optional

import paddle
import paddle.nn as nn

from ...common.transformers.activations import ACT2FN
from ...common.transformers.transformers import (
    BatchNormHFStateDictMixin,
    PretrainedModel,
)
from ._config import PPLCNetConfig


def make_divisible(v: float, divisor: int = 8, min_value: Optional[int] = None) -> int:
    """
    Ensure the number of channels is a multiple of the specified divisor (common optimization for mobile networks)

    Args:
        v: Original number of channels
        divisor: Divisor, default 8
        min_value: Minimum number of channels, default None (takes divisor)

    Returns:
        Adjusted number of channels (integer)
    """

    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class PPLCNetSqueezeExcitationModule(nn.Layer):
    """
    Squeeze-and-Excitation (SE) Module: Adaptive feature recalibration
    Enhances the model's ability to focus on important channels by learning channel-wise attention weights.
    """

    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)

        self.convolutions = nn.LayerList()
        for in_channels, out_channels, activation in [
            [channel, channel // reduction, nn.ReLU()],
            [channel // reduction, channel, nn.Hardsigmoid()],
        ]:
            self.convolutions.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                )
            )
            self.convolutions.append(activation)

    def forward(self, hidden_state):
        residual = hidden_state
        hidden_state = self.avg_pool(hidden_state)
        for layer in self.convolutions:
            hidden_state = layer(hidden_state)
        hidden_state = residual * hidden_state

        return hidden_state


class PPLCNetConvLayer(nn.Layer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        activation: str = "hardswish",
        groups: int = 1,
    ):
        super().__init__()
        self.convolution = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False,
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


class PPLCNetDepthwiseSeparableConvLayer(nn.Layer):
    """
    Depthwise Separable Convolution Layer: Depthwise Conv -> SE Module (optional) -> Pointwise Conv
    Core component of lightweight models (e.g., MobileNet, PP-LCNet) that significantly reduces
    the number of parameters and computational cost.
    """

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
        self.depthwise_convolution = PPLCNetConvLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=in_channels,
            activation=config.hidden_act,
        )
        self.squeeze_excitation_module = (
            PPLCNetSqueezeExcitationModule(in_channels, config.reduction)
            if use_squeeze_excitation
            else nn.Identity()
        )
        self.pointwise_convolution = PPLCNetConvLayer(
            in_channels=in_channels,
            kernel_size=1,
            out_channels=out_channels,
            stride=1,
            activation=config.hidden_act,
        )

    def forward(self, hidden_state):
        hidden_state = self.depthwise_convolution(hidden_state)
        hidden_state = self.squeeze_excitation_module(hidden_state)
        hidden_state = self.pointwise_convolution(hidden_state)

        return hidden_state


class PPLCNetBlock(nn.Layer):
    def __init__(self, config, stage_index):
        super().__init__()
        self.config = config
        blocks = config.block_configs[stage_index]

        self.layers = nn.LayerList()
        for (
            kernel_size,
            in_channels,
            out_channels,
            stride,
            use_squeeze_excitation,
        ) in blocks:
            scaled_in_channels = make_divisible(
                in_channels * config.scale, config.divisor
            )
            scaled_out_channels = make_divisible(
                out_channels * config.scale, config.divisor
            )

            depthwise_block = PPLCNetDepthwiseSeparableConvLayer(
                in_channels=scaled_in_channels,
                out_channels=scaled_out_channels,
                kernel_size=kernel_size,
                stride=stride,
                use_squeeze_excitation=use_squeeze_excitation,
                config=config,
            )
            self.layers.append(depthwise_block)

    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class PPLCNetEncoder(PretrainedModel):
    def __init__(self, config: PPLCNetConfig) -> None:
        super().__init__(config)

        # stem
        self.convolution = PPLCNetConvLayer(
            in_channels=3,
            kernel_size=3,
            out_channels=make_divisible(
                config.stem_channels * config.scale, config.divisor
            ),
            stride=config.stem_stride,
            activation=config.hidden_act,
        )
        # stages
        self.blocks = nn.LayerList()
        for stage_index in range(len(config.block_configs)):
            block = PPLCNetBlock(config, stage_index)
            self.blocks.append(block)

    def forward(self, pixel_values):
        hidden_states = self.convolution(pixel_values)
        for block in self.blocks:
            hidden_states = block(hidden_states)

        return hidden_states


class PPLCNet(BatchNormHFStateDictMixin, PretrainedModel):
    config_class = PPLCNetConfig

    def __init__(self, config: PPLCNetConfig) -> None:
        super().__init__(config)

        self.encoder = PPLCNetEncoder(config)
        self.config = config
        self.num_labels = config.num_labels
        last_block_out_channels = config.block_configs[-1][-1][2]
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.last_convolution = nn.Conv2d(
            in_channels=make_divisible(
                last_block_out_channels * config.scale, config.divisor
            ),
            out_channels=config.class_expand,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.act_fn = ACT2FN[config.hidden_act]
        self.hidden_dropout_prob = config.hidden_dropout_prob

        self.flatten = nn.Flatten(start_axis=1, stop_axis=-1)
        self.head = (
            nn.Linear(config.class_expand, config.num_labels)
            if config.num_labels > 0
            else nn.Identity()
        )

    def forward(self, x: List) -> List:
        pixel_values = paddle.to_tensor(x[0])

        outputs = self.encoder(pixel_values)

        last_hidden_state = self.avg_pool(outputs)

        last_hidden_state = self.last_convolution(last_hidden_state)
        last_hidden_state = self.act_fn(last_hidden_state)
        last_hidden_state = last_hidden_state * (1 - self.hidden_dropout_prob)

        last_hidden_state = self.flatten(last_hidden_state)
        last_hidden_state = self.head(last_hidden_state)

        # align postprocessing in ClasTransformersPredictor.process
        last_hidden_state = last_hidden_state.softmax()

        return [last_hidden_state.cpu().numpy()]

    def get_transpose_weight_keys(self):
        t_layers = ["head"]
        keys = []
        for key, _ in self.get_hf_state_dict().items():
            for t_layer in t_layers:
                if t_layer in key and key.endswith("weight"):
                    keys.append(key)
        return keys
