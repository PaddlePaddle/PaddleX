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

from typing import Any, List

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ...common.transformers.activations import ACT2FN
from ...common.transformers.transformers import (
    BatchNormHFStateDictMixin,
    PretrainedModel,
)
from ._config import UVDocConfig


class UVDocConvLayer(nn.Layer):
    """Convolutional layer with batch normalization and activation."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        padding_mode="zeros",
        bias=False,
        dilation=1,
        activation="relu",
    ):
        super().__init__()
        self.convolution = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
            dilation=dilation,
            bias_attr=False if not bias else None,
        )
        self.normalization = nn.BatchNorm2D(out_channels)
        self.activation = (
            ACT2FN[activation] if activation is not None else nn.Identity()
        )

    def forward(self, hidden_state):
        hidden_state = self.convolution(hidden_state)
        hidden_state = self.normalization(hidden_state)
        hidden_state = self.activation(hidden_state)
        return hidden_state


class UVDocResidualBlock(nn.Layer):
    """Residual block with dilation support."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        downsample=False,
        activation="relu",
    ):
        super().__init__()
        self.conv_down = (
            UVDocConvLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                bias=True,
                activation=None,
            )
            if downsample
            else nn.Identity()
        )
        self.conv_start = UVDocConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        )
        self.conv_final = UVDocConvLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=True,
            dilation=dilation,
            activation=None,
        )
        self.act_fn = ACT2FN[activation] if activation is not None else nn.Identity()

    def forward(self, hidden_states):
        residual = self.conv_down(hidden_states)
        hidden_states = self.conv_start(hidden_states)
        hidden_states = self.conv_final(hidden_states)
        hidden_states = hidden_states + residual
        hidden_states = self.act_fn(hidden_states)
        return hidden_states


class UVDocResNetStage(nn.Layer):
    """A ResNet stage containing multiple residual blocks."""

    def __init__(self, config, stage_index):
        super().__init__()
        stages = config.resnet_configs[stage_index]
        self.layers = nn.LayerList()
        for in_channels, out_channels, dilation, downsample in stages:
            self.layers.append(
                UVDocResidualBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=2 if downsample else 1,
                    padding=dilation * 2,
                    dilation=dilation,
                    downsample=downsample,
                    kernel_size=config.kernel_size,
                )
            )

    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class UVDocResNet(nn.Layer):
    """Initial resnet_head and resnet_down stages."""

    def __init__(self, config):
        super().__init__()
        self.resnet_head = nn.LayerList()
        for i in range(len(config.resnet_head)):
            self.resnet_head.append(
                UVDocConvLayer(
                    in_channels=config.resnet_head[i][0],
                    out_channels=config.resnet_head[i][1],
                    kernel_size=config.kernel_size,
                    stride=2,
                    padding=config.kernel_size // 2,
                )
            )

        self.resnet_down = nn.LayerList()
        for stage_index in range(len(config.resnet_configs)):
            self.resnet_down.append(UVDocResNetStage(config, stage_index))

    def forward(self, hidden_states):
        for head in self.resnet_head:
            hidden_states = head(hidden_states)
        for stage in self.resnet_down:
            hidden_states = stage(hidden_states)
        return hidden_states


class UVDocBridgeBlock(nn.Layer):
    """Bridge block with dilated convolutions for long-range dependencies."""

    def __init__(self, config, bridge_index):
        super().__init__()
        self.blocks = nn.LayerList()
        bridge = config.stage_configs[bridge_index]
        for in_channels, dilation in bridge:
            self.blocks.append(
                UVDocConvLayer(
                    in_channels, in_channels, padding=dilation, dilation=dilation
                )
            )

    def forward(self, hidden_states):
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return hidden_states


class UVDocBridge(nn.Layer):
    """Bridge module containing multiple bridge blocks."""

    def __init__(self, config):
        super().__init__()
        self.bridge = nn.LayerList()
        for bridge_index in range(len(config.stage_configs)):
            self.bridge.append(UVDocBridgeBlock(config, bridge_index))

    def forward(self, hidden_states):
        feature_maps = []
        for layer in self.bridge:
            feature_maps.append(layer(hidden_states))
        return feature_maps


class UVDocPointPositions2D(nn.Layer):
    """Module for predicting 2D point positions for document rectification."""

    def __init__(self, config):
        super().__init__()
        self.conv_down = UVDocConvLayer(
            in_channels=config.out_point_positions2D[0][0],
            out_channels=config.out_point_positions2D[0][1],
            kernel_size=config.kernel_size,
            stride=1,
            padding=config.kernel_size // 2,
            padding_mode=config.padding_mode,
            activation=config.hidden_act,
        )
        self.conv_up = nn.Conv2D(
            in_channels=config.out_point_positions2D[1][0],
            out_channels=config.out_point_positions2D[1][1],
            kernel_size=config.kernel_size,
            stride=1,
            padding=config.kernel_size // 2,
            padding_mode=config.padding_mode,
        )

    def forward(self, hidden_states):
        hidden_states = self.conv_down(hidden_states)
        hidden_states = self.conv_up(hidden_states)
        return hidden_states


class UVDocBackbone(nn.Layer):
    """UVDoc backbone with ResNet and bridge modules for feature extraction."""

    def __init__(self, backbone_config):
        super().__init__()
        self.resnet = UVDocResNet(backbone_config)
        self.bridge = UVDocBridge(backbone_config)

    def forward(self, pixel_values):
        hidden_states = self.resnet(pixel_values)
        feature_maps = self.bridge(hidden_states)
        return feature_maps


class UVDocHead(nn.Layer):
    """UVDoc output head with bridge connector and point position prediction."""

    def __init__(self, config):
        super().__init__()
        num_bridge_layers = len(config.backbone_config.stage_configs)

        self.bridge_connector = UVDocConvLayer(
            in_channels=config.bridge_connector[0] * num_bridge_layers,
            out_channels=config.bridge_connector[1],
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
        )
        self.out_point_positions2D = UVDocPointPositions2D(config)

    def forward(self, hidden_states):
        hidden_states = self.bridge_connector(hidden_states)
        hidden_states = self.out_point_positions2D(hidden_states)
        return hidden_states


class UVDocNet(BatchNormHFStateDictMixin, PretrainedModel):
    """UVDoc model for document image rectification."""

    config_class = UVDocConfig
    _keys_to_ignore_on_load_unexpected = ["num_batches_tracked"]

    def __init__(self, config: UVDocConfig):
        super().__init__(config)
        self.backbone = UVDocBackbone(config.backbone_config)
        self.head = UVDocHead(config)
        self.upsample_size = config.upsample_size
        self.upsample_mode = config.upsample_mode

    def forward(self, x: Any) -> List:
        x = paddle.to_tensor(x[0])

        image = x
        h_ori, w_ori = x.shape[2:]
        x = F.interpolate(
            x,
            size=self.upsample_size,
            mode=self.upsample_mode,
            align_corners=True,
        )

        feature_maps = self.backbone(x)
        fused = paddle.concat(feature_maps, axis=1)
        out = self.head(fused)

        bm_up = F.interpolate(
            out,
            size=(h_ori, w_ori),
            mode=self.upsample_mode,
            align_corners=True,
        )
        bm = bm_up.transpose([0, 2, 3, 1])
        result = F.grid_sample(image, bm, align_corners=True)

        return [result.cpu().numpy()]

    def _get_forward_key_rules(self):
        default_rules = super()._get_forward_key_rules()
        custom_rules = [
            (
                "head.out_point_positions2D.conv_down.activation._weight",
                "_weight",
                "weight",
            )
        ]
        return default_rules + custom_rules

    def _get_reverse_key_rules(self):
        default_rules = super()._get_reverse_key_rules()
        custom_rules = [
            (
                "head.out_point_positions2D.conv_down.activation.weight",
                "weight",
                "_weight",
            )
        ]
        return default_rules + custom_rules
