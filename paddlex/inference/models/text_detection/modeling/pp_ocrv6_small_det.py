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
from ._config_pp_ocrv6_small_det import PPOCRV6SmallDetConfig


class PPOCRV6SmallDetConvBatchnormLayer(nn.Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=1,
        groups=1,
        activation="relu",
        bias=False,
        convolution_transpose=False,
    ):
        super().__init__()
        if convolution_transpose:
            self.convolution = nn.Conv2DTranspose(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
            )
        else:
            self.convolution = nn.Conv2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias_attr=None if bias else False,
            )
        self.norm = nn.BatchNorm2D(out_channels)
        self.act_fn = nn.Identity() if activation is None else ACT2FN[activation]

    def forward(self, hidden_states):
        hidden_states = self.convolution(hidden_states)
        hidden_states = self.norm(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        return hidden_states


class PPOCRV6SmallDetSqueezeExcitationModule(nn.Layer):
    def __init__(self, in_channels, reduction, activation="relu"):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(output_size=1)
        self.conv1 = nn.Conv2D(
            in_channels=in_channels,
            out_channels=in_channels // reduction,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.conv2 = nn.Conv2D(
            in_channels=in_channels // reduction,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.act_fn = ACT2FN[activation]

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.avg_pool(hidden_states)
        hidden_states = self.conv2(self.act_fn(self.conv1(hidden_states)))
        hidden_states = paddle.clip(0.2 * hidden_states + 0.5, min=0.0, max=1.0)
        return residual * hidden_states


class PPOCRV6SmallDetResidualSqueezeExcitationLayer(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, reduction):
        super().__init__()
        self.in_conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=int(kernel_size // 2),
            bias_attr=False,
        )
        self.squeeze_excitation_block = PPOCRV6SmallDetSqueezeExcitationModule(
            out_channels, reduction
        )

    def forward(self, hidden_states):
        hidden_states = self.in_conv(hidden_states)
        hidden_states = hidden_states + self.squeeze_excitation_block(hidden_states)
        return hidden_states


class PPOCRV6SmallDetDepthwiseSeparableConvLayer(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, reduction):
        super().__init__()
        self.depthwise_convolution = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=in_channels,
            bias_attr=None,
        )
        self.squeeze_excitation_module = PPOCRV6SmallDetSqueezeExcitationModule(
            out_channels // 4, reduction
        )
        self.pointwise_convolution = nn.Conv2D(
            in_channels=out_channels,
            out_channels=out_channels // 4,
            kernel_size=1,
            bias_attr=False,
        )

    def forward(self, hidden_states):
        hidden_states = self.depthwise_convolution(hidden_states)
        hidden_states = self.pointwise_convolution(hidden_states)
        hidden_states = hidden_states + self.squeeze_excitation_module(hidden_states)
        return hidden_states


class PPOCRV6SmallDetNeck(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.interpolate_mode = config.interpolate_mode

        self.insert_conv = nn.LayerList()
        self.input_conv = nn.LayerList()
        for i in range(len(config.layer_list_out_channels)):
            self.insert_conv.append(
                PPOCRV6SmallDetResidualSqueezeExcitationLayer(
                    config.layer_list_out_channels[i],
                    config.neck_out_channels,
                    1,
                    config.reduction,
                )
            )
            self.input_conv.append(
                PPOCRV6SmallDetDepthwiseSeparableConvLayer(
                    config.neck_out_channels,
                    config.neck_out_channels,
                    config.dilated_kernel_size,
                    config.reduction,
                )
            )

    def forward(self, feature_maps):
        fused = []
        for conv, feature in zip(self.insert_conv, feature_maps):
            hidden_states = conv(feature)
            fused.append(hidden_states)

        for i in range(2, -1, -1):
            fused[i] = fused[i] + F.interpolate(
                fused[i + 1], scale_factor=2, mode=self.interpolate_mode
            )

        features = []
        for conv, feat in zip(
            self.input_conv, [fused[0], fused[1], fused[2], fused[3]]
        ):
            features.append(conv(feat))

        processed = []
        upsample_scales = [1, 2, 4, 8]
        for feat, scale in zip(features, upsample_scales):
            if scale != 1:
                hidden_states = F.interpolate(
                    feat, scale_factor=scale, mode=self.interpolate_mode
                )
            else:
                hidden_states = feat
            processed.append(hidden_states)

        return paddle.concat(processed[::-1], axis=1)


class PPOCRV6SmallDetHead(nn.Layer):
    def __init__(self, config):
        super().__init__()
        in_channels = config.neck_out_channels
        kernel_list = config.kernel_list

        self.conv_down = PPOCRV6SmallDetConvBatchnormLayer(
            in_channels=in_channels,
            out_channels=in_channels // 4,
            kernel_size=kernel_list[0],
            padding=int(kernel_list[0] // 2),
        )
        self.conv_up = PPOCRV6SmallDetConvBatchnormLayer(
            in_channels=in_channels // 4,
            out_channels=in_channels // 4,
            kernel_size=kernel_list[1],
            stride=2,
            convolution_transpose=True,
        )
        self.conv_final = nn.Conv2DTranspose(
            in_channels=in_channels // 4,
            out_channels=1,
            kernel_size=kernel_list[2],
            stride=2,
        )

    def forward(self, hidden_states):
        hidden_states = self.conv_down(hidden_states)
        hidden_states = self.conv_up(hidden_states)
        hidden_states = self.conv_final(hidden_states)
        hidden_states = F.sigmoid(hidden_states)
        return hidden_states


class PPOCRV6SmallDetModel(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.backbone = PPLCNetV4Backbone(config.backbone_config)
        self.neck = PPOCRV6SmallDetNeck(config)

    def forward(self, pixel_values):
        backbone_outputs = self.backbone(pixel_values)
        return self.neck(backbone_outputs)


class PPOCRV6SmallDet(BatchNormHFStateDictMixin, PretrainedModel):
    config_class = PPOCRV6SmallDetConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = PPOCRV6SmallDetModel(config)
        self.head = PPOCRV6SmallDetHead(config)

    def forward(self, x: List) -> List:
        x = paddle.to_tensor(x[0])
        neck_output = self.model(x)
        output = self.head(neck_output)
        return [output.cpu().numpy()]
