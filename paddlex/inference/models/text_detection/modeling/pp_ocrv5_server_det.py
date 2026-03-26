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
from ...image_classification.modeling.hgnetv2 import HGNetV2Backbone
from ._config_pp_ocrv5_server import PPOCRV5ServerDetConfig


class PPOCRV5ServerDetConvBatchnormLayer(nn.Layer):
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


class PPOCRV5ServerDetIntraclassBlock(nn.Layer):
    def __init__(self, intraclass_block_config, in_channels, reduce_factor):
        super().__init__()
        reduced_channels = in_channels // reduce_factor

        self.conv_reduce_channel = nn.Conv2D(
            in_channels,
            reduced_channels,
            *intraclass_block_config["reduce_channel"],
        )

        self.vertical_long_to_small_conv_longratio = nn.Conv2D(
            reduced_channels,
            reduced_channels,
            *intraclass_block_config["vertical_long_to_small_conv_longratio"],
        )
        self.vertical_long_to_small_conv_midratio = nn.Conv2D(
            reduced_channels,
            reduced_channels,
            *intraclass_block_config["vertical_long_to_small_conv_midratio"],
        )
        self.vertical_long_to_small_conv_shortratio = nn.Conv2D(
            reduced_channels,
            reduced_channels,
            *intraclass_block_config["vertical_long_to_small_conv_shortratio"],
        )

        self.horizontal_small_to_long_conv_longratio = nn.Conv2D(
            reduced_channels,
            reduced_channels,
            *intraclass_block_config["horizontal_small_to_long_conv_longratio"],
        )
        self.horizontal_small_to_long_conv_midratio = nn.Conv2D(
            reduced_channels,
            reduced_channels,
            *intraclass_block_config["horizontal_small_to_long_conv_midratio"],
        )
        self.horizontal_small_to_long_conv_shortratio = nn.Conv2D(
            reduced_channels,
            reduced_channels,
            *intraclass_block_config["horizontal_small_to_long_conv_shortratio"],
        )

        self.symmetric_conv_long_longratio = nn.Conv2D(
            reduced_channels,
            reduced_channels,
            *intraclass_block_config["symmetric_conv_long_longratio"],
        )
        self.symmetric_conv_long_midratio = nn.Conv2D(
            reduced_channels,
            reduced_channels,
            *intraclass_block_config["symmetric_conv_long_midratio"],
        )
        self.symmetric_conv_long_shortratio = nn.Conv2D(
            reduced_channels,
            reduced_channels,
            *intraclass_block_config["symmetric_conv_long_shortratio"],
        )

        self.conv_final = PPOCRV5ServerDetConvBatchnormLayer(
            in_channels=reduced_channels,
            out_channels=in_channels,
            kernel_size=intraclass_block_config["return_channel"][0],
            stride=intraclass_block_config["return_channel"][1],
            padding=intraclass_block_config["return_channel"][2],
            bias=True,
        )

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.conv_reduce_channel(hidden_states)

        hidden_states = (
            self.symmetric_conv_long_longratio(hidden_states)
            + self.vertical_long_to_small_conv_longratio(hidden_states)
            + self.horizontal_small_to_long_conv_longratio(hidden_states)
        )
        hidden_states = (
            self.symmetric_conv_long_midratio(hidden_states)
            + self.vertical_long_to_small_conv_midratio(hidden_states)
            + self.horizontal_small_to_long_conv_midratio(hidden_states)
        )
        hidden_states = (
            self.symmetric_conv_long_shortratio(hidden_states)
            + self.vertical_long_to_small_conv_shortratio(hidden_states)
            + self.horizontal_small_to_long_conv_shortratio(hidden_states)
        )

        hidden_states = self.conv_final(hidden_states)
        return residual + hidden_states


class PPOCRV5ServerDetNeck(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.interpolate_mode = config.interpolate_mode
        self.scale_factor_list = config.scale_factor_list
        self.num_backbone_stages = len(config.backbone_config.stage_out_channels)

        backbone_stage_output_channels = config.backbone_config.stage_out_channels

        self.input_channel_adjustment_convolution = nn.LayerList()
        self.input_feature_projection_convolution = nn.LayerList()
        self.path_aggregation_head_convolution = nn.LayerList()
        self.path_aggregation_lateral_convolution = nn.LayerList()

        for i in range(len(backbone_stage_output_channels)):
            self.input_channel_adjustment_convolution.append(
                nn.Conv2D(
                    in_channels=backbone_stage_output_channels[i],
                    out_channels=config.neck_out_channels,
                    kernel_size=1,
                    bias_attr=False,
                )
            )
            self.input_feature_projection_convolution.append(
                nn.Conv2D(
                    in_channels=config.neck_out_channels,
                    out_channels=config.neck_out_channels // 4,
                    kernel_size=9,
                    padding=4,
                    bias_attr=False,
                )
            )
            if i > 0:
                self.path_aggregation_head_convolution.append(
                    nn.Conv2D(
                        in_channels=config.neck_out_channels // 4,
                        out_channels=config.neck_out_channels // 4,
                        kernel_size=3,
                        padding=1,
                        stride=2,
                        bias_attr=False,
                    )
                )
            self.path_aggregation_lateral_convolution.append(
                nn.Conv2D(
                    in_channels=config.neck_out_channels // 4,
                    out_channels=config.neck_out_channels // 4,
                    kernel_size=9,
                    padding=4,
                    bias_attr=False,
                )
            )

        self.intraclass_blocks = nn.LayerList()
        for _ in range(config.intraclass_block_number):
            self.intraclass_blocks.append(
                PPOCRV5ServerDetIntraclassBlock(
                    config.intraclass_block_config,
                    config.neck_out_channels // 4,
                    reduce_factor=config.reduce_factor,
                )
            )

    def forward(self, backbone_stage_feature_maps):
        channel_adjusted = []
        for i, feature_map in enumerate(backbone_stage_feature_maps):
            hidden_states = self.input_channel_adjustment_convolution[i](feature_map)
            channel_adjusted.append(hidden_states)

        top_down = [None] * self.num_backbone_stages
        top_down[self.num_backbone_stages - 1] = channel_adjusted[
            self.num_backbone_stages - 1
        ]
        for i in range(self.num_backbone_stages - 2, -1, -1):
            top_down[i] = channel_adjusted[i] + F.interpolate(
                top_down[i + 1], scale_factor=2, mode=self.interpolate_mode
            )

        projected = []
        for i in range(self.num_backbone_stages):
            hidden_states = (
                top_down[i]
                if i < self.num_backbone_stages - 1
                else channel_adjusted[-1]
            )
            hidden_states = self.input_feature_projection_convolution[i](hidden_states)
            projected.append(hidden_states)

        bottom_up = [None] * self.num_backbone_stages
        bottom_up[0] = projected[0]
        for i in range(1, self.num_backbone_stages):
            bottom_up[i] = projected[i] + self.path_aggregation_head_convolution[i - 1](
                bottom_up[i - 1]
            )

        lateral_refined = []
        for i in range(self.num_backbone_stages):
            hidden_states = projected[0] if i == 0 else bottom_up[i]
            hidden_states = self.path_aggregation_lateral_convolution[i](hidden_states)
            lateral_refined.append(hidden_states)

        intraclass_refined = [
            block(feature)
            for block, feature in zip(self.intraclass_blocks, lateral_refined)
        ]

        upsampled = [
            (
                F.interpolate(
                    feature, scale_factor=scale_factor, mode=self.interpolate_mode
                )
                if scale_factor > 1
                else feature
            )
            for feature, scale_factor in zip(intraclass_refined, self.scale_factor_list)
        ]

        return paddle.concat(upsampled[::-1], axis=1)


class PPOCRV5ServerDetSegmentationHead(nn.Layer):
    def __init__(self, config):
        super().__init__()
        in_channels = config.neck_out_channels
        kernel_list = config.kernel_list

        self.conv_down = PPOCRV5ServerDetConvBatchnormLayer(
            in_channels=in_channels,
            out_channels=in_channels // 4,
            kernel_size=kernel_list[0],
            padding=int(kernel_list[0] // 2),
        )
        self.conv_up = PPOCRV5ServerDetConvBatchnormLayer(
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
        feature = hidden_states
        hidden_states = self.conv_final(hidden_states)
        hidden_states = F.sigmoid(hidden_states)
        return hidden_states, feature


class PPOCRV5ServerDetLocalModule(nn.Layer):
    def __init__(self, in_channels, out_channels, hidden_act):
        super().__init__()
        self.convolution_backbone = PPOCRV5ServerDetConvBatchnormLayer(
            in_channels=in_channels + 1,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            activation=hidden_act,
        )
        self.convolution_final = nn.Conv2D(
            in_channels=out_channels,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, hidden_states, init_map):
        hidden_states = paddle.concat([init_map, hidden_states], axis=1)
        hidden_states = self.convolution_backbone(hidden_states)
        hidden_states = self.convolution_final(hidden_states)
        return hidden_states


class PPOCRV5ServerDetHead(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.binarize_head = PPOCRV5ServerDetSegmentationHead(config)
        self.upsample_convolution = nn.Upsample(
            scale_factor=config.scale_factor, mode=config.interpolate_mode
        )
        self.local_refinement_module = PPOCRV5ServerDetLocalModule(
            config.neck_out_channels // 4,
            config.neck_out_channels // 4,
            config.hidden_act,
        )

    def forward(self, hidden_states):
        hidden_states, feature = self.binarize_head(hidden_states)
        residual = hidden_states
        feature = self.upsample_convolution(feature)
        hidden_states = self.local_refinement_module(feature, hidden_states)
        hidden_states = F.sigmoid(hidden_states)
        return 0.5 * (residual + hidden_states)


class PPOCRV5ServerDetModel(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.backbone = HGNetV2Backbone(config.backbone_config)
        self.neck = PPOCRV5ServerDetNeck(config)

    def forward(self, pixel_values):
        backbone_outputs = self.backbone(pixel_values)
        return self.neck(backbone_outputs)


class PPOCRV5ServerDet(BatchNormHFStateDictMixin, PretrainedModel):
    config_class = PPOCRV5ServerDetConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = PPOCRV5ServerDetModel(config)
        self.head = PPOCRV5ServerDetHead(config)

    def forward(self, x: List) -> List:
        x = paddle.to_tensor(x[0])
        neck_output = self.model(x)
        output = self.head(neck_output)
        return [output.cpu().numpy()]
