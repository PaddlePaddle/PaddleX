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

"""SLANet / SLANet_plus table structure recognition model aligned with HF transformers.

Architecture: PP-LCNet multi-scale backbone + CSP-PAN neck + GRU-attention SLA head.
Safetensors key names match HF ``transformers.models.slanet`` exactly so weights load
directly from ``PaddlePaddle/SLANet{,_plus}_safetensors``.
"""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ...common.transformers.activations import ACT2FN
from ...common.transformers.transformers import (
    BatchNormHFStateDictMixin,
    PretrainedModel,
)
from ...image_classification.modeling._config import PPLCNetConfig
from ...image_classification.modeling.pplcnet import PPLCNetEncoder
from ._config_slanet import SLANetConfig
from .slanext import SLANeXtAttentionGRUCell, SLANeXtMLP

__all__ = ["SLANet"]


class SLANetConvLayer(nn.Layer):
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


class SLANetDepthwiseSeparableConvLayer(nn.Layer):
    def __init__(self, in_channels, out_channels, stride, kernel_size, config):
        super().__init__()
        self.depthwise_convolution = SLANetConvLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=in_channels,
            activation=config.hidden_act,
        )
        self.squeeze_excitation_module = nn.Identity()
        self.pointwise_convolution = SLANetConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            activation=config.hidden_act,
        )

    def forward(self, hidden_state):
        hidden_state = self.depthwise_convolution(hidden_state)
        hidden_state = self.squeeze_excitation_module(hidden_state)
        hidden_state = self.pointwise_convolution(hidden_state)
        return hidden_state


class SLANetBottleneck(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, activation, config):
        super().__init__()
        self.conv1 = SLANetConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            activation=activation,
        )
        self.conv2 = SLANetDepthwiseSeparableConvLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            config=config,
        )

    def forward(self, hidden_states):
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.conv2(hidden_states)
        return hidden_states


class SLANetCSPLayer(nn.Layer):
    """Cross Stage Partial (CSP) layer mirroring ``transformers.models.slanet.SLANetCSPLayer``."""

    def __init__(
        self,
        config,
        in_channels,
        out_channels,
        kernel_size=3,
        expansion=0.5,
        num_blocks=1,
        activation="hardswish",
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = SLANetConvLayer(in_channels, hidden_channels, 1, activation=activation)
        self.conv2 = SLANetConvLayer(in_channels, hidden_channels, 1, activation=activation)
        self.conv3 = SLANetConvLayer(
            2 * hidden_channels, out_channels, 1, activation=activation
        )
        self.bottlenecks = nn.LayerList(
            [
                SLANetBottleneck(
                    hidden_channels, hidden_channels, kernel_size, activation, config
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, hidden_states):
        residual = self.conv1(hidden_states)
        hidden_states = self.conv2(hidden_states)
        for bottleneck in self.bottlenecks:
            hidden_states = bottleneck(hidden_states)
        hidden_states = paddle.concat([hidden_states, residual], axis=1)
        hidden_states = self.conv3(hidden_states)
        return hidden_states


class SLANetCSPPAN(nn.Layer):
    """CSP-PAN: Path Aggregation Network with CSP layers.

    Port of ``transformers.models.slanet.SLANetCSPPAN``.
    """

    def __init__(self, config, in_channel_list):
        super().__init__()
        out_channels = config.post_conv_out_channels
        activation = config.hidden_act
        kernel_size = config.csp_kernel_size
        csp_num_blocks = config.csp_num_blocks

        self.channel_projector = nn.LayerList(
            [
                SLANetConvLayer(
                    in_channels=in_channel_list[i],
                    out_channels=out_channels,
                    kernel_size=1,
                    activation=activation,
                )
                for i in range(len(in_channel_list))
            ]
        )

        self.top_down_blocks = nn.LayerList(
            [
                SLANetCSPLayer(
                    config,
                    out_channels * 2,
                    out_channels,
                    kernel_size=kernel_size,
                    num_blocks=csp_num_blocks,
                    activation=activation,
                )
                for _ in range(len(in_channel_list) - 1, 0, -1)
            ]
        )

        self.downsamples = nn.LayerList(
            [
                SLANetDepthwiseSeparableConvLayer(
                    out_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=2,
                    config=config,
                )
                for _ in range(len(in_channel_list) - 1)
            ]
        )
        self.bottom_up_blocks = nn.LayerList(
            [
                SLANetCSPLayer(
                    config,
                    out_channels * 2,
                    out_channels,
                    kernel_size=kernel_size,
                    num_blocks=csp_num_blocks,
                    activation=activation,
                )
                for _ in range(len(in_channel_list) - 1)
            ]
        )

    def forward(self, hidden_states):
        projected_features = [
            self.channel_projector[i](hidden_states[i])
            for i in range(len(self.channel_projector))
        ]

        top_down_features = [projected_features[-1]]
        for top_down_block, low_level_feature in zip(
            self.top_down_blocks, reversed(projected_features[:-1])
        ):
            high_level_feature = top_down_features[-1]
            upsampled_feature = F.interpolate(
                high_level_feature,
                size=low_level_feature.shape[-2:],
                mode="nearest",
            )
            fused_feature = top_down_block(
                paddle.concat([upsampled_feature, low_level_feature], axis=1)
            )
            top_down_features.append(fused_feature)

        pyramid_features = list(reversed(top_down_features))
        output_feature = pyramid_features[0]
        for downsample_layer, bottom_up_block, high_level_feature in zip(
            self.downsamples, self.bottom_up_blocks, pyramid_features[1:]
        ):
            downsampled_feature = downsample_layer(output_feature)
            output_feature = bottom_up_block(
                paddle.concat([downsampled_feature, high_level_feature], axis=1)
            )

        # [B, C, H, W] -> [B, H*W, C]
        output = output_feature.flatten(2).transpose([0, 2, 1])
        return output


def _build_pplcnet_config(backbone_kwargs):
    """Translate HF-style backbone_config dict into a PPLCNetConfig.

    Drops HF-only keys (``out_features``, ``out_indices``) that the Paddle
    encoder doesn't consume — the multi-scale capture lives in
    ``SLANetVisionBackbone.forward`` instead.
    """
    kwargs = dict(backbone_kwargs)
    kwargs.pop("out_features", None)
    kwargs.pop("out_indices", None)
    kwargs.pop("model_type", None)
    return PPLCNetConfig(**kwargs)


class SLANetVisionBackbone(nn.Layer):
    """PP-LCNet backbone that returns the last 4 of 5 stage feature maps.

    State-dict keys under ``encoder.*`` reuse :class:`PPLCNetEncoder` exactly,
    so ``backbone.vision_backbone.encoder.*`` matches the HF safetensors keys.
    """

    def __init__(self, backbone_config: PPLCNetConfig):
        super().__init__()
        self.encoder = PPLCNetEncoder(backbone_config)

    def forward(self, pixel_values):
        hidden_states = self.encoder.convolution(pixel_values)
        feature_maps = []
        for idx, block in enumerate(self.encoder.blocks):
            hidden_states = block(hidden_states)
            # HF backbone_config.out_indices = [2, 3, 4, 5] maps to stages 2..5,
            # i.e. block indices 1..4 here (block 0 = stage 1 is skipped).
            if idx >= 1:
                feature_maps.append(hidden_states)
        return feature_maps


def _compute_feature_channels(backbone_config: PPLCNetConfig):
    """Output channels of the last 4 of 5 PP-LCNet stages (scale-aware)."""
    from ...image_classification.modeling.pplcnet import make_divisible

    channels = []
    for stage_blocks in backbone_config.block_configs:
        # block_configs entry: [kernel, in_ch, out_ch, stride, use_se]
        out_ch = stage_blocks[-1][2]
        channels.append(
            make_divisible(out_ch * backbone_config.scale, backbone_config.divisor)
        )
    return channels[1:]  # drop stage 1 — HF out_indices starts at stage 2


class SLANetBackbone(nn.Layer):
    def __init__(self, config: SLANetConfig):
        super().__init__()
        backbone_config = _build_pplcnet_config(config.backbone_config)
        self.vision_backbone = SLANetVisionBackbone(backbone_config)
        in_channel_list = _compute_feature_channels(backbone_config)
        self.post_csp_pan = SLANetCSPPAN(config, in_channel_list)

    def forward(self, pixel_values):
        feature_maps = self.vision_backbone(pixel_values)
        return self.post_csp_pan(feature_maps)


class SLANetSLAHead(nn.Layer):
    """Autoregressive SLA head. Architecture identical to SLANeXtSLAHead.

    Note: transformers' ``SLANetSLAHead(SLANeXtSLAHead): pass`` — same structure,
    same early-exit loop. We reuse the SLANeXt attention cell and MLP here.
    """

    def __init__(self, config: SLANetConfig):
        super().__init__()
        self.config = config
        self.structure_attention_cell = SLANeXtAttentionGRUCell(
            config.post_conv_out_channels,
            config.hidden_size,
            config.out_channels,
        )
        self.structure_generator = SLANeXtMLP(config.hidden_size, config.out_channels)

    def forward(self, hidden_states):
        batch_size = hidden_states.shape[0]
        features = paddle.zeros([batch_size, self.config.hidden_size], dtype="float32")
        predicted_chars = paddle.zeros([batch_size], dtype="int64")

        structure_preds_list = []
        structure_ids_list = []
        for _ in range(self.config.max_text_length + 1):
            embedding_feature = F.one_hot(
                predicted_chars, self.config.out_channels
            ).astype("float32")
            features, _ = self.structure_attention_cell(
                features,
                hidden_states.astype("float32"),
                embedding_feature,
            )
            structure_step = self.structure_generator(features)
            predicted_chars = structure_step.argmax(axis=1)

            structure_preds_list.append(structure_step)
            structure_ids_list.append(predicted_chars)
            if (
                paddle.stack(structure_ids_list, axis=1)
                .equal(paddle.to_tensor(self.config.out_channels - 1))
                .any(axis=-1)
                .all()
            ):
                break

        structure_preds = paddle.stack(structure_preds_list, axis=1)
        structure_probs = F.softmax(structure_preds, axis=-1)
        return structure_probs


class SLANet(BatchNormHFStateDictMixin, PretrainedModel):
    """SLANet / SLANet_plus table structure recognition model.

    Key hierarchy (matches HF safetensors):
        backbone.vision_backbone.encoder.*   → PPLCNetEncoder
        backbone.post_csp_pan.*              → SLANetCSPPAN
        head.structure_attention_cell.*      → attention GRU
        head.structure_generator.{fc1,fc2}.* → two-layer MLP
    """

    config_class = SLANetConfig

    def __init__(self, config: SLANetConfig):
        super().__init__(config)
        self.config = config
        self.backbone = SLANetBackbone(config)
        self.head = SLANetSLAHead(config)

    def forward(self, x):
        pixel_values = paddle.to_tensor(x[0])

        if pixel_values.shape[1] == 1:
            pixel_values = paddle.expand(pixel_values, [-1, 3, -1, -1])

        features = self.backbone(pixel_values)
        structure_probs = self.head(features)

        # HF SLANet has no location head. Return structure only; the
        # table-rec predictor + TableLabelDecode treat a single-element
        # prediction as "no cell polygons".
        return [structure_probs]

    def get_transpose_weight_keys(self):
        t_layers = [
            "structure_attention_cell.score",
            "structure_attention_cell.input_to_hidden",
            "structure_attention_cell.hidden_to_hidden",
            "structure_generator.fc1",
            "structure_generator.fc2",
        ]
        keys = []
        for key, _ in self.get_hf_state_dict().items():
            for t_layer in t_layers:
                if t_layer in key and key.endswith("weight"):
                    keys.append(key)
        return keys
