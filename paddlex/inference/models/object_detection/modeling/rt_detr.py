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

"""RT-DETR model with HF-compatible parameter naming.

Architecture mirrors HuggingFace transformers' RTDetrForObjectDetection so that
both ``paddle_dynamic`` and ``transformers`` backends can load the same
HF-format safetensors without heavy runtime conversion.

Key naming conventions (matching HF transformers):
  - model.backbone.model.*          (HGNetV2 backbone)
  - model.encoder_input_proj.*      (encoder input projections)
  - model.encoder.aifi.*            (AIFI transformer encoder layers)
  - model.encoder.{lateral,fpn,downsample,pan}_*  (hybrid FPN/PAN)
  - model.decoder.layers.*          (decoder transformer layers)
  - model.decoder.class_embed.*     (per-layer class prediction)
  - model.decoder.bbox_embed.*      (per-layer bbox prediction)

A lightweight ``set_hf_state_dict`` handles minor key differences between the
safetensors on-disk format and the model's parameter names (see
``_apply_rt_detr_key_conversion`` in pp_doclayout_v2.py).
"""

from __future__ import absolute_import, division, print_function

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ...common.transformers.activations import ACT2CLS, ACT2FN
from ...common.transformers.transformers import (
    BatchNormHFStateDictMixin,
    PretrainedModel,
)
from ...image_classification.modeling.hgnetv2 import HGNetV2Backbone
from ._config_rt_detr import RTDETRConfig
from .pp_doclayout_v2 import (
    _apply_rt_detr_key_conversion,
    _reverse_rt_detr_key_conversion,
)

__all__ = ["RTDETR"]


def bbox_cxcywh_to_xyxy(x):
    cxcy, wh = paddle.split(x, 2, axis=-1)
    return paddle.concat([cxcy - 0.5 * wh, cxcy + 0.5 * wh], axis=-1)


def inverse_sigmoid(x, eps=1e-5):
    x = x.clip(min=0, max=1)
    x1 = x.clip(min=eps)
    x2 = (1 - x).clip(min=eps)
    return paddle.log(x1 / x2)


class RTDETRFrozenBatchNorm2d(nn.Layer):
    """BatchNorm2d where the batch statistics and the affine parameters are fixed."""

    def __init__(self, n):
        super().__init__()
        self.register_buffer("weight", paddle.ones([n]))
        self.register_buffer("bias", paddle.zeros([n]))
        self.register_buffer("_mean", paddle.zeros([n]))
        self.register_buffer("_variance", paddle.ones([n]))

    def forward(self, x):
        weight = self.weight.reshape([1, -1, 1, 1])
        bias = self.bias.reshape([1, -1, 1, 1])
        running_var = self._variance.reshape([1, -1, 1, 1])
        running_mean = self._mean.reshape([1, -1, 1, 1])
        epsilon = 1e-5
        scale = weight * (running_var + epsilon).rsqrt()
        bias = bias - running_mean * scale
        return x * scale + bias


class RTDETRMLPPredictionHead(nn.Layer):
    """Simple multi-layer perceptron for bbox/class prediction."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.LayerList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class RTDETTMLP(nn.Layer):
    """Feed-forward MLP used in encoder and decoder layers."""

    def __init__(self, config, hidden_size, intermediate_size, activation_function):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.activation_fn = ACT2FN[activation_function]
        self.activation_dropout = config.activation_dropout
        self.dropout = config.dropout

    def forward(self, hidden_states):
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(
            hidden_states, p=self.activation_dropout, training=self.training
        )
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        return hidden_states


class RTDETRSelfAttention(nn.Layer):
    """Multi-headed self-attention. Position embeddings added to queries and keys."""

    def __init__(
        self, config, hidden_size, num_attention_heads, dropout=0.0, bias=True
    ):
        super().__init__()
        self.head_dim = hidden_size // num_attention_heads
        self.num_heads = num_attention_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = dropout

        self.k_proj = nn.Linear(hidden_size, hidden_size, bias_attr=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias_attr=bias)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias_attr=bias)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias_attr=bias)

    def forward(self, hidden_states, attention_mask=None, position_embeddings=None):
        batch_size, seq_len, _ = hidden_states.shape

        query_key_input = (
            hidden_states + position_embeddings
            if position_embeddings is not None
            else hidden_states
        )

        query_states = (
            self.q_proj(query_key_input)
            .reshape([batch_size, seq_len, self.num_heads, self.head_dim])
            .transpose([0, 2, 1, 3])
        )
        key_states = (
            self.k_proj(query_key_input)
            .reshape([batch_size, seq_len, self.num_heads, self.head_dim])
            .transpose([0, 2, 1, 3])
        )
        value_states = (
            self.v_proj(hidden_states)
            .reshape([batch_size, seq_len, self.num_heads, self.head_dim])
            .transpose([0, 2, 1, 3])
        )

        attn_weights = (
            paddle.matmul(query_states, key_states.transpose([0, 1, 3, 2]))
            * self.scaling
        )

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, axis=-1)
        attn_weights = F.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )

        attn_output = paddle.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose([0, 2, 1, 3]).reshape(
            [batch_size, seq_len, -1]
        )
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class RTDETRConvNormLayer(nn.Layer):
    """Conv layer with conv/norm attribute names."""

    def __init__(
        self,
        config,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding=None,
        activation=None,
    ):
        super().__init__()
        self.conv = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2 if padding is None else padding,
            bias_attr=False,
        )
        self.norm = nn.BatchNorm2D(out_channels, epsilon=config.batch_norm_eps)
        self.activation = nn.Identity() if activation is None else ACT2CLS[activation]()

    def forward(self, hidden_state):
        hidden_state = self.conv(hidden_state)
        hidden_state = self.norm(hidden_state)
        hidden_state = self.activation(hidden_state)
        return hidden_state


class RTDETREncoderLayer(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.normalize_before = config.normalize_before
        self.hidden_size = config.encoder_hidden_dim

        self.self_attn = RTDETRSelfAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_attention_heads=config.encoder_attention_heads,
            dropout=config.dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(
            self.hidden_size, epsilon=config.layer_norm_eps
        )
        self.dropout = config.dropout
        self.mlp = RTDETTMLP(
            config,
            self.hidden_size,
            config.encoder_ffn_dim,
            config.encoder_activation_function,
        )
        self.final_layer_norm = nn.LayerNorm(
            self.hidden_size, epsilon=config.layer_norm_eps
        )

    def forward(self, hidden_states, attention_mask, spatial_position_embeddings=None):
        residual = hidden_states
        if self.normalize_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=spatial_position_embeddings,
        )

        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        if not self.normalize_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        if self.normalize_before:
            hidden_states = self.final_layer_norm(hidden_states)
        residual = hidden_states

        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states
        if not self.normalize_before:
            hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states


class RTDETRRepVggBlock(nn.Layer):
    """RepVGG architecture block."""

    def __init__(self, config):
        super().__init__()
        activation = config.activation_function
        hidden_channels = int(config.encoder_hidden_dim * config.hidden_expansion)
        self.conv1 = RTDETRConvNormLayer(
            config, hidden_channels, hidden_channels, 3, 1, padding=1
        )
        self.conv2 = RTDETRConvNormLayer(
            config, hidden_channels, hidden_channels, 1, 1, padding=0
        )
        self.activation = nn.Identity() if activation is None else ACT2CLS[activation]()

    def forward(self, x):
        y = self.conv1(x) + self.conv2(x)
        return self.activation(y)


class RTDETRCSPRepLayer(nn.Layer):
    """Cross Stage Partial (CSP) network layer with RepVGG blocks."""

    def __init__(self, config):
        super().__init__()
        in_channels = config.encoder_hidden_dim * 2
        out_channels = config.encoder_hidden_dim
        num_blocks = 3
        activation = config.activation_function

        hidden_channels = int(out_channels * config.hidden_expansion)
        self.conv1 = RTDETRConvNormLayer(
            config, in_channels, hidden_channels, 1, 1, activation=activation
        )
        self.conv2 = RTDETRConvNormLayer(
            config, in_channels, hidden_channels, 1, 1, activation=activation
        )
        self.bottlenecks = nn.Sequential(
            *[RTDETRRepVggBlock(config) for _ in range(num_blocks)]
        )
        if hidden_channels != out_channels:
            self.conv3 = RTDETRConvNormLayer(
                config, hidden_channels, out_channels, 1, 1, activation=activation
            )
        else:
            self.conv3 = nn.Identity()

    def forward(self, hidden_state):
        hidden_state_1 = self.conv1(hidden_state)
        hidden_state_1 = self.bottlenecks(hidden_state_1)
        hidden_state_2 = self.conv2(hidden_state)
        return self.conv3(hidden_state_1 + hidden_state_2)


class RTDETRSinePositionEmbedding(nn.Layer):
    """2D sinusoidal position embedding."""

    def __init__(self, embed_dim=256, temperature=10000):
        super().__init__()
        self.embed_dim = embed_dim
        self.temperature = temperature

    def forward(self, width, height, dtype):
        grid_w = paddle.arange(width).astype(dtype)
        grid_h = paddle.arange(height).astype(dtype)
        grid_h, grid_w = paddle.meshgrid(grid_h, grid_w)

        if self.embed_dim % 4 != 0:
            raise ValueError(
                "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
            )
        pos_dim = self.embed_dim // 4
        omega = paddle.arange(pos_dim).astype(dtype) / pos_dim
        omega = 1.0 / (self.temperature**omega)

        out_w = grid_w.flatten().unsqueeze(-1) @ omega.unsqueeze(0)
        out_h = grid_h.flatten().unsqueeze(-1) @ omega.unsqueeze(0)

        return paddle.concat(
            [out_h.sin(), out_h.cos(), out_w.sin(), out_w.cos()], axis=1
        ).unsqueeze(0)


class RTDETRAIFILayer(nn.Layer):
    """AIFI (Attention-based Intra-scale Feature Interaction) layer."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder_hidden_dim = config.encoder_hidden_dim
        self.eval_size = config.eval_size

        self.position_embedding = RTDETRSinePositionEmbedding(
            embed_dim=self.encoder_hidden_dim,
            temperature=config.positional_encoding_temperature,
        )
        self.layers = nn.LayerList(
            [RTDETREncoderLayer(config) for _ in range(config.encoder_layers)]
        )

    def forward(self, hidden_states):
        batch_size = hidden_states.shape[0]
        height, width = hidden_states.shape[2:]

        hidden_states = hidden_states.flatten(2).transpose([0, 2, 1])

        if self.training or self.eval_size is None:
            pos_embed = self.position_embedding(
                width=width,
                height=height,
                dtype=hidden_states.dtype,
            )
        else:
            pos_embed = None

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=None,
                spatial_position_embeddings=pos_embed,
            )

        hidden_states = hidden_states.transpose([0, 2, 1]).reshape(
            [batch_size, self.encoder_hidden_dim, height, width]
        )

        return hidden_states


class RTDETRHybridEncoder(nn.Layer):
    """Hybrid encoder: AIFI layers + top-down FPN + bottom-up PAN."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.in_channels = config.encoder_in_channels
        self.feat_strides = config.feat_strides
        self.encoder_hidden_dim = config.encoder_hidden_dim
        self.encode_proj_layers = config.encode_proj_layers
        self.eval_size = config.eval_size
        self.num_fpn_stages = len(self.in_channels) - 1
        self.num_pan_stages = len(self.in_channels) - 1

        # AIFI layers
        self.aifi = nn.LayerList(
            [RTDETRAIFILayer(config) for _ in range(len(self.encode_proj_layers))]
        )

        # top-down FPN
        self.lateral_convs = nn.LayerList()
        self.fpn_blocks = nn.LayerList()
        for _ in range(self.num_fpn_stages):
            lateral_conv = RTDETRConvNormLayer(
                config,
                in_channels=self.encoder_hidden_dim,
                out_channels=self.encoder_hidden_dim,
                kernel_size=1,
                stride=1,
                activation=config.activation_function,
            )
            fpn_block = RTDETRCSPRepLayer(config)
            self.lateral_convs.append(lateral_conv)
            self.fpn_blocks.append(fpn_block)

        # bottom-up PAN
        self.downsample_convs = nn.LayerList()
        self.pan_blocks = nn.LayerList()
        for _ in range(self.num_pan_stages):
            downsample_conv = RTDETRConvNormLayer(
                config,
                in_channels=self.encoder_hidden_dim,
                out_channels=self.encoder_hidden_dim,
                kernel_size=3,
                stride=2,
                activation=config.activation_function,
            )
            pan_block = RTDETRCSPRepLayer(config)
            self.downsample_convs.append(downsample_conv)
            self.pan_blocks.append(pan_block)

    def forward(self, feature_maps):
        # AIFI: Apply transformer encoder to specified feature levels
        if self.config.encoder_layers > 0:
            for i, enc_ind in enumerate(self.encode_proj_layers):
                feature_maps[enc_ind] = self.aifi[i](feature_maps[enc_ind])

        # top-down FPN
        fpn_feature_maps = [feature_maps[-1]]
        for idx, (lateral_conv, fpn_block) in enumerate(
            zip(self.lateral_convs, self.fpn_blocks)
        ):
            backbone_feature_map = feature_maps[self.num_fpn_stages - idx - 1]
            top_fpn_feature_map = fpn_feature_maps[-1]
            top_fpn_feature_map = lateral_conv(top_fpn_feature_map)
            fpn_feature_maps[-1] = top_fpn_feature_map
            top_fpn_feature_map = F.interpolate(
                top_fpn_feature_map, scale_factor=2.0, mode="nearest"
            )
            fused_feature_map = paddle.concat(
                [top_fpn_feature_map, backbone_feature_map], axis=1
            )
            new_fpn_feature_map = fpn_block(fused_feature_map)
            fpn_feature_maps.append(new_fpn_feature_map)

        fpn_feature_maps.reverse()

        # bottom-up PAN
        pan_feature_maps = [fpn_feature_maps[0]]
        for idx, (downsample_conv, pan_block) in enumerate(
            zip(self.downsample_convs, self.pan_blocks)
        ):
            top_pan_feature_map = pan_feature_maps[-1]
            fpn_feature_map = fpn_feature_maps[idx + 1]
            downsampled_feature_map = downsample_conv(top_pan_feature_map)
            fused_feature_map = paddle.concat(
                [downsampled_feature_map, fpn_feature_map], axis=1
            )
            new_pan_feature_map = pan_block(fused_feature_map)
            pan_feature_maps.append(new_pan_feature_map)

        return pan_feature_maps


class MultiScaleDeformableAttention(nn.Layer):
    """Eager fallback implementation using grid_sample."""

    def forward(
        self,
        value,
        value_spatial_shapes,
        value_spatial_shapes_list,
        level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step,
    ):
        batch_size, _, num_heads, hidden_dim = value.shape
        _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
        value_list = paddle.split(
            value,
            [height * width for height, width in value_spatial_shapes_list],
            axis=1,
        )
        sampling_grids = 2 * sampling_locations - 1
        sampling_value_list = []
        for level_id, (height, width) in enumerate(value_spatial_shapes_list):
            value_l_ = (
                value_list[level_id]
                .flatten(2)
                .transpose([0, 2, 1])
                .reshape([batch_size * num_heads, hidden_dim, height, width])
            )
            sampling_grid_l_ = (
                sampling_grids[:, :, :, level_id]
                .transpose([0, 2, 1, 3])
                .reshape([batch_size * num_heads, num_queries, num_points, 2])
            )
            sampling_value_l_ = F.grid_sample(
                value_l_,
                sampling_grid_l_,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            sampling_value_list.append(sampling_value_l_)

        attention_weights = attention_weights.transpose([0, 2, 1, 3, 4]).reshape(
            [batch_size * num_heads, 1, num_queries, num_levels * num_points]
        )
        output = (
            (paddle.stack(sampling_value_list, axis=-2).flatten(-2) * attention_weights)
            .sum(-1)
            .reshape([batch_size, num_heads * hidden_dim, num_queries])
        )
        return output.transpose([0, 2, 1])


class RTDETRMultiscaleDeformableAttention(nn.Layer):
    """Multiscale deformable attention as proposed in Deformable DETR."""

    def __init__(self, config, num_heads, n_points):
        super().__init__()
        self.attn = MultiScaleDeformableAttention()

        self.d_model = config.d_model
        self.n_levels = config.num_feature_levels
        self.n_heads = num_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(
            config.d_model, num_heads * self.n_levels * n_points * 2
        )
        self.attention_weights = nn.Linear(
            config.d_model, num_heads * self.n_levels * n_points
        )
        self.value_proj = nn.Linear(config.d_model, config.d_model)
        self.output_proj = nn.Linear(config.d_model, config.d_model)

        self.im2col_step = 64

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        position_embeddings=None,
        reference_points=None,
        spatial_shapes=None,
        spatial_shapes_list=None,
        level_start_index=None,
    ):
        if position_embeddings is not None:
            hidden_states = hidden_states + position_embeddings

        batch_size, num_queries, _ = hidden_states.shape
        batch_size, sequence_length, _ = encoder_hidden_states.shape

        value = self.value_proj(encoder_hidden_states)
        if attention_mask is not None:
            value = paddle.where(
                attention_mask.unsqueeze(-1).astype("bool"),
                value,
                paddle.zeros_like(value),
            )
        value = value.reshape(
            [batch_size, sequence_length, self.n_heads, self.d_model // self.n_heads]
        )
        sampling_offsets = self.sampling_offsets(hidden_states).reshape(
            [batch_size, num_queries, self.n_heads, self.n_levels, self.n_points, 2]
        )
        attn_weights = self.attention_weights(hidden_states).reshape(
            [batch_size, num_queries, self.n_heads, self.n_levels * self.n_points]
        )
        attn_weights = F.softmax(attn_weights, axis=-1).reshape(
            [batch_size, num_queries, self.n_heads, self.n_levels, self.n_points]
        )

        num_coordinates = reference_points.shape[-1]
        if num_coordinates == 2:
            offset_normalizer = paddle.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], axis=-1
            )
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif num_coordinates == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets
                / self.n_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            raise ValueError(
                f"Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]}"
            )

        output = self.attn(
            value,
            spatial_shapes,
            spatial_shapes_list,
            level_start_index,
            sampling_locations,
            attn_weights,
            self.im2col_step,
        )

        output = self.output_proj(output)
        return output, attn_weights


class RTDETRDecoderLayer(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.d_model

        self.self_attn = RTDETRSelfAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_attention_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.dropout = config.dropout

        self.self_attn_layer_norm = nn.LayerNorm(
            self.hidden_size, epsilon=config.layer_norm_eps
        )
        self.encoder_attn = RTDETRMultiscaleDeformableAttention(
            config,
            num_heads=config.decoder_attention_heads,
            n_points=config.decoder_n_points,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(
            self.hidden_size, epsilon=config.layer_norm_eps
        )
        self.mlp = RTDETTMLP(
            config,
            self.hidden_size,
            config.decoder_ffn_dim,
            config.decoder_activation_function,
        )
        self.final_layer_norm = nn.LayerNorm(
            self.hidden_size, epsilon=config.layer_norm_eps
        )

    def forward(
        self,
        hidden_states,
        object_queries_position_embeddings=None,
        reference_points=None,
        spatial_shapes=None,
        spatial_shapes_list=None,
        level_start_index=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        residual = hidden_states

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=encoder_attention_mask,
            position_embeddings=object_queries_position_embeddings,
        )

        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states

        # Cross-Attention
        hidden_states, _ = self.encoder_attn(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            position_embeddings=object_queries_position_embeddings,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            level_start_index=level_start_index,
        )

        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states


class RTDETRDecoder(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.dropout = config.dropout
        self.layers = nn.LayerList(
            [RTDETRDecoderLayer(config) for _ in range(config.decoder_layers)]
        )
        self.query_pos_head = RTDETRMLPPredictionHead(
            4, 2 * config.d_model, config.d_model, num_layers=2
        )

        # Per-layer prediction heads (with_box_refine=True)
        self.class_embed = nn.LayerList(
            [
                nn.Linear(config.d_model, config.num_labels)
                for _ in range(config.decoder_layers)
            ]
        )
        self.bbox_embed = nn.LayerList(
            [
                RTDETRMLPPredictionHead(config.d_model, config.d_model, 4, num_layers=3)
                for _ in range(config.decoder_layers)
            ]
        )

        self.num_queries = config.num_queries

    def forward(
        self,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        reference_points=None,
        spatial_shapes=None,
        spatial_shapes_list=None,
        level_start_index=None,
    ):
        hidden_states = inputs_embeds

        intermediate_reference_points = []
        intermediate_logits = []

        reference_points = F.sigmoid(reference_points)

        for idx, decoder_layer in enumerate(self.layers):
            reference_points_input = reference_points.unsqueeze(2)
            object_queries_position_embeddings = self.query_pos_head(reference_points)

            hidden_states = decoder_layer(
                hidden_states,
                object_queries_position_embeddings=object_queries_position_embeddings,
                encoder_hidden_states=encoder_hidden_states,
                reference_points=reference_points_input,
                spatial_shapes=spatial_shapes,
                spatial_shapes_list=spatial_shapes_list,
                level_start_index=level_start_index,
                encoder_attention_mask=encoder_attention_mask,
            )

            # Per-layer bbox refinement
            predicted_corners = self.bbox_embed[idx](hidden_states)
            new_reference_points = F.sigmoid(
                predicted_corners + inverse_sigmoid(reference_points)
            )
            reference_points = new_reference_points.detach()

            intermediate_reference_points.append(new_reference_points)
            intermediate_logits.append(self.class_embed[idx](hidden_states))

        intermediate_reference_points = paddle.stack(
            intermediate_reference_points, axis=1
        )
        intermediate_logits = paddle.stack(intermediate_logits, axis=1)

        return {
            "intermediate_logits": intermediate_logits,
            "intermediate_reference_points": intermediate_reference_points,
        }


def replace_batch_norm(model):
    """Recursively replace all nn.BatchNorm2D with RTDETRFrozenBatchNorm2d."""
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2D):
            new_module = RTDETRFrozenBatchNorm2d(module._num_features)
            new_module.weight.set_value(module.weight)
            new_module.bias.set_value(module.bias)
            new_module._mean.set_value(module._mean)
            new_module._variance.set_value(module._variance)
            setattr(model, name, new_module)

        if len(list(module.children())) > 0:
            replace_batch_norm(module)


class RTDETRConvEncoder(nn.Layer):
    """Convolutional backbone using HGNetV2Backbone."""

    def __init__(self, config):
        super().__init__()
        backbone = HGNetV2Backbone(config.backbone_config)

        if config.freeze_backbone_batch_norms:
            with paddle.no_grad():
                replace_batch_norm(backbone)
        self.model = backbone
        self.intermediate_channel_sizes = backbone.out_channels

    def forward(self, pixel_values):
        features = self.model(pixel_values)
        return features


class RTDETRModel(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Create backbone
        self.backbone = RTDETRConvEncoder(config)

        # Create encoder input projection layers (channels from config, not backbone)
        encoder_input_proj_list = []
        for in_channels in config.encoder_in_channels:
            encoder_input_proj_list.append(
                nn.Sequential(
                    nn.Conv2D(
                        in_channels,
                        config.encoder_hidden_dim,
                        kernel_size=1,
                        bias_attr=False,
                    ),
                    nn.BatchNorm2D(config.encoder_hidden_dim),
                )
            )
        self.encoder_input_proj = nn.LayerList(encoder_input_proj_list)

        # Create encoder
        self.encoder = RTDETRHybridEncoder(config)

        # Denoising embedding (HF convention: num_labels+1 with padding_idx)
        self.denoising_class_embed = nn.Embedding(
            config.num_labels + 1, config.d_model, padding_idx=config.num_labels
        )

        # Decoder embedding
        if config.learn_initial_query:
            self.weight_embedding = nn.Embedding(config.num_queries, config.d_model)

        # Encoder head
        self.enc_output = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model, epsilon=config.layer_norm_eps),
        )
        self.enc_score_head = nn.Linear(config.d_model, config.num_labels)
        self.enc_bbox_head = RTDETRMLPPredictionHead(
            config.d_model, config.d_model, 4, num_layers=3
        )

        # Create decoder input projection layers
        num_backbone_outs = len(config.decoder_in_channels)
        decoder_input_proj_list = []
        for i in range(num_backbone_outs):
            in_channels = config.decoder_in_channels[i]
            decoder_input_proj_list.append(
                nn.Sequential(
                    nn.Conv2D(
                        in_channels, config.d_model, kernel_size=1, bias_attr=False
                    ),
                    nn.BatchNorm2D(config.d_model, epsilon=config.batch_norm_eps),
                )
            )
        for _ in range(config.num_feature_levels - num_backbone_outs):
            decoder_input_proj_list.append(
                nn.Sequential(
                    nn.Conv2D(
                        in_channels,
                        config.d_model,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias_attr=False,
                    ),
                    nn.BatchNorm2D(config.d_model, epsilon=config.batch_norm_eps),
                )
            )
            in_channels = config.d_model
        self.decoder_input_proj = nn.LayerList(decoder_input_proj_list)

        # Decoder
        self.decoder = RTDETRDecoder(config)

    def generate_anchors(self, spatial_shapes, grid_size=0.05, dtype="float32"):
        anchors = []
        for level, (height, width) in enumerate(spatial_shapes):
            grid_y, grid_x = paddle.meshgrid(
                paddle.arange(end=height).astype(dtype),
                paddle.arange(end=width).astype(dtype),
            )
            grid_xy = paddle.stack([grid_x, grid_y], axis=-1)
            grid_xy = grid_xy.unsqueeze(0) + 0.5
            grid_xy[..., 0] /= width
            grid_xy[..., 1] /= height
            wh = paddle.ones_like(grid_xy) * grid_size * (2.0**level)
            anchors.append(
                paddle.concat([grid_xy, wh], axis=-1).reshape([-1, height * width, 4])
            )
        eps = 1e-2
        anchors = paddle.concat(anchors, axis=1)
        valid_mask = ((anchors > eps) & (anchors < 1 - eps)).all(axis=-1, keepdim=True)
        anchors = paddle.log(anchors / (1 - anchors))
        anchors = paddle.where(
            valid_mask, anchors, paddle.full_like(anchors, float("inf"))
        )
        return anchors, valid_mask

    def forward(self, pixel_values, pixel_mask=None):
        batch_size = pixel_values.shape[0]

        # Backbone — may return more stages than encoder needs; take last N
        features = self.backbone(pixel_values)
        num_proj = len(self.encoder_input_proj)
        features = features[-num_proj:]
        proj_feats = [
            self.encoder_input_proj[level](source)
            for level, source in enumerate(features)
        ]

        # Encoder (hybrid FPN/PAN)
        encoder_outputs = self.encoder(proj_feats)

        # Prepare decoder inputs
        sources = []
        for level, source in enumerate(encoder_outputs):
            sources.append(self.decoder_input_proj[level](source))

        if self.config.num_feature_levels > len(sources):
            _len_sources = len(sources)
            sources.append(self.decoder_input_proj[_len_sources](encoder_outputs[-1]))
            for i in range(_len_sources + 1, self.config.num_feature_levels):
                sources.append(self.decoder_input_proj[i](encoder_outputs[-1]))

        # Flatten sources for decoder
        source_flatten = []
        spatial_shapes_list = []
        spatial_shapes = paddle.empty([len(sources), 2], dtype="int64")
        for level, source in enumerate(sources):
            h, w = source.shape[-2:]
            spatial_shapes[level, 0] = h
            spatial_shapes[level, 1] = w
            spatial_shapes_list.append((h, w))
            source = source.flatten(2).transpose([0, 2, 1])
            source_flatten.append(source)
        source_flatten = paddle.concat(source_flatten, axis=1)
        level_start_index = paddle.concat(
            [paddle.zeros([1], dtype="int64"), spatial_shapes.prod(1).cumsum(0)[:-1]]
        )

        dtype = source_flatten.dtype

        spatial_shapes_tuple = tuple(spatial_shapes_list)
        anchors, valid_mask = self.generate_anchors(spatial_shapes_tuple, dtype=dtype)

        memory = valid_mask.astype(source_flatten.dtype) * source_flatten

        output_memory = self.enc_output(memory)

        enc_outputs_class = self.enc_score_head(output_memory)
        enc_outputs_coord_logits = self.enc_bbox_head(output_memory) + anchors

        _, topk_ind = paddle.topk(
            enc_outputs_class.max(-1), self.config.num_queries, axis=1
        )

        reference_points_unact = paddle.take_along_axis(
            enc_outputs_coord_logits,
            topk_ind.unsqueeze(-1).expand([-1, -1, enc_outputs_coord_logits.shape[-1]]),
            axis=1,
        )

        enc_topk_bboxes = F.sigmoid(reference_points_unact)

        if self.config.learn_initial_query:
            target = self.weight_embedding.weight.unsqueeze(0).expand(
                [batch_size, -1, -1]
            )
        else:
            target = paddle.take_along_axis(
                output_memory,
                topk_ind.unsqueeze(-1).expand([-1, -1, output_memory.shape[-1]]),
                axis=1,
            )
            target = target.detach()

        init_reference_points = reference_points_unact.detach()

        # Decoder
        decoder_outputs = self.decoder(
            inputs_embeds=target,
            encoder_hidden_states=source_flatten,
            encoder_attention_mask=None,
            reference_points=init_reference_points,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            level_start_index=level_start_index,
        )

        return decoder_outputs


class DETRPostProcess(object):
    __shared__ = ["num_classes", "use_focal_loss"]
    __inject__ = []

    def __init__(
        self,
        num_classes=80,
        num_top_queries=100,
        use_focal_loss=False,
        bbox_decode_type="origin",
    ):
        super(DETRPostProcess, self).__init__()
        assert bbox_decode_type in ["origin", "pad"]
        self.num_classes = num_classes
        self.num_top_queries = num_top_queries
        self.use_focal_loss = use_focal_loss
        self.bbox_decode_type = bbox_decode_type

    def __call__(self, head_out, im_shape, scale_factor, pad_shape):
        bboxes, logits = head_out

        bbox_pred = bbox_cxcywh_to_xyxy(bboxes)

        origin_shape = paddle.floor(im_shape / scale_factor + 0.5)
        img_h, img_w = paddle.split(origin_shape, 2, axis=-1)
        if self.bbox_decode_type == "pad":
            out_shape = pad_shape / im_shape * origin_shape
            out_shape = out_shape.flip(1).tile([1, 2]).unsqueeze(1)
        elif self.bbox_decode_type == "origin":
            out_shape = origin_shape.flip(1).tile([1, 2]).unsqueeze(1)
        else:
            raise Exception(f"Wrong `bbox_decode_type`: {self.bbox_decode_type}.")
        bbox_pred *= out_shape

        scores = (
            F.sigmoid(logits) if self.use_focal_loss else F.softmax(logits)[:, :, :-1]
        )

        if not self.use_focal_loss:
            scores, labels = scores.max(-1), scores.argmax(-1)
            if scores.shape[1] > self.num_top_queries:
                scores, index = paddle.topk(scores, self.num_top_queries, axis=-1)
                batch_ind = (
                    paddle.arange(end=scores.shape[0])
                    .unsqueeze(-1)
                    .tile([1, self.num_top_queries])
                )
                index = paddle.stack([batch_ind, index], axis=-1)
                labels = paddle.gather_nd(labels, index)
                bbox_pred = paddle.gather_nd(bbox_pred, index)
        else:
            scores, index = paddle.topk(
                scores.flatten(1), self.num_top_queries, axis=-1
            )
            labels = index % self.num_classes
            index = index // self.num_classes
            batch_ind = (
                paddle.arange(end=scores.shape[0])
                .unsqueeze(-1)
                .tile([1, self.num_top_queries])
            )
            index = paddle.stack([batch_ind, index], axis=-1)
            bbox_pred = paddle.gather_nd(bbox_pred, index)

        bbox_pred = paddle.concat(
            [labels.unsqueeze(-1).astype("float32"), scores.unsqueeze(-1), bbox_pred],
            axis=-1,
        )
        bbox_num = paddle.to_tensor(self.num_top_queries, dtype="int32").tile(
            [bbox_pred.shape[0]]
        )
        bbox_pred = bbox_pred.reshape([-1, 6])
        return bbox_pred, bbox_num


class RTDETR(BatchNormHFStateDictMixin, PretrainedModel):

    config_class = RTDETRConfig
    _keys_to_ignore_on_load_unexpected = ["num_batches_tracked"]

    def __init__(self, config):
        super(RTDETR, self).__init__(config)
        self.config = config

        self.model = RTDETRModel(config)

        self.post_process = DETRPostProcess(
            num_classes=config.num_labels,
            num_top_queries=config.num_queries,
            use_focal_loss=config.use_focal_loss,
        )

    def forward(self, inputs):
        pixel_values = paddle.to_tensor(inputs[1])
        im_shape = paddle.to_tensor(inputs[0])
        scale_factor = paddle.to_tensor(inputs[2])

        decoder_outputs = self.model(pixel_values)

        intermediate_reference_points = decoder_outputs["intermediate_reference_points"]
        intermediate_logits = decoder_outputs["intermediate_logits"]

        # Take last layer outputs
        pred_boxes = intermediate_reference_points[:, -1]
        logits = intermediate_logits[:, -1]

        head_out = (pred_boxes, logits)
        pad_shape = paddle.to_tensor(
            [[pixel_values.shape[2], pixel_values.shape[3]]] * pixel_values.shape[0],
            dtype="float32",
        )
        bbox, bbox_num = self.post_process(
            head_out,
            im_shape,
            scale_factor,
            pad_shape,
        )

        output = [bbox, bbox_num]
        return output

    def get_transpose_weight_keys(self):
        t_layers = [
            "fc",
            "o_proj",
            "out_proj",
            "output_proj",
            "q_proj",
            "k_proj",
            "v_proj",
            "enc_bbox_head",
            "enc_output",
            "query_pos_head",
            "enc_score_head",
            "encoder_attn",
            "decoder.bbox_embed",
            "decoder.class_embed",
            "sampling_offsets",
            "attention_weights",
            "value_proj",
        ]
        keys = []
        for key, _ in self.get_hf_state_dict().items():
            for t_layer in t_layers:
                if (
                    t_layer in key
                    and key.endswith("weight")
                    and "norm" not in key
                    and "LayerNorm" not in key
                    and "layer_norm" not in key
                    and "enc_output.1" not in key
                ):
                    keys.append(key)
        return keys

    def set_hf_state_dict(self, state_dict, *args, **kwargs):
        converted = _apply_rt_detr_key_conversion(state_dict)
        return super().set_hf_state_dict(converted, *args, **kwargs)

    def get_hf_state_dict(self, *args, **kwargs):
        return _reverse_rt_detr_key_conversion(
            super().get_hf_state_dict(*args, **kwargs)
        )
