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

from __future__ import absolute_import, division, print_function

import math

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ...common.transformers.activations import ACT2FN, ACT2CLS
from ...common.transformers.transformers import (
    BatchNormHFStateDictMixin,
    PretrainedModel,
)
from ...image_classification.modeling.hgnetv2 import HGNetV2Backbone
from ._config_pp_doclayout_v3 import PPDocLayoutV3Config
from .pp_doclayout_v2 import _apply_rt_detr_key_conversion, _reverse_rt_detr_key_conversion


def bbox_cxcywh_to_xyxy(x):
    cxcy, wh = paddle.split(x, 2, axis=-1)
    return paddle.concat([cxcy - 0.5 * wh, cxcy + 0.5 * wh], axis=-1)


__all__ = ["PPDocLayoutV3"]


def inverse_sigmoid(x, eps=1e-5):
    x = x.clip(min=0, max=1)
    x1 = x.clip(min=eps)
    x2 = (1 - x).clip(min=eps)
    return paddle.log(x1 / x2)


def get_order(order_logits):
    order_scores = paddle.nn.functional.sigmoid(order_logits)
    B, N, _ = order_scores.shape
    one = paddle.ones([N, N], dtype=order_scores.dtype)
    upper = paddle.triu(one, 1)
    lower = paddle.tril(one, -1)
    Q = order_scores * upper + (1.0 - paddle.transpose(order_scores, [0, 2, 1])) * lower
    order_votes = paddle.sum(Q, axis=1)
    order_pointers = paddle.argsort(order_votes, axis=1)
    order_seq = paddle.full(order_pointers.shape, -1, dtype=order_pointers.dtype)
    batch_indices = paddle.arange(B).reshape([-1, 1]).expand([B, N])
    order_seq[batch_indices, order_pointers] = paddle.arange(N).expand([B, N])
    return order_seq, order_votes


def mask_to_box_coordinate(mask, dtype):
    """Convert binary masks to normalized bounding box coordinates (cx, cy, w, h)."""
    mask = mask.astype("bool")
    mask_float = mask.astype(dtype)
    height, width = mask.shape[-2:]

    y_coords, x_coords = paddle.meshgrid(
        paddle.arange(height), paddle.arange(width)
    )
    x_coords = x_coords.astype(dtype)
    y_coords = y_coords.astype(dtype)

    finfo_max = paddle.to_tensor(np.finfo(np.float32).max, dtype=dtype)

    x_coords_masked = x_coords * mask_float
    x_max = x_coords_masked.flatten(start_axis=-2).max(axis=-1) + 1
    x_min = (
        paddle.where(mask, x_coords_masked, finfo_max.expand(x_coords_masked.shape))
        .flatten(start_axis=-2)
        .min(axis=-1)
    )

    y_coords_masked = y_coords * mask_float
    y_max = y_coords_masked.flatten(start_axis=-2).max(axis=-1) + 1
    y_min = (
        paddle.where(mask, y_coords_masked, finfo_max.expand(y_coords_masked.shape))
        .flatten(start_axis=-2)
        .min(axis=-1)
    )

    unnormalized_bbox = paddle.stack([x_min, y_min, x_max, y_max], axis=-1)
    is_mask_non_empty = paddle.any(mask, axis=[-2, -1]).unsqueeze(-1).astype(dtype)
    unnormalized_bbox = unnormalized_bbox * is_mask_non_empty

    norm_tensor = paddle.to_tensor([width, height, width, height], dtype=dtype)
    normalized_bbox_xyxy = unnormalized_bbox / norm_tensor

    x_min_norm, y_min_norm, x_max_norm, y_max_norm = paddle.unbind(
        normalized_bbox_xyxy, axis=-1
    )

    center_x = (x_min_norm + x_max_norm) / 2
    center_y = (y_min_norm + y_max_norm) / 2
    box_width = x_max_norm - x_min_norm
    box_height = y_max_norm - y_min_norm

    return paddle.stack([center_x, center_y, box_width, box_height], axis=-1)


class PPDocLayoutV3FrozenBatchNorm2d(nn.Layer):
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


class PPDocLayoutV3GlobalPointer(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.head_size = config.global_pointer_head_size
        self.dense = nn.Linear(config.d_model, self.head_size * 2)
        self.dropout = nn.Dropout(config.gp_dropout_value)

    def forward(self, inputs):
        batch_size, sequence_length, _ = inputs.shape
        query_key_projection = self.dense(inputs).reshape(
            [batch_size, sequence_length, 2, self.head_size]
        )
        query_key_projection = self.dropout(query_key_projection)
        queries, keys = paddle.unbind(query_key_projection, axis=2)

        logits = paddle.matmul(queries, keys.transpose([0, 2, 1])) / (self.head_size ** 0.5)
        mask = paddle.tril(paddle.ones([sequence_length, sequence_length])).astype("bool")
        logits = paddle.where(
            mask.unsqueeze(0).expand([batch_size, sequence_length, sequence_length]),
            paddle.full_like(logits, -1e4),
            logits,
        )

        return logits


class PPDocLayoutV3MLPPredictionHead(nn.Layer):
    """Simple multi-layer perceptron for bbox prediction."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.LayerList(
            nn.Linear(n, k)
            for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class PPDocLayoutV3ConvLayer(nn.Layer):
    """Conv layer with convolution/normalization attribute names (for mask feature modules)."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation="relu"):
        super().__init__()
        self.convolution = nn.Conv2D(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride,
            padding=kernel_size // 2, bias_attr=False,
        )
        self.normalization = nn.BatchNorm2D(out_channels)
        self.activation = ACT2FN[activation] if activation is not None else nn.Identity()

    def forward(self, x):
        x = self.convolution(x)
        x = self.normalization(x)
        x = self.activation(x)
        return x


class PPDocLayoutV3ScaleHead(nn.Layer):
    def __init__(self, in_channels, feature_channels, fpn_stride, base_stride, align_corners=False):
        super().__init__()
        head_length = max(1, int(np.log2(fpn_stride) - np.log2(base_stride)))
        self.layers = nn.LayerList()
        for k in range(head_length):
            in_c = in_channels if k == 0 else feature_channels
            self.layers.append(PPDocLayoutV3ConvLayer(in_c, feature_channels, 3, 1, "silu"))
            if fpn_stride != base_stride:
                self.layers.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=align_corners))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class PPDocLayoutV3MaskFeatFPN(nn.Layer):
    def __init__(
        self,
        in_channels=[256, 256, 256],
        fpn_strides=[32, 16, 8],
        feature_channels=256,
        dropout_ratio=0.0,
        out_channels=256,
        align_corners=False,
    ):
        super().__init__()

        reorder_index = np.argsort(fpn_strides, axis=0).tolist()
        in_channels = [in_channels[i] for i in reorder_index]
        fpn_strides = [fpn_strides[i] for i in reorder_index]

        self.reorder_index = reorder_index
        self.fpn_strides = fpn_strides
        self.dropout_ratio = dropout_ratio
        self.align_corners = align_corners
        if self.dropout_ratio > 0:
            self.dropout = nn.Dropout2D(dropout_ratio)

        self.scale_heads = nn.LayerList()
        for i in range(len(fpn_strides)):
            self.scale_heads.append(
                PPDocLayoutV3ScaleHead(
                    in_channels=in_channels[i],
                    feature_channels=feature_channels,
                    fpn_stride=fpn_strides[i],
                    base_stride=fpn_strides[0],
                    align_corners=align_corners,
                )
            )
        self.output_conv = PPDocLayoutV3ConvLayer(feature_channels, out_channels, 3, 1, "silu")

    def forward(self, inputs):
        x = [inputs[i] for i in self.reorder_index]

        output = self.scale_heads[0](x[0])
        for i in range(1, len(self.fpn_strides)):
            output = output + F.interpolate(
                self.scale_heads[i](x[i]),
                size=output.shape[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )

        if self.dropout_ratio > 0:
            output = self.dropout(output)
        output = self.output_conv(output)
        return output


class PPDocLayoutV3EncoderMaskOutput(nn.Layer):
    def __init__(self, in_channels, num_prototypes):
        super().__init__()
        self.base_conv = PPDocLayoutV3ConvLayer(in_channels, in_channels, 3, 1, "silu")
        self.conv = nn.Conv2D(in_channels, num_prototypes, kernel_size=1)

    def forward(self, x):
        x = self.base_conv(x)
        x = self.conv(x)
        return x


class PPDocLayoutV3MLP(nn.Layer):
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
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        return hidden_states


class PPDocLayoutV3SelfAttention(nn.Layer):
    """Multi-headed self-attention. Position embeddings added to queries and keys (not values)."""

    def __init__(self, config, hidden_size, num_attention_heads, dropout=0.0, bias=True):
        super().__init__()
        self.head_dim = hidden_size // num_attention_heads
        self.num_heads = num_attention_heads
        self.scaling = self.head_dim ** -0.5
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

        attn_weights = paddle.matmul(query_states, key_states.transpose([0, 1, 3, 2])) * self.scaling

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, axis=-1)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        attn_output = paddle.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose([0, 2, 1, 3]).reshape([batch_size, seq_len, -1])
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class PPDocLayoutV3ConvNormLayer(nn.Layer):
    """Conv layer with conv/norm attribute names (for encoder/decoder)."""

    def __init__(self, config, in_channels, out_channels, kernel_size, stride, padding=None, activation=None):
        super().__init__()
        self.conv = nn.Conv2D(
            in_channels, out_channels, kernel_size, stride,
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


class PPDocLayoutV3EncoderLayer(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.normalize_before = config.normalize_before
        self.hidden_size = config.encoder_hidden_dim

        self.self_attn = PPDocLayoutV3SelfAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_attention_heads=config.encoder_attention_heads,
            dropout=config.dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout = config.dropout
        self.mlp = PPDocLayoutV3MLP(
            config, self.hidden_size, config.encoder_ffn_dim, config.encoder_activation_function
        )
        self.final_layer_norm = nn.LayerNorm(self.hidden_size, epsilon=config.layer_norm_eps)

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


class PPDocLayoutV3RepVggBlock(nn.Layer):
    """RepVGG architecture block."""

    def __init__(self, config):
        super().__init__()
        activation = config.activation_function
        hidden_channels = int(config.encoder_hidden_dim * config.hidden_expansion)
        self.conv1 = PPDocLayoutV3ConvNormLayer(config, hidden_channels, hidden_channels, 3, 1, padding=1)
        self.conv2 = PPDocLayoutV3ConvNormLayer(config, hidden_channels, hidden_channels, 1, 1, padding=0)
        self.activation = nn.Identity() if activation is None else ACT2CLS[activation]()

    def forward(self, x):
        y = self.conv1(x) + self.conv2(x)
        return self.activation(y)


class PPDocLayoutV3CSPRepLayer(nn.Layer):
    """Cross Stage Partial (CSP) network layer with RepVGG blocks."""

    def __init__(self, config):
        super().__init__()
        in_channels = config.encoder_hidden_dim * 2
        out_channels = config.encoder_hidden_dim
        num_blocks = 3
        activation = config.activation_function

        hidden_channels = int(out_channels * config.hidden_expansion)
        self.conv1 = PPDocLayoutV3ConvNormLayer(config, in_channels, hidden_channels, 1, 1, activation=activation)
        self.conv2 = PPDocLayoutV3ConvNormLayer(config, in_channels, hidden_channels, 1, 1, activation=activation)
        self.bottlenecks = nn.Sequential(*[PPDocLayoutV3RepVggBlock(config) for _ in range(num_blocks)])
        if hidden_channels != out_channels:
            self.conv3 = PPDocLayoutV3ConvNormLayer(config, hidden_channels, out_channels, 1, 1, activation=activation)
        else:
            self.conv3 = nn.Identity()

    def forward(self, hidden_state):
        hidden_state_1 = self.conv1(hidden_state)
        hidden_state_1 = self.bottlenecks(hidden_state_1)
        hidden_state_2 = self.conv2(hidden_state)
        return self.conv3(hidden_state_1 + hidden_state_2)


class PPDocLayoutV3SinePositionEmbedding(nn.Layer):
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
            raise ValueError("Embed dimension must be divisible by 4 for 2D sin-cos position embedding")
        pos_dim = self.embed_dim // 4
        omega = paddle.arange(pos_dim).astype(dtype) / pos_dim
        omega = 1.0 / (self.temperature ** omega)

        out_w = grid_w.flatten().unsqueeze(-1) @ omega.unsqueeze(0)
        out_h = grid_h.flatten().unsqueeze(-1) @ omega.unsqueeze(0)

        return paddle.concat([out_h.sin(), out_h.cos(), out_w.sin(), out_w.cos()], axis=1).unsqueeze(0)


class PPDocLayoutV3AIFILayer(nn.Layer):
    """AIFI (Attention-based Intra-scale Feature Interaction) layer."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder_hidden_dim = config.encoder_hidden_dim
        self.eval_size = config.eval_size

        self.position_embedding = PPDocLayoutV3SinePositionEmbedding(
            embed_dim=self.encoder_hidden_dim,
            temperature=config.positional_encoding_temperature,
        )
        self.layers = nn.LayerList([PPDocLayoutV3EncoderLayer(config) for _ in range(config.encoder_layers)])

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

        hidden_states = (
            hidden_states.transpose([0, 2, 1]).reshape([batch_size, self.encoder_hidden_dim, height, width])
        )

        return hidden_states


class PPDocLayoutV3HybridEncoder(nn.Layer):
    """Hybrid encoder: AIFI layers + top-down FPN + bottom-up PAN + mask features."""

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
        self.aifi = nn.LayerList([PPDocLayoutV3AIFILayer(config) for _ in range(len(self.encode_proj_layers))])

        # top-down FPN
        self.lateral_convs = nn.LayerList()
        self.fpn_blocks = nn.LayerList()
        for _ in range(self.num_fpn_stages):
            lateral_conv = PPDocLayoutV3ConvNormLayer(
                config,
                in_channels=self.encoder_hidden_dim,
                out_channels=self.encoder_hidden_dim,
                kernel_size=1,
                stride=1,
                activation=config.activation_function,
            )
            fpn_block = PPDocLayoutV3CSPRepLayer(config)
            self.lateral_convs.append(lateral_conv)
            self.fpn_blocks.append(fpn_block)

        # bottom-up PAN
        self.downsample_convs = nn.LayerList()
        self.pan_blocks = nn.LayerList()
        for _ in range(self.num_pan_stages):
            downsample_conv = PPDocLayoutV3ConvNormLayer(
                config,
                in_channels=self.encoder_hidden_dim,
                out_channels=self.encoder_hidden_dim,
                kernel_size=3,
                stride=2,
                activation=config.activation_function,
            )
            pan_block = PPDocLayoutV3CSPRepLayer(config)
            self.downsample_convs.append(downsample_conv)
            self.pan_blocks.append(pan_block)

        # Mask feature head (V3-specific)
        feat_strides = config.feat_strides
        mask_feature_channels = config.mask_feature_channels
        self.mask_feature_head = PPDocLayoutV3MaskFeatFPN(
            [self.encoder_hidden_dim] * len(feat_strides),
            feat_strides,
            feature_channels=mask_feature_channels[0],
            out_channels=mask_feature_channels[1],
        )
        self.encoder_mask_lateral = PPDocLayoutV3ConvLayer(config.x4_feat_dim, mask_feature_channels[1], 3, 1, "silu")
        self.encoder_mask_output = PPDocLayoutV3EncoderMaskOutput(
            in_channels=mask_feature_channels[1], num_prototypes=config.num_prototypes
        )

    def forward(self, feature_maps, x4_feat):
        # AIFI: Apply transformer encoder to specified feature levels
        if self.config.encoder_layers > 0:
            for i, enc_ind in enumerate(self.encode_proj_layers):
                feature_maps[enc_ind] = self.aifi[i](feature_maps[enc_ind])

        # top-down FPN
        fpn_feature_maps = [feature_maps[-1]]
        for idx, (lateral_conv, fpn_block) in enumerate(zip(self.lateral_convs, self.fpn_blocks)):
            backbone_feature_map = feature_maps[self.num_fpn_stages - idx - 1]
            top_fpn_feature_map = fpn_feature_maps[-1]
            top_fpn_feature_map = lateral_conv(top_fpn_feature_map)
            fpn_feature_maps[-1] = top_fpn_feature_map
            top_fpn_feature_map = F.interpolate(top_fpn_feature_map, scale_factor=2.0, mode="nearest")
            fused_feature_map = paddle.concat([top_fpn_feature_map, backbone_feature_map], axis=1)
            new_fpn_feature_map = fpn_block(fused_feature_map)
            fpn_feature_maps.append(new_fpn_feature_map)

        fpn_feature_maps.reverse()

        # bottom-up PAN
        pan_feature_maps = [fpn_feature_maps[0]]
        for idx, (downsample_conv, pan_block) in enumerate(zip(self.downsample_convs, self.pan_blocks)):
            top_pan_feature_map = pan_feature_maps[-1]
            fpn_feature_map = fpn_feature_maps[idx + 1]
            downsampled_feature_map = downsample_conv(top_pan_feature_map)
            fused_feature_map = paddle.concat([downsampled_feature_map, fpn_feature_map], axis=1)
            new_pan_feature_map = pan_block(fused_feature_map)
            pan_feature_maps.append(new_pan_feature_map)

        # Mask feature processing (V3-specific)
        mask_feat = self.mask_feature_head(pan_feature_maps)
        mask_feat = F.interpolate(mask_feat, scale_factor=2, mode="bilinear", align_corners=False)
        mask_feat = mask_feat + self.encoder_mask_lateral(x4_feat)
        mask_feat = self.encoder_mask_output(mask_feat)

        return pan_feature_maps, mask_feat


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

        attention_weights = (
            attention_weights.transpose([0, 2, 1, 3, 4])
            .reshape([batch_size * num_heads, 1, num_queries, num_levels * num_points])
        )
        output = (
            (paddle.stack(sampling_value_list, axis=-2).flatten(-2) * attention_weights)
            .sum(-1)
            .reshape([batch_size, num_heads * hidden_dim, num_queries])
        )
        return output.transpose([0, 2, 1])


class PPDocLayoutV3MultiscaleDeformableAttention(nn.Layer):
    """Multiscale deformable attention as proposed in Deformable DETR."""

    def __init__(self, config, num_heads, n_points):
        super().__init__()
        self.attn = MultiScaleDeformableAttention()

        self.d_model = config.d_model
        self.n_levels = config.num_feature_levels
        self.n_heads = num_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(config.d_model, num_heads * self.n_levels * n_points * 2)
        self.attention_weights = nn.Linear(config.d_model, num_heads * self.n_levels * n_points)
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
        value = value.reshape([batch_size, sequence_length, self.n_heads, self.d_model // self.n_heads])
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
                + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError(f"Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]}")

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


class PPDocLayoutV3DecoderLayer(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.d_model

        self.self_attn = PPDocLayoutV3SelfAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_attention_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.dropout = config.dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.hidden_size, epsilon=config.layer_norm_eps)
        self.encoder_attn = PPDocLayoutV3MultiscaleDeformableAttention(
            config,
            num_heads=config.decoder_attention_heads,
            n_points=config.decoder_n_points,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.hidden_size, epsilon=config.layer_norm_eps)
        self.mlp = PPDocLayoutV3MLP(
            config, self.hidden_size, config.decoder_ffn_dim, config.decoder_activation_function
        )
        self.final_layer_norm = nn.LayerNorm(self.hidden_size, epsilon=config.layer_norm_eps)

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


class PPDocLayoutV3Decoder(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.dropout = config.dropout
        self.layers = nn.LayerList([PPDocLayoutV3DecoderLayer(config) for _ in range(config.decoder_layers)])
        self.query_pos_head = PPDocLayoutV3MLPPredictionHead(4, 2 * config.d_model, config.d_model, num_layers=2)

        # Set by PPDocLayoutV3Model
        self.bbox_embed = None
        self.class_embed = None

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
        order_head=None,
        global_pointer=None,
        mask_query_head=None,
        norm=None,
        mask_feat=None,
    ):
        hidden_states = inputs_embeds

        intermediate = []
        intermediate_reference_points = []
        intermediate_logits = []
        decoder_out_order_logits = []
        decoder_out_masks = []

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

            # Iterative bounding box refinement
            if self.bbox_embed is not None:
                predicted_corners = self.bbox_embed(hidden_states)
                new_reference_points = F.sigmoid(predicted_corners + inverse_sigmoid(reference_points))
                reference_points = new_reference_points.detach()

            intermediate.append(hidden_states)
            intermediate_reference_points.append(
                new_reference_points if self.bbox_embed is not None else reference_points
            )

            # Mask and class prediction at each decoder layer
            out_query = norm(hidden_states)
            mask_query_embed = mask_query_head(out_query)
            batch_size, mask_dim, _ = mask_query_embed.shape
            _, _, mask_h, mask_w = mask_feat.shape
            out_mask = paddle.bmm(mask_query_embed, mask_feat.flatten(start_axis=2)).reshape(
                [batch_size, mask_dim, mask_h, mask_w]
            )
            decoder_out_masks.append(out_mask)

            if self.class_embed is not None:
                logits = self.class_embed(out_query)
                intermediate_logits.append(logits)

            if order_head is not None and global_pointer is not None:
                valid_query = out_query[:, -self.num_queries:] if self.num_queries is not None else out_query
                order_logits = global_pointer(order_head[idx](valid_query))
                decoder_out_order_logits.append(order_logits)

        intermediate = paddle.stack(intermediate, axis=1)
        intermediate_reference_points = paddle.stack(intermediate_reference_points, axis=1)
        if self.class_embed is not None:
            intermediate_logits = paddle.stack(intermediate_logits, axis=1)
        if order_head is not None and global_pointer is not None:
            decoder_out_order_logits = paddle.stack(decoder_out_order_logits, axis=1)
        decoder_out_masks = paddle.stack(decoder_out_masks, axis=1)

        return {
            "last_hidden_state": hidden_states,
            "intermediate_hidden_states": intermediate,
            "intermediate_logits": intermediate_logits,
            "intermediate_reference_points": intermediate_reference_points,
            "out_order_logits": decoder_out_order_logits,
            "out_masks": decoder_out_masks,
        }


def replace_batch_norm(model):
    """Recursively replace all nn.BatchNorm2D with PPDocLayoutV3FrozenBatchNorm2d."""
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2D):
            new_module = PPDocLayoutV3FrozenBatchNorm2d(module._num_features)
            new_module.weight.set_value(module.weight)
            new_module.bias.set_value(module.bias)
            new_module._mean.set_value(module._mean)
            new_module._variance.set_value(module._variance)
            setattr(model, name, new_module)

        if len(list(module.children())) > 0:
            replace_batch_norm(module)


class PPDocLayoutV3ConvEncoder(nn.Layer):
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


class PPDocLayoutV3Model(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Create backbone
        self.backbone = PPDocLayoutV3ConvEncoder(config)
        intermediate_channel_sizes = self.backbone.intermediate_channel_sizes

        # Create encoder input projection layers (skip stage1, project stages 2-4)
        encoder_input_proj_list = []
        for i in range(len(intermediate_channel_sizes)):
            in_channels = intermediate_channel_sizes[i]
            encoder_input_proj_list.append(
                nn.Sequential(
                    nn.Conv2D(in_channels, config.encoder_hidden_dim, kernel_size=1, bias_attr=False),
                    nn.BatchNorm2D(config.encoder_hidden_dim),
                )
            )
        self.encoder_input_proj = nn.LayerList(encoder_input_proj_list[1:])

        # Create encoder
        self.encoder = PPDocLayoutV3HybridEncoder(config)

        # denoising embedding (ForObjectDetection version: no padding_idx)
        self.denoising_class_embed = nn.Embedding(config.num_labels, config.d_model)

        # decoder embedding
        if config.learn_initial_query:
            self.weight_embedding = nn.Embedding(config.num_queries, config.d_model)

        # encoder head
        self.enc_output = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model, epsilon=config.layer_norm_eps),
        )
        self.enc_score_head = nn.Linear(config.d_model, config.num_labels)
        self.enc_bbox_head = PPDocLayoutV3MLPPredictionHead(config.d_model, config.d_model, 4, num_layers=3)

        # Create decoder input projection layers
        num_backbone_outs = len(config.decoder_in_channels)
        decoder_input_proj_list = []
        for i in range(num_backbone_outs):
            in_channels = config.decoder_in_channels[i]
            decoder_input_proj_list.append(
                nn.Sequential(
                    nn.Conv2D(in_channels, config.d_model, kernel_size=1, bias_attr=False),
                    nn.BatchNorm2D(config.d_model, epsilon=config.batch_norm_eps),
                )
            )
        for _ in range(config.num_feature_levels - num_backbone_outs):
            decoder_input_proj_list.append(
                nn.Sequential(
                    nn.Conv2D(in_channels, config.d_model, kernel_size=3, stride=2, padding=1, bias_attr=False),
                    nn.BatchNorm2D(config.d_model, epsilon=config.batch_norm_eps),
                )
            )
            in_channels = config.d_model
        self.decoder_input_proj = nn.LayerList(decoder_input_proj_list)

        # decoder
        self.decoder = PPDocLayoutV3Decoder(config)

        # Order prediction (V3-specific)
        self.decoder_order_head = nn.LayerList(
            [nn.Linear(config.d_model, config.d_model) for _ in range(config.decoder_layers)]
        )
        self.decoder_global_pointer = PPDocLayoutV3GlobalPointer(config)
        self.decoder_norm = nn.LayerNorm(config.d_model, epsilon=config.layer_norm_eps)

        # Tie decoder class_embed and bbox_embed to encoder heads (weight sharing)
        self.decoder.class_embed = self.enc_score_head
        self.decoder.bbox_embed = self.enc_bbox_head

        # Mask (V3-specific)
        self.mask_enhanced = config.mask_enhanced
        self.mask_query_head = PPDocLayoutV3MLPPredictionHead(
            config.d_model, config.d_model, config.num_prototypes, num_layers=3
        )

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
            wh = paddle.ones_like(grid_xy) * grid_size * (2.0 ** level)
            anchors.append(paddle.concat([grid_xy, wh], axis=-1).reshape([-1, height * width, 4]))
        eps = 1e-2
        anchors = paddle.concat(anchors, axis=1)
        valid_mask = ((anchors > eps) & (anchors < 1 - eps)).all(axis=-1, keepdim=True)
        anchors = paddle.log(anchors / (1 - anchors))
        anchors = paddle.where(valid_mask, anchors, paddle.full_like(anchors, float("inf")))
        return anchors, valid_mask

    def forward(self, pixel_values, pixel_mask=None):
        batch_size, num_channels, height, width = pixel_values.shape

        if pixel_mask is None:
            pixel_mask = paddle.ones([batch_size, height, width])

        # Backbone: returns 4 features (stage1..stage4)
        features = self.backbone(pixel_values)
        x4_feat = features[0]  # stage1 feature for mask lateral
        remaining_features = features[1:]  # stages 2-4
        proj_feats = [self.encoder_input_proj[level](source) for level, source in enumerate(remaining_features)]

        # Encoder (hybrid encoder + mask feature head)
        encoder_outputs, mask_feat = self.encoder(proj_feats, x4_feat)

        # _get_encoder_input
        sources = []
        for level, source in enumerate(encoder_outputs):
            sources.append(self.decoder_input_proj[level](source))

        if self.config.num_feature_levels > len(sources):
            _len_sources = len(sources)
            sources.append(self.decoder_input_proj[_len_sources](encoder_outputs[-1]))
            for i in range(_len_sources + 1, self.config.num_feature_levels):
                sources.append(self.decoder_input_proj[i](encoder_outputs[-1]))

        # Prepare encoder inputs (by flattening)
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

        # No denoising at inference
        batch_size = source_flatten.shape[0]
        dtype = source_flatten.dtype

        spatial_shapes_tuple = tuple(spatial_shapes_list)
        anchors, valid_mask = self.generate_anchors(spatial_shapes_tuple, dtype=dtype)

        memory = valid_mask.astype(source_flatten.dtype) * source_flatten

        output_memory = self.enc_output(memory)

        enc_outputs_class = self.enc_score_head(output_memory)
        enc_outputs_coord_logits = self.enc_bbox_head(output_memory) + anchors

        _, topk_ind = paddle.topk(enc_outputs_class.max(-1), self.config.num_queries, axis=1)

        reference_points_unact = paddle.take_along_axis(
            enc_outputs_coord_logits,
            topk_ind.unsqueeze(-1).expand([-1, -1, enc_outputs_coord_logits.shape[-1]]),
            axis=1,
        )

        # _get_pred_class_and_mask
        batch_ind = paddle.arange(memory.shape[0]).unsqueeze(1)
        target = output_memory[batch_ind, topk_ind]
        out_query = self.decoder_norm(target)
        mask_query_embed = self.mask_query_head(out_query)
        batch_size_m, mask_dim, _ = mask_query_embed.shape

        enc_topk_bboxes = F.sigmoid(reference_points_unact)

        # extract region features
        if self.config.learn_initial_query:
            target = self.weight_embedding.weight.unsqueeze(0).expand([batch_size, -1, -1])
        else:
            target = paddle.take_along_axis(
                output_memory,
                topk_ind.unsqueeze(-1).expand([-1, -1, output_memory.shape[-1]]),
                axis=1,
            )
            target = target.detach()

        if self.mask_enhanced:
            _, _, mask_h, mask_w = mask_feat.shape
            enc_out_masks = paddle.bmm(mask_query_embed, mask_feat.flatten(start_axis=2)).reshape(
                [batch_size_m, mask_dim, mask_h, mask_w]
            )
            reference_points = mask_to_box_coordinate(enc_out_masks > 0, dtype=reference_points_unact.dtype)
            reference_points_unact = inverse_sigmoid(reference_points)

        init_reference_points = reference_points_unact.detach()

        # decoder
        decoder_outputs = self.decoder(
            inputs_embeds=target,
            encoder_hidden_states=source_flatten,
            encoder_attention_mask=None,
            reference_points=init_reference_points,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            level_start_index=level_start_index,
            order_head=self.decoder_order_head,
            global_pointer=self.decoder_global_pointer,
            mask_query_head=self.mask_query_head,
            norm=self.decoder_norm,
            mask_feat=mask_feat,
        )

        return decoder_outputs


class PPDocLayoutV3PostProcess:
    def __init__(
        self,
        num_classes=25,
        num_top_queries=300,
        use_focal_loss=True,
        bbox_decode_type="origin",
    ):
        self.num_classes = num_classes
        self.num_top_queries = num_top_queries
        self.use_focal_loss = use_focal_loss
        self.bbox_decode_type = bbox_decode_type

    def __call__(self, head_out, order_logits, im_shape, scale_factor, pad_shape):
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

        scores = F.sigmoid(logits) if self.use_focal_loss else F.softmax(logits)[:, :, :-1]

        pad_order_seq, pad_order_votes = get_order(order_logits)

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
                pad_order_seq = paddle.gather_nd(pad_order_seq, index)
                pad_order_votes = paddle.gather_nd(pad_order_votes, index)
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
            pad_order_seq = paddle.gather_nd(pad_order_seq, index)
            pad_order_votes = paddle.gather_nd(pad_order_votes, index)

        bbox_pred = paddle.concat(
            [
                labels.unsqueeze(-1).astype("float32"),
                scores.unsqueeze(-1),
                bbox_pred,
                pad_order_seq.unsqueeze(-1).astype("float32"),
                pad_order_votes.unsqueeze(-1).astype("float32"),
            ],
            axis=-1,
        )
        bbox_num = paddle.to_tensor(self.num_top_queries, dtype="int32").tile(
            [bbox_pred.shape[0]]
        )
        bbox_pred = bbox_pred.reshape([-1, 8])
        return bbox_pred, bbox_num


class PPDocLayoutV3(BatchNormHFStateDictMixin, PretrainedModel):

    config_class = PPDocLayoutV3Config

    def __init__(self, config):
        super(PPDocLayoutV3, self).__init__(config)
        self.config = config

        # Build the detection model
        self.model = PPDocLayoutV3Model(config)

        # Post-process
        self.post_process = PPDocLayoutV3PostProcess(
            num_top_queries=config.num_queries,
            use_focal_loss=True,
        )

    def forward(self, inputs):
        pixel_values = paddle.to_tensor(inputs[1])
        im_shape = paddle.to_tensor(inputs[0])
        scale_factor = paddle.to_tensor(inputs[2])

        # Run the detection model
        decoder_outputs = self.model(pixel_values)

        intermediate_reference_points = decoder_outputs["intermediate_reference_points"]
        intermediate_logits = decoder_outputs["intermediate_logits"]
        order_logits = decoder_outputs["out_order_logits"]

        # Take last layer outputs
        pred_boxes = intermediate_reference_points[:, -1]
        logits = intermediate_logits[:, -1]
        order_logits = order_logits[:, -1]

        # Post-process
        head_out = (pred_boxes, logits)
        pad_shape = paddle.to_tensor(
            [[pixel_values.shape[2], pixel_values.shape[3]]] * pixel_values.shape[0],
            dtype="float32",
        )
        bbox, bbox_num = self.post_process(
            head_out,
            order_logits,
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
            "decoder_order_head",
            "decoder_global_pointer",
            "mask_query_head",
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
        # decoder.class_embed and decoder.bbox_embed are tied to enc_score_head and
        # enc_bbox_head respectively (HF _tied_weights_keys). Safetensors only stores
        # the canonical enc_*_head keys, so we duplicate them under the decoder paths
        # so that Paddle's set_state_dict does not warn about missing keys.
        aliases = {}
        for k, v in converted.items():
            if k.startswith("model.enc_score_head."):
                aliases[k.replace("model.enc_score_head.", "model.decoder.class_embed.")] = v
            elif k.startswith("model.enc_bbox_head."):
                aliases[k.replace("model.enc_bbox_head.", "model.decoder.bbox_embed.")] = v
        converted.update(aliases)
        return super().set_hf_state_dict(converted, *args, **kwargs)

    def get_hf_state_dict(self, *args, **kwargs):
        state_dict = _reverse_rt_detr_key_conversion(
            super().get_hf_state_dict(*args, **kwargs)
        )
        # Remove tied-weight duplicates: decoder.class_embed and decoder.bbox_embed
        # are aliases of enc_score_head and enc_bbox_head; only keep the canonical keys.
        return {
            k: v
            for k, v in state_dict.items()
            if not k.startswith("model.decoder.class_embed.")
            and not k.startswith("model.decoder.bbox_embed.")
        }
