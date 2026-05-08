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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ...common.transformers.activations import ACT2FN, ACT2CLS
from ...common.transformers.transformers import (
    BatchNormHFStateDictMixin,
    PretrainedModel,
)
from ...image_classification.modeling.hgnetv2 import HGNetV2Backbone
from ._config_pp_doclayout_v2 import PPDocLayoutV2Config


def bbox_cxcywh_to_xyxy(x):
    cxcy, wh = paddle.split(x, 2, axis=-1)
    return paddle.concat([cxcy - 0.5 * wh, cxcy + 0.5 * wh], axis=-1)

__all__ = ["PPDocLayoutV2"]


def _apply_rt_detr_key_conversion(state_dict):
    """Convert safetensors old key names to new HF transformers key names.

    Matches the rt_detr conversion_mapping in HF transformers:
    - out_proj -> o_proj
    - layers.N.fc1 -> layers.N.mlp.fc1
    - layers.N.fc2 -> layers.N.mlp.fc2
    - encoder.encoder.N.layers -> encoder.aifi.N.layers
    """
    import re

    new_sd = {}
    for k, v in state_dict.items():
        k = k.replace("out_proj", "o_proj")
        k = re.sub(r"layers\.(\d+)\.fc1", r"layers.\1.mlp.fc1", k)
        k = re.sub(r"layers\.(\d+)\.fc2", r"layers.\1.mlp.fc2", k)
        k = re.sub(r"encoder\.encoder\.(\d+)\.layers", r"encoder.aifi.\1.layers", k)
        new_sd[k] = v
    return new_sd


def _reverse_rt_detr_key_conversion(state_dict):
    """Reverse conversion: new HF key names back to safetensors old key names."""
    import re

    new_sd = {}
    for k, v in state_dict.items():
        k = k.replace("o_proj", "out_proj")
        k = re.sub(r"layers\.(\d+)\.mlp\.fc1", r"layers.\1.fc1", k)
        k = re.sub(r"layers\.(\d+)\.mlp\.fc2", r"layers.\1.fc2", k)
        k = re.sub(r"encoder\.aifi\.(\d+)\.layers", r"encoder.encoder.\1.layers", k)
        new_sd[k] = v
    return new_sd



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


# PPDocLayoutV2FrozenBatchNorm2d

class PPDocLayoutV2FrozenBatchNorm2d(nn.Layer):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    """

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



class PPDocLayoutV2GlobalPointer(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.head_size = config.global_pointer_head_size
        self.dense = nn.Linear(config.hidden_size, self.head_size * 2)
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
        # masked_fill: where mask is True, fill with -1e4
        logits = paddle.where(
            mask.unsqueeze(0).expand([batch_size, sequence_length, sequence_length]),
            paddle.full_like(logits, -1e4),
            logits,
        )

        return logits


class PPDocLayoutV2PositionRelationEmbedding(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.relation_bias_embed_dim
        self.scale = config.relation_bias_scale
        self.pos_proj = nn.Conv2D(
            in_channels=self.embed_dim * 4,
            out_channels=config.num_attention_heads,
            kernel_size=1,
        )
        inv_freq = self._compute_inv_freq(config)
        self.register_buffer("inv_freq", inv_freq, persistable=False)

    @staticmethod
    def _compute_inv_freq(config):
        base = config.relation_bias_theta
        dim = config.relation_bias_embed_dim
        half_dim = dim // 2
        inv_freq = 1.0 / (
            base ** (paddle.arange(0, dim, 2).astype("float32") / half_dim)
        )
        return inv_freq

    def box_relative_encoding(self, source_boxes, target_boxes=None, epsilon=1e-5):
        source_boxes = source_boxes.unsqueeze(-2)
        target_boxes = target_boxes.unsqueeze(-3)
        source_coordinates, source_dim = source_boxes[..., :2], source_boxes[..., 2:]
        target_coordinates, target_dim = target_boxes[..., :2], target_boxes[..., 2:]

        coordinate_difference = paddle.abs(source_coordinates - target_coordinates)
        relative_coordinates = paddle.log(coordinate_difference / (source_dim + epsilon) + 1.0)
        relative_dim = paddle.log((source_dim + epsilon) / (target_dim + epsilon))

        relative_encoding = paddle.concat([relative_coordinates, relative_dim], axis=-1)
        return relative_encoding

    def get_position_embedding(self, x, scale=100.0):
        embedding = (x * scale).unsqueeze(-1) * self.inv_freq
        embedding = paddle.concat(
            [embedding.sin(), embedding.cos()], axis=-1
        ).flatten(start_axis=-2).astype(x.dtype)
        return embedding

    def forward(self, source_boxes, target_boxes=None):
        if target_boxes is None:
            target_boxes = source_boxes
        with paddle.no_grad():
            relative_encoding = self.box_relative_encoding(source_boxes, target_boxes)
            position_embedding = self.get_position_embedding(relative_encoding, self.scale)
            position_embedding = position_embedding.transpose([0, 3, 1, 2])
        out = self.pos_proj(position_embedding)
        return out


class PPDocLayoutV2ReadingOrderSelfAttention(nn.Layer):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.has_relative_attention_bias = config.has_relative_attention_bias
        self.has_spatial_attention_bias = config.has_spatial_attention_bias

    def cogview_attention(self, attention_scores, alpha=32):
        scaled_attention_scores = attention_scores / alpha
        max_value = scaled_attention_scores.max(axis=-1, keepdim=True)
        new_attention_scores = (scaled_attention_scores - max_value) * alpha
        return nn.functional.softmax(new_attention_scores, axis=-1)

    def forward(self, hidden_states, attention_mask=None, rel_pos=None, rel_2d_pos=None):
        batch_size, seq_length, _ = hidden_states.shape
        query_layer = (
            self.query(hidden_states)
            .reshape([batch_size, -1, self.num_attention_heads, self.attention_head_size])
            .transpose([0, 2, 1, 3])
        )
        key_layer = (
            self.key(hidden_states)
            .reshape([batch_size, -1, self.num_attention_heads, self.attention_head_size])
            .transpose([0, 2, 1, 3])
        )
        value_layer = (
            self.value(hidden_states)
            .reshape([batch_size, -1, self.num_attention_heads, self.attention_head_size])
            .transpose([0, 2, 1, 3])
        )

        attention_scores = paddle.matmul(
            query_layer / math.sqrt(self.attention_head_size),
            key_layer.transpose([0, 1, 3, 2]),
        )

        if rel_2d_pos is not None:
            attention_scores += rel_2d_pos
        elif self.has_relative_attention_bias:
            attention_scores += rel_pos / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = self.cogview_attention(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = paddle.matmul(attention_probs, value_layer)
        context_layer = context_layer.transpose([0, 2, 1, 3])
        new_context_layer_shape = list(context_layer.shape[:-2]) + [self.all_head_size]
        context_layer = context_layer.reshape(new_context_layer_shape)

        return context_layer, attention_probs


class PPDocLayoutV2ReadingOrderSelfOutput(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.norm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.norm(hidden_states + input_tensor)
        return hidden_states


class PPDocLayoutV2ReadingOrderIntermediate(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class PPDocLayoutV2ReadingOrderOutput(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.norm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.norm(hidden_states + input_tensor)
        return hidden_states


class PPDocLayoutV2ReadingOrderAttention(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.self = PPDocLayoutV2ReadingOrderSelfAttention(config)
        self.output = PPDocLayoutV2ReadingOrderSelfOutput(config)

    def forward(self, hidden_states, attention_mask=None, rel_pos=None, rel_2d_pos=None):
        residual = hidden_states
        attention_output, _ = self.self(
            hidden_states,
            attention_mask,
            rel_pos=rel_pos,
            rel_2d_pos=rel_2d_pos,
        )
        attention_output = self.output(attention_output, residual)
        return attention_output


class PPDocLayoutV2ReadingOrderLayer(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = PPDocLayoutV2ReadingOrderAttention(config)
        self.intermediate = PPDocLayoutV2ReadingOrderIntermediate(config)
        self.output = PPDocLayoutV2ReadingOrderOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        rel_pos=None,
        rel_2d_pos=None,
    ):
        attention_output = self.attention(
            hidden_states,
            attention_mask,
            rel_pos=rel_pos,
            rel_2d_pos=rel_2d_pos,
        )
        layer_output = self.feed_forward_chunk(attention_output)
        return layer_output

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class PPDocLayoutV2ReadingOrderEncoder(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.LayerList(
            [PPDocLayoutV2ReadingOrderLayer(config) for _ in range(config.num_hidden_layers)]
        )

        self.has_relative_attention_bias = config.has_relative_attention_bias
        self.has_spatial_attention_bias = config.has_spatial_attention_bias

        if self.has_relative_attention_bias:
            self.rel_pos_bins = config.rel_pos_bins
            self.max_rel_pos = config.max_rel_pos
            self.rel_pos_bias = nn.Linear(self.rel_pos_bins, config.num_attention_heads, bias_attr=False)

        if self.has_spatial_attention_bias:
            self.max_rel_2d_pos = config.max_rel_2d_pos
            self.rel_2d_pos_bins = config.rel_2d_pos_bins
        self.rel_bias_module = PPDocLayoutV2PositionRelationEmbedding(config)

    def relative_position_bucket(self, relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        ret = 0
        if bidirectional:
            num_buckets //= 2
            ret += (relative_position > 0).astype("int64") * num_buckets
            n = paddle.abs(relative_position)
        else:
            n = paddle.maximum(-relative_position, paddle.zeros_like(relative_position))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            paddle.log(n.astype("float32") / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).astype("int64")
        val_if_large = paddle.minimum(val_if_large, paddle.full_like(val_if_large, num_buckets - 1))

        ret += paddle.where(is_small, n, val_if_large)
        return ret

    def _cal_1d_pos_emb(self, position_ids):
        rel_pos_mat = position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1)

        rel_pos = self.relative_position_bucket(
            rel_pos_mat,
            num_buckets=self.rel_pos_bins,
            max_distance=self.max_rel_pos,
        )
        with paddle.no_grad():
            # self.rel_pos_bias.weight has shape [rel_pos_bins, num_attention_heads] in Paddle
            # We need: rel_pos_bias.weight.T[rel_pos] -> [B, seq, seq, num_heads] -> permute to [B, num_heads, seq, seq]
            rel_pos = self.rel_pos_bias.weight.transpose([1, 0])[rel_pos].transpose([0, 3, 1, 2])
        return rel_pos

    def _cal_2d_pos_emb(self, bbox):
        x_min, y_min, x_max, y_max = (
            bbox[..., 0],
            bbox[..., 1],
            bbox[..., 2],
            bbox[..., 3],
        )

        width = (x_max - x_min).clip(min=1e-3)
        height = (y_max - y_min).clip(min=1e-3)

        center_x = (x_min + x_max) * 0.5
        center_y = (y_min + y_max) * 0.5

        center_width_height_bbox = paddle.stack([center_x, center_y, width, height], axis=-1)

        result = self.rel_bias_module(center_width_height_bbox)
        return result

    def forward(
        self,
        hidden_states,
        bbox=None,
        attention_mask=None,
        position_ids=None,
    ):
        rel_pos = self._cal_1d_pos_emb(position_ids) if self.has_relative_attention_bias else None
        rel_2d_pos = self._cal_2d_pos_emb(bbox) if self.has_spatial_attention_bias else None

        for layer_module in self.layer:
            hidden_states = layer_module(
                hidden_states,
                attention_mask,
                rel_pos=rel_pos,
                rel_2d_pos=rel_2d_pos,
            )

        return hidden_states


class PPDocLayoutV2TextEmbeddings(nn.Layer):
    """PPDocLayoutV2 text embeddings with spatial (layout) embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.register_buffer(
            "position_ids",
            paddle.arange(config.max_position_embeddings).unsqueeze(0),
            persistable=False,
        )

        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

        self.x_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.coordinate_size)
        self.y_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.coordinate_size)
        self.h_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.shape_size)
        self.w_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.shape_size)
        self.norm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        spatial_embed_dim = 4 * config.coordinate_size + 2 * config.shape_size
        self.spatial_proj = nn.Linear(spatial_embed_dim, config.hidden_size)

    def calculate_spatial_position_embeddings(self, bbox):
        left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
        upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
        right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
        lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])

        h_position_embeddings = self.h_position_embeddings(
            paddle.clip(bbox[:, :, 3] - bbox[:, :, 1], 0, 1023)
        )
        w_position_embeddings = self.w_position_embeddings(
            paddle.clip(bbox[:, :, 2] - bbox[:, :, 0], 0, 1023)
        )

        spatial_position_embeddings = paddle.concat(
            [
                left_position_embeddings,
                upper_position_embeddings,
                right_position_embeddings,
                lower_position_embeddings,
                h_position_embeddings,
                w_position_embeddings,
            ],
            axis=-1,
        )
        return spatial_position_embeddings

    def create_position_ids_from_input_ids(self, input_ids, padding_idx):
        mask = (input_ids != padding_idx).astype("int32")
        incremental_indices = paddle.cumsum(mask, axis=1).astype(mask.dtype) * mask
        return incremental_indices.astype("int64") + padding_idx

    def forward(
        self,
        input_ids=None,
        bbox=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
    ):
        if position_ids is None:
            if input_ids is not None:
                position_ids = self.create_position_ids_from_input_ids(input_ids, self.padding_idx)
            else:
                input_shape = inputs_embeds.shape[:-1]
                sequence_length = input_shape[1]
                position_ids = paddle.arange(
                    self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype="int64"
                ).unsqueeze(0).expand(input_shape)

        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        if token_type_ids is None:
            token_type_ids = paddle.zeros(input_shape, dtype="int64")

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings

        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings

        spatial_position_embeddings = self.calculate_spatial_position_embeddings(bbox)
        spatial_position_embeddings = self.spatial_proj(spatial_position_embeddings)
        embeddings += spatial_position_embeddings
        return embeddings


class PPDocLayoutV2ReadingOrder(nn.Layer):
    """PP-DocLayoutV2 ReadingOrder Model with encoder and GlobalPointer head."""

    def __init__(self, config):
        super().__init__()
        self.embeddings = PPDocLayoutV2TextEmbeddings(config)
        self.label_embeddings = nn.Embedding(config.num_classes, config.hidden_size)
        self.label_features_projection = nn.Linear(config.hidden_size, config.hidden_size)
        self.encoder = PPDocLayoutV2ReadingOrderEncoder(config)
        self.relative_head = PPDocLayoutV2GlobalPointer(config)
        self.config = config

    def forward(self, boxes, labels=None, mask=None):
        batch_size, seq_len = mask.shape
        num_pred = mask.sum(axis=1)

        input_ids = paddle.full(
            [batch_size, seq_len + 2], self.config.pad_token_id, dtype="int64"
        )
        input_ids[:, 0] = self.config.start_token_id

        pred_col_idx = paddle.arange(seq_len + 2).unsqueeze(0)
        pred_mask = (pred_col_idx >= 1) & (pred_col_idx <= num_pred.unsqueeze(1))
        input_ids[pred_mask] = self.config.pred_token_id
        end_col_indices = num_pred + 1
        input_ids[paddle.arange(batch_size), end_col_indices] = self.config.end_token_id

        pad_box = paddle.zeros(shape=[boxes.shape[0], 1, boxes.shape[-1]], dtype=boxes.dtype)
        pad_boxes = paddle.concat([pad_box, boxes, pad_box], axis=1)
        bbox_embedding = self.embeddings(input_ids=input_ids, bbox=pad_boxes.astype("int64"))

        if labels is not None:
            label_embs = self.label_embeddings(labels)
            label_proj = self.label_features_projection(label_embs)
            pad = paddle.zeros(
                shape=[label_proj.shape[0], 1, label_proj.shape[-1]], dtype=label_proj.dtype
            )
            label_proj = paddle.concat([pad, label_proj, pad], axis=1)
        else:
            label_proj = paddle.zeros_like(bbox_embedding)

        final_embeddings = bbox_embedding + label_proj
        final_embeddings = self.embeddings.norm(final_embeddings)
        final_embeddings = self.embeddings.dropout(final_embeddings)

        # Create attention mask: True for valid positions
        attention_mask_bool = pred_col_idx < (num_pred + 2).unsqueeze(1)
        # Convert to additive mask: 0 for valid, large negative for invalid
        attention_mask = paddle.zeros_like(attention_mask_bool, dtype=final_embeddings.dtype)
        attention_mask = paddle.where(
            attention_mask_bool,
            paddle.zeros_like(attention_mask, dtype=final_embeddings.dtype),
            paddle.full_like(attention_mask, -1e9, dtype=final_embeddings.dtype),
        )
        # Expand to [batch, 1, 1, seq_len] for broadcasting with attention scores [batch, heads, seq, seq]
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        position_ids = paddle.arange(seq_len + 2).unsqueeze(0).expand([batch_size, seq_len + 2])

        encoder_output = self.encoder(
            hidden_states=final_embeddings,
            bbox=pad_boxes,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        token = encoder_output[:, 1: 1 + seq_len, :]
        read_order_logits = self.relative_head(token)
        return read_order_logits


# Detection model components

class PPDocLayoutV2MLPPredictionHead(nn.Layer):
    """Simple multi-layer perceptron for bbox prediction."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.LayerList(
            [nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])]
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = nn.functional.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class PPDocLayoutV2MLP(nn.Layer):
    def __init__(self, config, hidden_size, intermediate_size, activation_function):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.activation_fn = ACT2FN[activation_function]
        self.activation_dropout = config.activation_dropout
        self.dropout = config.dropout

    def forward(self, hidden_states):
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        return hidden_states


class PPDocLayoutV2SelfAttention(nn.Layer):
    """Multi-headed self-attention. Position embeddings added to queries and keys."""

    def __init__(self, config, hidden_size, num_attention_heads, dropout=0.0, bias=True):
        super().__init__()
        self.config = config
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

        query_key_input = hidden_states + position_embeddings if position_embeddings is not None else hidden_states

        query_states = self.q_proj(query_key_input).reshape(
            [batch_size, seq_len, self.num_heads, self.head_dim]
        ).transpose([0, 2, 1, 3])
        key_states = self.k_proj(query_key_input).reshape(
            [batch_size, seq_len, self.num_heads, self.head_dim]
        ).transpose([0, 2, 1, 3])
        value_states = self.v_proj(hidden_states).reshape(
            [batch_size, seq_len, self.num_heads, self.head_dim]
        ).transpose([0, 2, 1, 3])

        attn_weights = paddle.matmul(query_states, key_states.transpose([0, 1, 3, 2])) * self.scaling

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, axis=-1)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        attn_output = paddle.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose([0, 2, 1, 3]).reshape([batch_size, seq_len, -1])
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class PPDocLayoutV2ConvNormLayer(nn.Layer):
    def __init__(self, config, in_channels, out_channels, kernel_size, stride, padding=None, activation=None):
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


class PPDocLayoutV2EncoderLayer(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.normalize_before = config.normalize_before
        self.hidden_size = config.encoder_hidden_dim

        self.self_attn = PPDocLayoutV2SelfAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_attention_heads=config.num_attention_heads,
            dropout=config.dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout_val = config.dropout
        self.mlp = PPDocLayoutV2MLP(
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

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout_val, training=self.training)
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


class PPDocLayoutV2RepVggBlock(nn.Layer):
    """RepVGG architecture block."""

    def __init__(self, config):
        super().__init__()
        activation = config.activation_function
        hidden_channels = int(config.encoder_hidden_dim * config.hidden_expansion)
        self.conv1 = PPDocLayoutV2ConvNormLayer(config, hidden_channels, hidden_channels, 3, 1, padding=1)
        self.conv2 = PPDocLayoutV2ConvNormLayer(config, hidden_channels, hidden_channels, 1, 1, padding=0)
        self.activation = nn.Identity() if activation is None else ACT2CLS[activation]()

    def forward(self, x):
        y = self.conv1(x) + self.conv2(x)
        return self.activation(y)


class PPDocLayoutV2CSPRepLayer(nn.Layer):
    """Cross Stage Partial (CSP) network layer with RepVGG blocks."""

    def __init__(self, config):
        super().__init__()
        in_channels = config.encoder_hidden_dim * 2
        out_channels = config.encoder_hidden_dim
        num_blocks = 3
        activation = config.activation_function

        hidden_channels = int(out_channels * config.hidden_expansion)
        self.conv1 = PPDocLayoutV2ConvNormLayer(config, in_channels, hidden_channels, 1, 1, activation=activation)
        self.conv2 = PPDocLayoutV2ConvNormLayer(config, in_channels, hidden_channels, 1, 1, activation=activation)
        self.bottlenecks = nn.Sequential(*[PPDocLayoutV2RepVggBlock(config) for _ in range(num_blocks)])
        if hidden_channels != out_channels:
            self.conv3 = PPDocLayoutV2ConvNormLayer(config, hidden_channels, out_channels, 1, 1, activation=activation)
        else:
            self.conv3 = nn.Identity()

    def forward(self, hidden_state):
        hidden_state_1 = self.conv1(hidden_state)
        hidden_state_1 = self.bottlenecks(hidden_state_1)
        hidden_state_2 = self.conv2(hidden_state)
        return self.conv3(hidden_state_1 + hidden_state_2)


class PPDocLayoutV2SinePositionEmbedding(nn.Layer):
    """2D sinusoidal position embedding used in RT-DETR hybrid encoder."""

    def __init__(self, embed_dim=256, temperature=10000):
        super().__init__()
        self.embed_dim = embed_dim
        self.temperature = temperature

    def forward(self, width, height, dtype):
        grid_w = paddle.arange(width).astype(dtype)
        grid_h = paddle.arange(height).astype(dtype)
        # paddle.meshgrid default indexing is "ij", so swap order for "xy" effect
        grid_h, grid_w = paddle.meshgrid(grid_h, grid_w)

        if self.embed_dim % 4 != 0:
            raise ValueError("Embed dimension must be divisible by 4 for 2D sin-cos position embedding")
        pos_dim = self.embed_dim // 4
        omega = paddle.arange(pos_dim).astype(dtype) / pos_dim
        omega = 1.0 / (self.temperature ** omega)

        out_w = grid_w.flatten().unsqueeze(-1) @ omega.unsqueeze(0)
        out_h = grid_h.flatten().unsqueeze(-1) @ omega.unsqueeze(0)

        return paddle.concat([out_h.sin(), out_h.cos(), out_w.sin(), out_w.cos()], axis=1).unsqueeze(0)


class PPDocLayoutV2AIFILayer(nn.Layer):
    """AIFI (Attention-based Intra-scale Feature Interaction) layer."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder_hidden_dim = config.encoder_hidden_dim
        self.eval_size = config.eval_size

        self.position_embedding = PPDocLayoutV2SinePositionEmbedding(
            embed_dim=self.encoder_hidden_dim,
            temperature=config.positional_encoding_temperature,
        )
        self.layers = nn.LayerList([PPDocLayoutV2EncoderLayer(config) for _ in range(config.encoder_layers)])

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


class PPDocLayoutV2HybridEncoder(nn.Layer):
    """
    Hybrid encoder: AIFI layers + top-down FPN + bottom-up PAN.
    """

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
        self.aifi = nn.LayerList([PPDocLayoutV2AIFILayer(config) for _ in range(len(self.encode_proj_layers))])

        # top-down FPN
        self.lateral_convs = nn.LayerList()
        self.fpn_blocks = nn.LayerList()
        for _ in range(self.num_fpn_stages):
            lateral_conv = PPDocLayoutV2ConvNormLayer(
                config,
                in_channels=self.encoder_hidden_dim,
                out_channels=self.encoder_hidden_dim,
                kernel_size=1,
                stride=1,
                activation=config.activation_function,
            )
            fpn_block = PPDocLayoutV2CSPRepLayer(config)
            self.lateral_convs.append(lateral_conv)
            self.fpn_blocks.append(fpn_block)

        # bottom-up PAN
        self.downsample_convs = nn.LayerList()
        self.pan_blocks = nn.LayerList()
        for _ in range(self.num_pan_stages):
            downsample_conv = PPDocLayoutV2ConvNormLayer(
                config,
                in_channels=self.encoder_hidden_dim,
                out_channels=self.encoder_hidden_dim,
                kernel_size=3,
                stride=2,
                activation=config.activation_function,
            )
            pan_block = PPDocLayoutV2CSPRepLayer(config)
            self.downsample_convs.append(downsample_conv)
            self.pan_blocks.append(pan_block)

    def forward(self, feature_maps):
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
            # [batch_size, H*W, num_heads, hidden_dim] -> [batch_size*num_heads, hidden_dim, H, W]
            value_l_ = (
                value_list[level_id]
                .flatten(2)
                .transpose([0, 2, 1])
                .reshape([batch_size * num_heads, hidden_dim, height, width])
            )
            # [batch_size, num_queries, num_heads, num_points, 2]
            # -> [batch_size*num_heads, num_queries, num_points, 2]
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


class PPDocLayoutV2MultiscaleDeformableAttention(nn.Layer):
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



class PPDocLayoutV2DecoderLayer(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.d_model

        self.self_attn = PPDocLayoutV2SelfAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_attention_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.dropout = config.dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.hidden_size, epsilon=config.layer_norm_eps)
        self.encoder_attn = PPDocLayoutV2MultiscaleDeformableAttention(
            config,
            num_heads=config.decoder_attention_heads,
            n_points=config.decoder_n_points,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.hidden_size, epsilon=config.layer_norm_eps)
        self.mlp = PPDocLayoutV2MLP(
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

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
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

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states


class PPDocLayoutV2Decoder(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.dropout = config.dropout
        self.layers = nn.LayerList([PPDocLayoutV2DecoderLayer(config) for _ in range(config.decoder_layers)])
        self.query_pos_head = PPDocLayoutV2MLPPredictionHead(4, 2 * config.d_model, config.d_model, num_layers=2)

        # Set by ForObjectDetection
        self.bbox_embed = None
        self.class_embed = None

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

        intermediate = []
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

            if self.bbox_embed is not None:
                predicted_corners = self.bbox_embed[idx](hidden_states)
                new_reference_points = F.sigmoid(predicted_corners + inverse_sigmoid(reference_points))
                reference_points = new_reference_points.detach()

            intermediate.append(hidden_states)
            intermediate_reference_points.append(
                new_reference_points if self.bbox_embed is not None else reference_points
            )

            if self.class_embed is not None:
                logits = self.class_embed[idx](hidden_states)
                intermediate_logits.append(logits)

        intermediate = paddle.stack(intermediate, axis=1)
        intermediate_reference_points = paddle.stack(intermediate_reference_points, axis=1)
        if self.class_embed is not None:
            intermediate_logits = paddle.stack(intermediate_logits, axis=1)

        return {
            "last_hidden_state": hidden_states,
            "intermediate_hidden_states": intermediate,
            "intermediate_logits": intermediate_logits,
            "intermediate_reference_points": intermediate_reference_points,
        }


# Backbone wrapper

def replace_batch_norm(model):
    """Recursively replace all nn.BatchNorm2D with PPDocLayoutV2FrozenBatchNorm2d."""
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2D):
            new_module = PPDocLayoutV2FrozenBatchNorm2d(module._num_features)
            new_module.weight.set_value(module.weight)
            new_module.bias.set_value(module.bias)
            new_module._mean.set_value(module._mean)
            new_module._variance.set_value(module._variance)
            setattr(model, name, new_module)

        if len(list(module.children())) > 0:
            replace_batch_norm(module)


class PPDocLayoutV2ConvEncoder(nn.Layer):
    """Convolutional backbone using HGNetV2Backbone."""

    def __init__(self, config):
        super().__init__()
        backbone = HGNetV2Backbone(config.backbone_config)

        if config.freeze_backbone_batch_norms:
            with paddle.no_grad():
                replace_batch_norm(backbone)
        self.model = backbone
        # Use encoder_in_channels from config (matches the selected backbone stages)
        self.intermediate_channel_sizes = config.encoder_in_channels

    def forward(self, pixel_values):
        features = self.model(pixel_values)
        # Select last N stages matching encoder_in_channels
        n = len(self.intermediate_channel_sizes)
        return features[-n:]



class PPDocLayoutV2Model(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Create backbone
        self.backbone = PPDocLayoutV2ConvEncoder(config)
        intermediate_channel_sizes = self.backbone.intermediate_channel_sizes

        # Create encoder input projection layers
        num_backbone_outs = len(intermediate_channel_sizes)
        encoder_input_proj_list = []
        for i in range(num_backbone_outs):
            in_channels = intermediate_channel_sizes[i]
            encoder_input_proj_list.append(
                nn.Sequential(
                    nn.Conv2D(in_channels, config.encoder_hidden_dim, kernel_size=1, bias_attr=False),
                    nn.BatchNorm2D(config.encoder_hidden_dim),
                )
            )
        self.encoder_input_proj = nn.LayerList(encoder_input_proj_list)

        # Create encoder
        self.encoder = PPDocLayoutV2HybridEncoder(config)

        # denoising embedding (always created for state dict compatibility)
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
        self.enc_bbox_head = PPDocLayoutV2MLPPredictionHead(config.d_model, config.d_model, 4, num_layers=3)

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
        self.decoder = PPDocLayoutV2Decoder(config)

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

        features = self.backbone(pixel_values)
        proj_feats = [self.encoder_input_proj[level](source) for level, source in enumerate(features)]

        encoder_outputs = self.encoder(proj_feats)

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

        enc_topk_bboxes = F.sigmoid(reference_points_unact)

        enc_topk_logits = paddle.take_along_axis(
            enc_outputs_class,
            topk_ind.unsqueeze(-1).expand([-1, -1, enc_outputs_class.shape[-1]]),
            axis=1,
        )

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
        )

        return decoder_outputs



class PPDocLayoutPostProcess:
    def __init__(
        self,
        num_classes=25,
        num_top_queries=100,
        dual_queries=False,
        dual_groups=0,
        use_focal_loss=False,
        with_mask=False,
        mask_stride=4,
        mask_threshold=0.5,
        use_avg_mask_score=False,
        bbox_decode_type="origin",
    ):
        self.num_classes = num_classes
        self.num_top_queries = num_top_queries
        self.dual_queries = dual_queries
        self.dual_groups = dual_groups
        self.use_focal_loss = use_focal_loss
        self.with_mask = with_mask
        self.mask_stride = mask_stride
        self.mask_threshold = mask_threshold
        self.use_avg_mask_score = use_avg_mask_score
        self.bbox_decode_type = bbox_decode_type

    def __call__(self, head_out, order_logits, im_shape, scale_factor, pad_shape):
        bboxes, logits, masks = head_out
        if self.dual_queries:
            num_queries = logits.shape[1]
            logits, bboxes = (
                logits[:, : int(num_queries // (self.dual_groups + 1)), :],
                bboxes[:, : int(num_queries // (self.dual_groups + 1)), :],
            )

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

        mask_pred = None
        if self.with_mask:
            assert masks is not None
            assert masks.shape[0] == 1
            masks = paddle.gather_nd(masks, index)
            if self.bbox_decode_type == "pad":
                masks = F.interpolate(
                    masks,
                    scale_factor=self.mask_stride,
                    mode="bilinear",
                    align_corners=False,
                )
                h, w = im_shape.astype("int32")[0]
                masks = masks[..., :h, :w]
            img_h = img_h[0].astype("int32")
            img_w = img_w[0].astype("int32")
            masks = F.interpolate(
                masks, size=[img_h, img_w], mode="bilinear", align_corners=False
            )
            mask_pred, scores = self._mask_postprocess(masks, scores)

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
        return bbox_pred, bbox_num, mask_pred



class PPDocLayoutV2(BatchNormHFStateDictMixin, PretrainedModel):

    config_class = PPDocLayoutV2Config

    def __init__(self, config):
        super(PPDocLayoutV2, self).__init__(config)
        self.config = config

        # Build the detection model (PPDocLayoutV2ForObjectDetection equivalent)
        self.model = PPDocLayoutV2Model(config)

        # Set up decoder bbox_embed and class_embed (from ForObjectDetection)
        num_pred = config.decoder_layers
        self.model.decoder.class_embed = nn.LayerList(
            [nn.Linear(config.d_model, config.num_labels) for _ in range(num_pred)]
        )
        self.model.decoder.bbox_embed = nn.LayerList(
            [PPDocLayoutV2MLPPredictionHead(config.d_model, config.d_model, 4, num_layers=3) for _ in range(num_pred)]
        )

        # Reading order model
        self.reading_order = PPDocLayoutV2ReadingOrder(config.reading_order_config)
        self.num_queries = config.num_queries

        # Post-process (used in the old-style forward path)
        self.post_process = PPDocLayoutPostProcess(
            num_top_queries=config.num_queries,
            use_focal_loss=True,
        )

    def forward(self, inputs):
        pixel_values = paddle.to_tensor(inputs[1])
        im_shape = paddle.to_tensor(inputs[0])
        scale_factor = paddle.to_tensor(inputs[2])

        # Run the detection model (backbone -> encoder -> decoder)
        decoder_outputs = self.model(pixel_values)

        intermediate_reference_points = decoder_outputs["intermediate_reference_points"]
        intermediate_logits = decoder_outputs["intermediate_logits"]

        # Take last layer outputs
        raw_bboxes = intermediate_reference_points[:, -1]
        logits = intermediate_logits[:, -1]

        # Convert center-format boxes to [x1,y1,x2,y2] in [0,1000] scale for reading order
        box_centers, box_sizes = raw_bboxes.split(2, axis=-1)
        bboxes = paddle.concat([box_centers - 0.5 * box_sizes, box_centers + 0.5 * box_sizes], axis=-1) * 1000
        bboxes = bboxes.clip(0.0, 1000.0)

        max_logits = logits.max(axis=-1)
        class_ids = logits.argmax(axis=-1)
        max_probs = F.sigmoid(max_logits)

        class_thresholds = paddle.to_tensor(self.config.class_thresholds, dtype="float32")
        thresholds = paddle.index_select(class_thresholds, class_ids.flatten(), axis=0).reshape(class_ids.shape)
        mask = max_probs >= thresholds

        indices = paddle.argsort(mask.astype("int32"), axis=1, descending=True)

        sorted_class_ids = paddle.take_along_axis(class_ids, indices, axis=1)
        sorted_boxes = paddle.take_along_axis(bboxes, indices.unsqueeze(-1).expand([-1, -1, 4]), axis=1)
        pred_boxes = paddle.take_along_axis(raw_bboxes, indices.unsqueeze(-1).expand([-1, -1, 4]), axis=1)
        logits_sorted = paddle.take_along_axis(
            logits, indices.unsqueeze(-1).expand([-1, -1, logits.shape[-1]]), axis=1
        )

        sorted_mask = paddle.take_along_axis(mask.astype("int32"), indices, axis=1).astype("bool")

        pad_boxes = paddle.where(sorted_mask.unsqueeze(-1), sorted_boxes, paddle.zeros_like(sorted_boxes))
        pad_class_ids = paddle.where(sorted_mask, sorted_class_ids, paddle.zeros_like(sorted_class_ids))

        class_order = paddle.to_tensor(self.config.class_order, dtype="int32")
        pad_class_ids = paddle.index_select(class_order, pad_class_ids.flatten(), axis=0).reshape(pad_class_ids.shape)

        order_logits = self.reading_order(
            boxes=pad_boxes,
            labels=pad_class_ids,
            mask=sorted_mask,
        )
        order_logits = order_logits[:, :, :self.num_queries]

        # Use the post-process for final output (sorted to match order_logits)
        head_out = (pred_boxes, logits_sorted, None)
        # Compute pad_shape from pixel_values shape
        pad_shape = paddle.to_tensor(
            [[pixel_values.shape[2], pixel_values.shape[3]]] * pixel_values.shape[0],
            dtype="float32",
        )
        bbox, bbox_num, mask_pred = self.post_process(
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
            "q_proj",
            "k_proj",
            "v_proj",
            "linear_1",
            "linear_2",
            "enc_bbox_head",
            "enc_output",
            "spatial_proj",
            "query",
            "key",
            "value",
            "intermediate",
            "attention",
            "output",
            "relative_head",
            "query_pos_head",
            "enc_score_head",
            "in_proj_weight",
            "linear1",
            "linear2",
            "label_features_projection",
            "reading_order.encoder.layer",
            "encoder_attn",
            "decoder.bbox_embed",
            "decoder.class_embed",
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
        return super().set_hf_state_dict(
            _apply_rt_detr_key_conversion(state_dict), *args, **kwargs
        )

    def get_hf_state_dict(self, *args, **kwargs):
        return _reverse_rt_detr_key_conversion(
            super().get_hf_state_dict(*args, **kwargs)
        )
