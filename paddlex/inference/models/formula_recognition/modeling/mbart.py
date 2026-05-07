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

"""Inference-only MBart decoder used by PP-FormulaNet, aligned with HF transformers.

State-dict layout matches ``transformers.models.pp_formulanet`` exactly so HF
safetensors load directly. KV cache uses a per-layer dict; cross-attention KV
is computed once on first decode step and reused.
"""

import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ...common.transformers.activations import ACT2FN

__all__ = ["MBartDecoder"]


class MBartLearnedPositionalEmbedding(nn.Embedding):
    """Learned positional embedding with a +2 offset on input position ids.

    Mirrors ``transformers.models.pp_formulanet.PPFormulaNetLearnedPositionalEmbedding``.
    The HF implementation reserves the first two embedding rows for special
    purposes (compat with Bart family); this offset is part of the saved
    weight shape (``num_embeddings + 2``), so we must preserve it on lookup.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, position_ids: paddle.Tensor) -> paddle.Tensor:
        return super().forward(position_ids + self.offset)


class MBartScaledWordEmbedding(nn.Embedding):
    """Token embedding scaled by ``sqrt(d_model)`` when scale_embedding=True."""

    def __init__(self, num_embeddings, embedding_dim, padding_idx, embed_scale=1.0):
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.embed_scale = embed_scale

    def forward(self, input_ids: paddle.Tensor) -> paddle.Tensor:
        return super().forward(input_ids) * self.embed_scale


class MBartAttention(nn.Layer):
    """Multi-headed attention with separate q/k/v/out projections.

    Mirrors ``transformers.models.pp_formulanet.PPFormulaNetAttention``. Used
    for both self- and cross-attention; cross-attention is signalled by passing
    ``key_value_states``.
    """

    def __init__(self, embed_dim: int, num_heads: int, bias: bool = True):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}"
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias_attr=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias_attr=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias_attr=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias_attr=bias)

    def _shape(self, x: paddle.Tensor) -> paddle.Tensor:
        # [B, T, C] -> [B, num_heads, T, head_dim]
        new_shape = x.shape[:-1] + [self.num_heads, self.head_dim]
        return x.reshape(new_shape).transpose([0, 2, 1, 3])

    def forward(
        self,
        hidden_states: paddle.Tensor,
        key_value_states: paddle.Tensor = None,
        attention_mask: paddle.Tensor = None,
        cache: dict = None,
    ) -> paddle.Tensor:
        is_cross_attention = key_value_states is not None
        query_states = self._shape(self.q_proj(hidden_states))

        if is_cross_attention and cache is not None and cache.get("cross_k") is not None:
            # Cross-attention KV is independent of decoded tokens; reuse from cache.
            key_states = cache["cross_k"]
            value_states = cache["cross_v"]
        else:
            current_states = key_value_states if is_cross_attention else hidden_states
            key_states = self._shape(self.k_proj(current_states))
            value_states = self._shape(self.v_proj(current_states))

            if cache is not None:
                if is_cross_attention:
                    cache["cross_k"] = key_states
                    cache["cross_v"] = value_states
                else:
                    if cache.get("self_k") is not None:
                        key_states = paddle.concat([cache["self_k"], key_states], axis=2)
                        value_states = paddle.concat(
                            [cache["self_v"], value_states], axis=2
                        )
                    cache["self_k"] = key_states
                    cache["self_v"] = value_states

        attn_weights = paddle.matmul(query_states, key_states, transpose_y=True)
        attn_weights = attn_weights * self.scaling

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights.astype("float32"), axis=-1).astype(
            query_states.dtype
        )
        attn_output = paddle.matmul(attn_weights, value_states)
        # [B, num_heads, T, head_dim] -> [B, T, embed_dim]
        attn_output = attn_output.transpose([0, 2, 1, 3]).reshape(
            [hidden_states.shape[0], hidden_states.shape[1], self.embed_dim]
        )
        attn_output = self.out_proj(attn_output)
        return attn_output


class MBartDecoderLayer(nn.Layer):
    """Pre-LN decoder block: self-attn → cross-attn → FFN."""

    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = MBartAttention(self.embed_dim, config.decoder_attention_heads)
        self.activation_fn = ACT2FN[config.activation_function]
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.encoder_attn = MBartAttention(self.embed_dim, config.decoder_attention_heads)
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: paddle.Tensor = None,
        encoder_hidden_states: paddle.Tensor = None,
        encoder_attention_mask: paddle.Tensor = None,
        cache: dict = None,
    ) -> paddle.Tensor:
        # Self attention
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            cache=cache,
        )
        hidden_states = residual + hidden_states

        # Cross attention
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)
            hidden_states = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                cache=cache,
            )
            hidden_states = residual + hidden_states

        # FFN
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


def _make_causal_mask(seq_len: int, past_len: int, dtype) -> paddle.Tensor:
    """Lower-triangular additive mask for self-attention with KV cache.

    Shape: [1, 1, seq_len, seq_len + past_len]. Future positions get
    ``-inf`` in fp32 (clamped to dtype min on cast).
    """
    total_len = past_len + seq_len
    mask = paddle.zeros([seq_len, total_len], dtype="float32")
    if seq_len > 1:
        causal = paddle.tril(paddle.ones([seq_len, seq_len], dtype="float32"))
        mask[:, past_len:] = (1.0 - causal) * paddle.finfo(paddle.float32).min
    return mask.unsqueeze(0).unsqueeze(0).astype(dtype)


class MBartDecoder(nn.Layer):
    """MBart decoder stack matching HF transformers state-dict layout.

    Hierarchy (saved keys):
        embed_tokens.weight
        embed_positions.weight       (size = max_position_embeddings + 2)
        layernorm_embedding.{weight,bias}
        layers.{i}.self_attn.{q,k,v,out}_proj.{weight,bias}
        layers.{i}.self_attn_layer_norm.{weight,bias}
        layers.{i}.encoder_attn.{q,k,v,out}_proj.{weight,bias}
        layers.{i}.encoder_attn_layer_norm.{weight,bias}
        layers.{i}.fc1.{weight,bias}
        layers.{i}.fc2.{weight,bias}
        layers.{i}.final_layer_norm.{weight,bias}
        layer_norm.{weight,bias}
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        self.embed_tokens = MBartScaledWordEmbedding(
            config.vocab_size, config.d_model, self.padding_idx, embed_scale=embed_scale
        )
        self.embed_positions = MBartLearnedPositionalEmbedding(
            config.max_position_embeddings, config.d_model
        )
        self.layers = nn.LayerList(
            [MBartDecoderLayer(config) for _ in range(config.decoder_layers)]
        )
        self.layernorm_embedding = nn.LayerNorm(config.d_model)
        self.layer_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        input_ids: paddle.Tensor,
        encoder_hidden_states: paddle.Tensor,
        cache: list = None,
        past_length: int = 0,
    ) -> tuple:
        """Forward one step (or one prefill block) through the decoder.

        Args:
            input_ids: [B, T] new tokens to decode.
            encoder_hidden_states: [B, S, d_model] projected vision features.
            cache: list of per-layer dicts for KV cache. If None, no caching.
            past_length: number of tokens already in cache (offset for position ids).
        """
        bsz, seq_len = input_ids.shape
        inputs_embeds = self.embed_tokens(input_ids)
        position_ids = paddle.arange(past_length, past_length + seq_len, dtype="int64")
        position_ids = position_ids.unsqueeze(0).expand([bsz, seq_len])
        positions = self.embed_positions(position_ids)

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        attention_mask = _make_causal_mask(seq_len, past_length, hidden_states.dtype)

        for idx, layer in enumerate(self.layers):
            layer_cache = cache[idx] if cache is not None else None
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                cache=layer_cache,
            )

        hidden_states = self.layer_norm(hidden_states)
        return hidden_states
