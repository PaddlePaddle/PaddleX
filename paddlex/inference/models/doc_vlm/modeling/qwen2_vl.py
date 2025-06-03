# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import paddle
import paddle.distributed.fleet.meta_parallel as mpu
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import Tensor
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker
from paddle.distributed.fleet.utils import recompute

from .....utils import logging
from ....utils.benchmark import (
    benchmark,
    get_inference_operations,
    set_inference_operations,
)
from ...common.vlm.activations import ACT2FN
from ...common.vlm.bert_padding import index_first_axis, pad_input, unpad_input
from ...common.vlm.flash_attn_utils import has_flash_attn_func
from ...common.vlm.transformers import PretrainedConfig, PretrainedModel
from ...common.vlm.transformers.model_outputs import (
    BaseModelOutputWithPast,
    ModelOutput,
)

flash_attn_func, flash_attn_varlen_func = has_flash_attn_func()
_IS_NPU = "npu" in paddle.get_device()

Linear = nn.Linear
ColumnParallelLinear = mpu.ColumnParallelLinear
RowParallelLinear = mpu.RowParallelLinear


class Qwen2VLVisionConfig(PretrainedConfig):
    model_type = "qwen2_vl"

    def __init__(
        self,
        depth=32,
        embed_dim=1280,
        hidden_size=3584,
        hidden_act="quick_gelu",
        mlp_ratio=4,
        num_heads=16,
        in_channels=3,
        patch_size=14,
        spatial_merge_size=2,
        temporal_patch_size=2,
        attn_implementation="eager",  # new added
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.depth = depth
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.attn_implementation = attn_implementation

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> "PretrainedConfig":

        config_dict, kwargs = cls.get_config_dict(
            pretrained_model_name_or_path, **kwargs
        )

        if config_dict.get("model_type") == "qwen2_vl":
            config_dict = config_dict["vision_config"]

        if (
            "model_type" in config_dict
            and hasattr(cls, "model_type")
            and config_dict["model_type"] != cls.model_type
        ):
            logging.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class Qwen2VLConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen2VLModel`]. It is used to instantiate a
    Qwen2-VL model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of
    Qwen2-VL-7B-Instruct [Qwen/Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 152064):
            Vocabulary size of the Qwen2VL model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Qwen2VLModel`]
        hidden_size (`int`, *optional*, defaults to 8192):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 29568):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 80):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 64):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to `32`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        rope_theta (`float`, *optional*, defaults to 1000000.0):
            The base period of the RoPE embeddings.
        use_sliding_window (`bool`, *optional*, defaults to `False`):
            Whether to use sliding window attention.
        sliding_window (`int`, *optional*, defaults to 4096):
            Sliding window attention (SWA) window size. If not specified, will default to `4096`.
        max_window_layers (`int`, *optional*, defaults to 80):
            The number of layers that use SWA (Sliding Window Attention). The bottom layers use SWA while the top use full attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        vision_config (`Dict`, *optional*):
            The config for the visual encoder initialization.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
            strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
            `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
            `max_position_embeddings` to the expected new maximum. See the following thread for more information on how
            these scaling strategies behave:
            https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/. This is an
            experimental feature, subject to breaking API changes in future versions.
    """

    model_type = "qwen2_vl"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=152064,
        hidden_size=8192,
        intermediate_size=29568,
        num_hidden_layers=80,
        num_attention_heads=64,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-05,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=1000000.0,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=80,
        attention_dropout=0.0,
        vision_config=None,
        rope_scaling=None,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            self.vision_config = Qwen2VLVisionConfig(**vision_config)
        elif vision_config is None:
            self.vision_config = Qwen2VLVisionConfig()

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.rope_scaling = rope_scaling

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


def get_triangle_upper_mask(x, mask=None):
    if mask is not None:
        return mask
    shape = x.shape
    shape[1] = 1
    mask = paddle.full(shape, paddle.finfo(x.dtype).min, dtype=x.dtype)
    mask = paddle.triu(mask, diagonal=1)
    mask.stop_gradient = True
    return mask


def parallel_matmul(
    x: Tensor, y: Tensor, transpose_y=True, tensor_parallel_output=True
):
    is_fleet_init = True
    tensor_parallel_degree = 1
    try:
        hcg = fleet.get_hybrid_communicate_group()
        model_parallel_group = hcg.get_model_parallel_group()
        tensor_parallel_degree = hcg.get_model_parallel_world_size()
    except:
        is_fleet_init = False

    if paddle.in_dynamic_mode():
        y_is_distributed = y.is_distributed
    else:
        y_is_distributed = tensor_parallel_degree > 1

    if is_fleet_init and tensor_parallel_degree > 1 and y_is_distributed:

        input_parallel = paddle.distributed.collective._c_identity(
            x, group=model_parallel_group
        )
        logits = paddle.matmul(input_parallel, y, transpose_y=transpose_y)

        if tensor_parallel_output:
            return logits
        return paddle.distributed.collective._c_concat(
            logits, group=model_parallel_group
        )

    else:
        logits = paddle.matmul(x, y, transpose_y=transpose_y)
        return logits


def _compute_default_rope_parameters(
    config: Optional[PretrainedConfig] = None,
    device: Optional["paddle.device"] = None,
    seq_len: Optional[int] = None,
    **rope_kwargs,
) -> Tuple["paddle.Tensor", float]:
    """
    Computes the inverse frequencies according to the original RoPE implementation
    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration.
        device (`paddle.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.
        rope_kwargs (`Dict`, *optional*):
            BC compatibility with the previous RoPE class instantiation, will be removed in v4.45.
    Returns:
        Tuple of (`paddle.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
    """
    if config is not None and len(rope_kwargs) > 0:
        raise ValueError(
            "Unexpected arguments: `**rope_kwargs` and `config` are mutually exclusive in "
            f"`_compute_default_rope_parameters`, got `rope_kwargs`={rope_kwargs} and `config`={config}"
        )
    if len(rope_kwargs) > 0:
        base = rope_kwargs["base"]
        dim = rope_kwargs["dim"]
    elif config is not None:
        base = config.rope_theta
        partial_rotary_factor = (
            config.partial_rotary_factor
            if hasattr(config, "partial_rotary_factor")
            else 1.0
        )
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        dim = int(head_dim * partial_rotary_factor)

    attention_factor = 1.0

    inv_freq = 1.0 / (
        base ** (paddle.arange(0, dim, 2, dtype="int64").astype("float32") / dim)
    )
    return inv_freq, attention_factor


ROPE_INIT_FUNCTIONS = {
    "default": _compute_default_rope_parameters,
}


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(axis=-1, dtype="int32")
    indices = paddle.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()  # [2, 1, 1323]
    cu_seqlens = F.pad(
        paddle.cumsum(seqlens_in_batch, axis=0), (1, 0), data_format="NCL"
    )
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def is_casual_mask(attention_mask):
    """
    Upper triangular of attention_mask equals to attention_mask is casual
    """
    return (paddle.triu(attention_mask) == attention_mask).all().item()


def _make_causal_mask(input_ids_shape, past_key_values_length):
    """
    Make causal mask used for self-attention
    """
    batch_size, target_length = input_ids_shape

    mask = paddle.tril(paddle.ones((target_length, target_length), dtype="bool"))

    if past_key_values_length > 0:
        mask = paddle.concat(
            [paddle.ones([target_length, past_key_values_length], dtype="bool"), mask],
            axis=-1,
        )

    return mask[None, None, :, :].expand(
        [batch_size, 1, target_length, target_length + past_key_values_length]
    )


def _expand_2d_mask(mask, dtype, tgt_length):
    """
    Expands attention_mask from `[batch_size, src_length]` to `[batch_size, 1, tgt_length, src_length]`.
    """
    batch_size, src_length = mask.shape[0], mask.shape[-1]
    tgt_length = tgt_length if tgt_length is not None else src_length

    mask = mask[:, None, None, :].astype("bool")
    mask.stop_gradient = True
    expanded_mask = mask.expand([batch_size, 1, tgt_length, src_length])

    return expanded_mask


@dataclass
class Qwen2VLCausalLMOutputWithPast(ModelOutput):
    """
    Base class for Qwen2VL causal language model (or autoregressive) outputs.

    Args:
        loss (`paddle.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`paddle.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(paddle.Tensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(paddle.Tensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(paddle.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `paddle.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(paddle.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `paddle.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        rope_deltas (`paddle.Tensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
    """

    loss: Optional[paddle.Tensor] = None
    logits: paddle.Tensor = None
    past_key_values: Optional[List[paddle.Tensor]] = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None
    attentions: Optional[Tuple[paddle.Tensor]] = None
    rope_deltas: Optional[paddle.Tensor] = None


class Qwen2VLRotaryEmbedding(nn.Layer):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        config: Optional[Qwen2VLConfig] = None,
    ):
        super().__init__()
        self.rope_kwargs = {}
        if config is None:
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:
            # BC: "rope_type" was originally "type"
            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get(
                    "rope_type", config.rope_scaling.get("type")
                )
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        self.inv_freq, self.attention_scaling = self.rope_init_fn(
            self.config, device, **self.rope_kwargs
        )
        self.original_inv_freq = self.inv_freq

        self._set_cos_sin_cache(seq_len=max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        t = paddle.arange(seq_len, dtype="float32")
        freqs = paddle.einsum("i,j->ij", t, self.inv_freq)
        emb = paddle.concat([freqs, freqs], axis=-1)
        self.cos_cached = emb.cos()
        self.sin_cached = emb.sin()

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = paddle.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.inv_freq = inv_freq
            self.max_seq_len_cached = seq_len

        if (
            seq_len < self.original_max_seq_len
            and self.max_seq_len_cached > self.original_max_seq_len
        ):  # reset
            self.inv_freq = self.original_inv_freq
            self.max_seq_len_cached = self.original_max_seq_len

    @paddle.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        inv_freq_expanded = (
            self.inv_freq[None, None, :, None]
            .astype("float32")
            .expand([3, position_ids.shape[1], -1, 1])
        )
        position_ids_expanded = position_ids[:, :, None, :].astype("float32")
        device_type = paddle.get_device()
        device_type = (
            device_type
            if isinstance(device_type, str) and device_type != "mps"
            else "cpu"
        )
        with paddle.amp.auto_cast():
            freqs = paddle.matmul(inv_freq_expanded, position_ids_expanded)
            freqs = freqs.transpose([0, 1, 3, 2])
            emb = paddle.concat((freqs, freqs), axis=-1)
            cos = emb.cos()
            sin = emb.sin()

        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.astype(x.dtype), sin.astype(x.dtype)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.concat([-x2, x1], axis=-1)


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors (https://qwenlm.github.io/blog/qwen2-vl/).

    Explanation:
        Multimodal 3D rotary position embedding is an extension to 1D rotary position embedding. The input embedding
        sequence contains vision (images / videos) embedding and text embedding or just contains text embedding. For
        vision embedding part, we apply rotary position embedding on temporal, height and width dimension separately.
        Here we split the channel dimension to 3 chunks for the temporal, height and width rotary position embedding.
        For text embedding part, we just apply 1D rotary position embedding. The three rotary position index (temporal,
        height and width) of text embedding is always the same, so the text embedding rotary position embedding has no
        difference with modern LLMs.

    Args:
        q (`paddle.Tensor`): The query tensor.
        k (`paddle.Tensor`): The key tensor.
        cos (`paddle.Tensor`): The cosine part of the rotary embedding.
        sin (`paddle.Tensor`): The sine part of the rotary embedding.
        position_ids (`paddle.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        mrope_section(`List(int)`):
            Multimodal rope section is for channel dimension of temporal, height and width in rope calculation.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(paddle.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """

    mrope_section = mrope_section * 2
    cos = paddle.concat(
        x=[m[i % 3] for i, m in enumerate(cos.split(mrope_section, axis=-1))], axis=-1
    ).unsqueeze(axis=unsqueeze_dim)
    sin = paddle.concat(
        x=[m[i % 3] for i, m in enumerate(sin.split(mrope_section, axis=-1))], axis=-1
    ).unsqueeze(axis=unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb_vision(
    tensor: paddle.Tensor, freqs: paddle.Tensor
) -> paddle.Tensor:
    orig_dtype = tensor.dtype

    with paddle.amp.auto_cast(False):
        tensor = tensor.astype(dtype="float32")
        cos = freqs.cos()
        sin = freqs.sin()
        cos = (
            cos.unsqueeze(1)
            .tile(repeat_times=[1, 1, 2])
            .unsqueeze(0)
            .astype(dtype="float32")
        )
        sin = (
            sin.unsqueeze(1)
            .tile(repeat_times=[1, 1, 2])
            .unsqueeze(0)
            .astype(dtype="float32")
        )
        output = tensor * cos + rotate_half(tensor) * sin
    output = paddle.cast(output, orig_dtype)
    return output


class VisionRotaryEmbedding(nn.Layer):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.inv_freq = 1.0 / theta ** (
            paddle.arange(start=0, end=dim, step=2, dtype="float32") / dim
        )

    def forward(self, seqlen: int) -> paddle.Tensor:
        seq = paddle.arange(seqlen).cast(self.inv_freq.dtype)
        freqs = paddle.outer(x=seq, y=self.inv_freq)
        return freqs


class PatchEmbed(nn.Layer):
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3D(
            in_channels,
            embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias_attr=False,
        )

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:

        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.reshape(
            [
                -1,
                self.in_channels,
                self.temporal_patch_size,
                self.patch_size,
                self.patch_size,
            ]
        )
        # NOTE（changwenbin）: AttributeError: 'Variable' object has no attribute 'to'.
        # hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).reshape([-1, self.embed_dim])
        # hidden_states = paddle.cast(hidden_states, dtype=target_dtype)
        hidden_states = self.proj(
            paddle.cast(hidden_states, dtype=target_dtype)
        ).reshape([-1, self.embed_dim])
        return hidden_states


class PatchMerger(nn.Layer):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = nn.LayerNorm(context_dim, epsilon=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.mlp(self.ln_q(x).reshape([-1, self.hidden_size]))
        return x


class VisionMlp(nn.Layer):
    def __init__(self, dim: int, hidden_dim: int, hidden_act: str) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = ACT2FN[hidden_act]
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x) -> paddle.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class VisionAttention(nn.Layer):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias_attr=True)
        self.proj = nn.Linear(dim, dim)
        self.head_dim = dim // num_heads  # must added

    def forward(
        self,
        hidden_states: paddle.Tensor,
        cu_seqlens: paddle.Tensor,
        rotary_pos_emb: paddle.Tensor = None,
    ) -> paddle.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = (
            self.qkv(hidden_states)
            .reshape([seq_length, 3, self.num_heads, -1])
            .transpose([1, 0, 2, 3])
            .unbind(0)
        )
        q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
        k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

        attention_mask = paddle.zeros([1, seq_length, seq_length], dtype="bool")
        for i in range(1, len(cu_seqlens)):
            attention_mask[
                ...,
                cu_seqlens[i - 1] : cu_seqlens[i],
                cu_seqlens[i - 1] : cu_seqlens[i],
            ] = True

        zero = paddle.zeros(attention_mask.shape, dtype=hidden_states.dtype)
        neg_inf = paddle.full_like(
            attention_mask,
            paddle.finfo(hidden_states.dtype).min,
            dtype=hidden_states.dtype,
        )
        attention_mask = paddle.where(attention_mask, zero, neg_inf)

        q = q.transpose([1, 0, 2])
        k = k.transpose([1, 0, 2])
        v = v.transpose([1, 0, 2])
        attn_weights = paddle.matmul(q, k.transpose([0, 2, 1])) / math.sqrt(
            self.head_dim
        )
        attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, axis=-1, dtype="float32")
        attn_output = paddle.matmul(attn_weights, v)
        attn_output = attn_output.transpose([1, 0, 2])
        attn_output = attn_output.reshape([seq_length, -1])
        attn_output = self.proj(attn_output)
        return attn_output


class VisionFlashAttention2(nn.Layer):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias_attr=True)
        self.proj = nn.Linear(dim, dim)
        self.head_dim = dim // num_heads  # must added

    def forward(
        self,
        hidden_states: paddle.Tensor,
        cu_seqlens: paddle.Tensor,
        rotary_pos_emb: paddle.Tensor = None,
    ) -> paddle.Tensor:
        seq_length = tuple(hidden_states.shape)[0]
        qkv = (
            self.qkv(hidden_states)
            .reshape([seq_length, 3, self.num_heads, -1])
            .transpose(perm=[1, 0, 2, 3])
        )
        q, k, v = qkv.unbind(axis=0)
        q = apply_rotary_pos_emb_vision(q.unsqueeze(axis=0), rotary_pos_emb).squeeze(
            axis=0
        )
        k = apply_rotary_pos_emb_vision(k.unsqueeze(axis=0), rotary_pos_emb).squeeze(
            axis=0
        )

        if _IS_NPU:
            attn_output = paddle.nn.functional.flash_attention_npu(
                q.astype("bfloat16"),
                k.astype("bfloat16"),
                v.astype("bfloat16"),
                is_varlen=True,
                batch_size=1,
                seq_length=seq_length,
            ).reshape([seq_length, -1])
        else:
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

            softmax_scale = self.head_dim**-0.5
            attn_output = (
                flash_attn_varlen_func(
                    q.astype("bfloat16"),
                    k.astype("bfloat16"),
                    v.astype("bfloat16"),
                    cu_seqlens,
                    cu_seqlens,
                    max_seqlen,
                    max_seqlen,
                    scale=softmax_scale,
                )[0]
                .squeeze(0)
                .reshape([seq_length, -1])
            )
        if self.proj.weight.dtype == paddle.bfloat16:
            attn_output = attn_output.astype(paddle.bfloat16)
        elif self.proj.weight.dtype == paddle.float16:
            attn_output = attn_output.astype(paddle.float16)
        elif self.proj.weight.dtype == paddle.float32:
            attn_output = attn_output.astype(paddle.float32)
        attn_output = self.proj(attn_output)
        return attn_output


def create_attention_module(config, module_type, layer_idx=None):
    if flash_attn_func is not None:
        if module_type == "qwen2vl":
            return Qwen2VLFlashAttention2(config, layer_idx)
        elif module_type == "vision":
            return VisionFlashAttention2(config.embed_dim, num_heads=config.num_heads)
    else:
        logging.warning_once(
            f"Warning: Flash Attention2 is not available for {module_type}, fallback to normal attention."
        )

    if module_type == "qwen2vl":
        return Qwen2VLAttention(config, layer_idx)
    elif module_type == "vision":
        return VisionAttention(config.embed_dim, num_heads=config.num_heads)


class Qwen2VLVisionBlock(nn.Layer):
    def __init__(self, config, attn_implementation: str = "flash_attention_2") -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim, epsilon=1e-6)
        self.norm2 = nn.LayerNorm(config.embed_dim, epsilon=1e-6)
        mlp_hidden_dim = int(config.embed_dim * config.mlp_ratio)

        self.attn = create_attention_module(config, "vision")
        self.mlp = VisionMlp(
            dim=config.embed_dim,
            hidden_dim=mlp_hidden_dim,
            hidden_act=config.hidden_act,
        )

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb) -> paddle.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


def _prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: paddle.Tensor,
    sequence_length: int,
    target_length: int,
    dtype: paddle.dtype,
    min_dtype: float,
    cache_position: paddle.Tensor,
    batch_size: int,
):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

    Args:
        attention_mask (`paddle.Tensor`):
            A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
        sequence_length (`int`):
            The sequence length being processed.
        target_length (`int`):
            The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
        dtype (`paddle.dtype`):
            The dtype to use for the 4D attention mask.
        min_dtype (`float`):
            The minimum value representable with the dtype `dtype`.
        cache_position (`paddle.Tensor`):
            Indices depicting the position of the input sequence tokens in the sequence.
        batch_size (`paddle.Tensor`):
            Batch size.
    """
    if attention_mask is not None and attention_mask.dim() == 4:
        causal_mask = attention_mask
    else:
        causal_mask = paddle.full(
            [sequence_length, target_length], fill_value=min_dtype, dtype=dtype
        )
        if sequence_length != 1:
            causal_mask = paddle.triu(x=causal_mask, diagonal=1)
        causal_mask *= paddle.arange(target_length) > cache_position.reshape([-1, 1])
        causal_mask = causal_mask[None, None, :, :].expand(
            shape=[batch_size, 1, -1, -1]
        )
        if attention_mask is not None:
            causal_mask = causal_mask.clone()
            mask_length = tuple(attention_mask.shape)[-1]
            padding_mask = (
                causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            )
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[
                :, :, :, :mask_length
            ].masked_fill(mask=padding_mask, value=min_dtype)

    return causal_mask


class Qwen2RMSNorm(nn.Layer):
    def __init__(self, config: Qwen2VLConfig, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = paddle.create_parameter(
            shape=[hidden_size],
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        if paddle.in_dynamic_mode():
            with paddle.amp.auto_cast(False):
                variance = hidden_states.astype("float32").pow(2).mean(-1, keepdim=True)
                hidden_states = (
                    paddle.rsqrt(variance + self.variance_epsilon) * hidden_states
                )
        else:
            variance = hidden_states.astype("float32").pow(2).mean(-1, keepdim=True)
            hidden_states = (
                paddle.rsqrt(variance + self.variance_epsilon) * hidden_states
            )

        if self.weight.dtype in [paddle.float16, paddle.bfloat16]:
            hidden_states = paddle.cast(hidden_states, self.weight.dtype)
        return hidden_states * self.weight


class Qwen2MLP(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.fuse_attention_ffn = config.fuse_attention_ffn
        self.tensor_parallel_degree = config.tensor_parallel_degree

        if config.tensor_parallel_degree > 1:

            self.gate_proj = ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                gather_output=False,
                has_bias=False,
            )
            self.up_proj = ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                gather_output=False,
                has_bias=False,
            )
            self.down_proj = RowParallelLinear(
                self.intermediate_size,
                self.hidden_size,
                input_is_parallel=True,
                has_bias=False,
            )
        else:
            self.gate_proj = Linear(
                self.hidden_size, self.intermediate_size, bias_attr=False
            )  # w1
            self.up_proj = Linear(
                self.hidden_size, self.intermediate_size, bias_attr=False
            )  # w3
            self.down_proj = Linear(
                self.intermediate_size, self.hidden_size, bias_attr=False
            )  # w2

        self.act_fn = ACT2FN[config.hidden_act]
        self.fuse_swiglu = False

    def forward(self, x):
        x, y = self.gate_proj(x), self.up_proj(x)
        if self.fuse_swiglu:
            x = self.act_fn(x, y)
        else:
            x = self.act_fn(x) * y

        return self.down_proj(x)


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: paddle.Tensor, n_rep: int) -> paddle.Tensor:
    """
    This is the equivalent of paddle.repeat_interleave(x, axis=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        [batch, num_key_value_heads, n_rep, slen, head_dim]
    )
    return hidden_states.reshape([batch, num_key_value_heads * n_rep, slen, head_dim])


class Qwen2VLAttention(nn.Layer):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: Qwen2VLConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logging.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        self.rope_scaling = config.rope_scaling
        # self.sequence_parallel = config.sequence_parallel

        if config.tensor_parallel_degree > 1:
            assert (
                self.num_heads % config.tensor_parallel_degree == 0
            ), f"num_heads: {self.num_heads}, tensor_parallel_degree: {config.tensor_parallel_degree}"
            self.num_heads = self.num_heads // config.tensor_parallel_degree

            assert (
                self.num_key_value_heads % config.tensor_parallel_degree == 0
            ), f"num_key_value_heads: {self.num_key_value_heads}, tensor_parallel_degree: {config.tensor_parallel_degree}"
            self.num_key_value_heads = (
                self.num_key_value_heads // config.tensor_parallel_degree
            )

        if config.tensor_parallel_degree > 1:
            self.q_proj = ColumnParallelLinear(
                self.hidden_size, self.hidden_size, has_bias=True, gather_output=False
            )
            self.k_proj = ColumnParallelLinear(self.hidden_size, self.config.num_key_value_heads * self.head_dim, has_bias=True, gather_output=False)  # fmt:skip
            self.v_proj = ColumnParallelLinear(self.hidden_size, self.config.num_key_value_heads * self.head_dim, has_bias=True, gather_output=False)  # fmt:skip
            self.o_proj = RowParallelLinear(
                self.hidden_size,
                self.hidden_size,
                has_bias=False,
                input_is_parallel=True,
            )
        else:
            self.q_proj = Linear(self.hidden_size, self.hidden_size, bias_attr=True)
            self.k_proj = Linear(
                self.hidden_size,
                self.config.num_key_value_heads * self.head_dim,
                bias_attr=True,
            )
            self.v_proj = Linear(
                self.hidden_size,
                self.config.num_key_value_heads * self.head_dim,
                bias_attr=True,
            )
            self.o_proj = Linear(self.hidden_size, self.hidden_size, bias_attr=False)

        self.rotary_emb = Qwen2VLRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,  # default true
        cache_position: Optional[paddle.Tensor] = None,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        bsz, q_len, _ = hidden_states.shape

        try:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        except:
            hidden_states = hidden_states.astype(self.config.dtype)
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        target_query_shape = [0, 0, self.num_heads, self.head_dim]
        target_key_value_shape = [0, 0, self.num_key_value_heads, self.head_dim]
        query_states = query_states.reshape(shape=target_query_shape)
        key_states = key_states.reshape(shape=target_key_value_shape)
        value_states = value_states.reshape(shape=target_key_value_shape)

        new_perm = [0, 2, 1, 3]
        query_states = query_states.transpose(new_perm)
        key_states = key_states.transpose(new_perm)
        value_states = value_states.transpose(new_perm)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += cache_position[0] + 1

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        if past_key_value is not None:
            key_states = paddle.concat([past_key_value[0], key_states], axis=2)
            value_states = paddle.concat([past_key_value[1], value_states], axis=2)
        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        query_states = query_states.astype("float32")
        key_states = key_states.astype("float32")
        value_states = value_states.astype("float32")

        attn_weights = paddle.matmul(
            query_states, key_states.transpose([0, 1, 3, 2])
        ) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, axis=-1, dtype="float32")
        attn_weights = nn.functional.dropout(
            x=attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = paddle.matmul(
            attn_weights.cast(self.config.dtype), value_states.cast(self.config.dtype)
        )

        if attn_output.shape != [bsz, self.num_heads, q_len, self.head_dim]:
            raise ValueError(
                f"`attn_output` should be of size {(bsz, q_len, self.num_heads, self.head_dim)}, but is"
                f" {attn_output.shape}"
            )

        attn_output = attn_output.transpose([0, 2, 1, 3])
        attn_output = attn_output.reshape([bsz, q_len, -1])

        if self.o_proj.weight.dtype == paddle.bfloat16:
            attn_output = attn_output.astype(paddle.bfloat16)
        elif self.o_proj.weight.dtype == paddle.float16:
            attn_output = attn_output.astype(paddle.float16)
        elif self.o_proj.weight.dtype == paddle.float32:
            attn_output = attn_output.astype(paddle.float32)

        attn_output = self.o_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value


class Qwen2VLFlashAttention2(Qwen2VLAttention):
    """
    Qwen2VL flash attention module, following Qwen2VL attention module. This module inherits from `Qwen2VLAttention`
    as the weights of the module stays untouched. The only required change would be on the forward pass
    where it needs to correctly call the public API of flash attention and deal with padding tokens
    in case the input contains any of them. Additionally, for sliding window attention, we apply SWA only to the bottom
    config.max_window_layers layers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,  # default true
        cache_position: Optional[paddle.Tensor] = None,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        bsz, q_len, _ = tuple(hidden_states.shape)

        try:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        except:
            hidden_states = hidden_states.astype("bfloat16")
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        target_query_shape = [0, 0, self.num_heads, self.head_dim]
        target_key_value_shape = [0, 0, self.num_key_value_heads, self.head_dim]
        query_states = query_states.reshape(shape=target_query_shape)
        key_states = key_states.reshape(shape=target_key_value_shape)
        value_states = value_states.reshape(shape=target_key_value_shape)

        new_perm = [0, 2, 1, 3]
        query_states = query_states.transpose(new_perm)
        key_states = key_states.transpose(new_perm)
        value_states = value_states.transpose(new_perm)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += cache_position[0] + 1

        # Because the input can be padded, the absolute sequence length depends on the max position id.
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        if past_key_value is not None:
            key_states = paddle.concat([past_key_value[0], key_states], axis=2)
            value_states = paddle.concat([past_key_value[1], value_states], axis=2)
        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Reashape to the expected shape for Flash Attention
        # [1, 3599, 12, 128]
        query_states = query_states.transpose(perm=[0, 2, 1, 3])
        key_states = key_states.transpose(perm=[0, 2, 1, 3])
        value_states = value_states.transpose(perm=[0, 2, 1, 3])

        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len
        )

        attn_output = attn_output.reshape([bsz, q_len, -1])
        attn_output = self.o_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`paddle.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`paddle.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`paddle.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`paddle.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        # Contains at least one padding token in the sequence
        causal = self.is_causal and query_length != 1

        if _IS_NPU:
            if attention_mask is not None:
                attn_output = paddle.nn.functional.flash_attention_npu(  # TODO: flash_attn_unpadded
                    query_states,
                    key_states,
                    value_states,
                    attn_mask=attention_mask,
                    dropout=dropout,
                    causal=causal,
                    is_varlen=True,
                )
            else:
                dtype = query_states.dtype
                attn_output = paddle.nn.functional.flash_attention_npu(  # TODO: flash_attn_unpadded
                    query_states.astype("bfloat16"),
                    key_states.astype("bfloat16"),
                    value_states.astype("bfloat16"),
                    attn_mask=attention_mask,
                    dropout=dropout,
                    causal=causal,
                )
                attn_output = attn_output.astype(dtype)
        else:
            head_dim = query_states.shape[-1]
            softmax_scale = head_dim**-0.5  # TODO: 需要手动加上

            if attention_mask is not None:
                batch_size = query_states.shape[0]
                (
                    query_states,
                    key_states,
                    value_states,
                    indices_q,
                    cu_seq_lens,
                    max_seq_lens,
                ) = self._unpad_input(
                    query_states, key_states, value_states, attention_mask, query_length
                )
                cu_seqlens_q, cu_seqlens_k = cu_seq_lens
                max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

                attn_output_unpad = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    scale=softmax_scale,  # not softmax_scale=
                    dropout=dropout,
                    causal=causal,
                )[0]

                attn_output = pad_input(
                    attn_output_unpad, indices_q, batch_size, query_length
                )
            else:
                attn_output = flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    dropout,
                    causal=causal,
                )[0]

        return attn_output

    def _unpad_input(
        self, query_layer, key_layer, value_layer, attention_mask, query_length
    ):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        # TODO：cuda error
        key_layer = index_first_axis(
            key_layer.reshape([batch_size * kv_seq_len, num_key_value_heads, head_dim]),
            indices_k,
        )
        value_layer = index_first_axis(
            value_layer.reshape(
                [batch_size * kv_seq_len, num_key_value_heads, head_dim]
            ),
            indices_k,
        )

        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(
                    [batch_size * kv_seq_len, self.num_heads, head_dim]
                ),
                indices_k,
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = paddle.arange(
                batch_size + 1, dtype=paddle.int32
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(
                query_layer, attention_mask
            )

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q.to(paddle.int64),
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


class Qwen2VLDecoderLayer(nn.Layer):
    def __init__(self, config: Qwen2VLConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        # use_sliding_window false
        if (
            config.use_sliding_window
            and config.attn_implementation != "flash_attention_2"
        ):
            logging.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config.attn_implementation}`; "
                "unexpected results may be encountered."
            )

        self.self_attn = create_attention_module(config, "qwen2vl", layer_idx=layer_idx)
        # self.self_attn = Qwen2VLAttention(config, layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(
            config, config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen2RMSNorm(
            config, config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[paddle.Tensor] = None,
        **kwargs,
    ):
        """
        Args:
            hidden_states (`paddle.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`paddle.Tensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(paddle.Tensor)`, *optional*): cached past key and value projection states
            cache_position (`paddle.Tensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class Qwen2VLPreTrainedModel(PretrainedModel):
    config_class = Qwen2VLConfig
    base_model_prefix = "model"
    _no_split_modules = ["Qwen2VLDecoderLayer", "Qwen2VLVisionBlock"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, layer):
        std = 0.2
        if isinstance(layer, (nn.Linear, nn.Conv3D)):
            nn.initializer.Normal(mean=0.0, std=std)(layer.weight)
            if layer.bias is not None:
                nn.initializer.Constant(0.0)(layer.bias)
        elif isinstance(layer, nn.Embedding):
            nn.initializer.Normal(mean=0.0, std=std)(layer.weight)
            if layer._padding_idx is not None:
                with paddle.no_grad():
                    layer.weight[layer._padding_idx] = 0.0


class Qwen2VisionTransformerPretrainedModel(Qwen2VLPreTrainedModel):
    config_class = Qwen2VLVisionConfig
    _no_split_modules = ["Qwen2VLVisionBlock"]

    def __init__(self, config) -> None:
        super().__init__(config)
        self.spatial_merge_size = config.spatial_merge_size

        self.patch_embed = PatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
        )

        head_dim = config.embed_dim // config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.LayerList(
            [Qwen2VLVisionBlock(config) for _ in range(config.depth)]
        )
        self.merger = PatchMerger(dim=config.hidden_size, context_dim=config.embed_dim)
        self.enable_recompute = False

    def get_dtype(self) -> paddle.dtype:
        return self.blocks[0].mlp.fc2.weight.dtype

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = paddle.arange(h).unsqueeze(1).expand([-1, w])
            hpos_ids = hpos_ids.reshape(
                [
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                ]
            )
            hpos_ids = hpos_ids.transpose(perm=[0, 2, 1, 3])
            hpos_ids = hpos_ids.flatten()

            wpos_ids = paddle.arange(w).unsqueeze(0).expand([h, -1])
            wpos_ids = wpos_ids.reshape(
                [
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                ]
            )
            wpos_ids = wpos_ids.transpose([0, 2, 1, 3])
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(
                paddle.stack(x=[hpos_ids, wpos_ids], axis=-1).tile(repeat_times=[t, 1])
            )
        pos_ids = paddle.concat(x=pos_ids, axis=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(start_axis=1)
        return rotary_pos_emb

    @paddle.jit.not_to_static
    def recompute_training_full(
        self,
        layer_module: nn.Layer,
        hidden_states: paddle.Tensor,
        cu_seqlens_now: paddle.Tensor,
        rotary_pos_emb: paddle.Tensor,
    ):
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        hidden_states = recompute(
            create_custom_forward(layer_module),
            hidden_states,
            cu_seqlens_now,
            rotary_pos_emb,
            # use_reentrant=self.config.recompute_use_reentrant,
        )
        return hidden_states

    def forward(
        self, hidden_states: paddle.Tensor, grid_thw: paddle.Tensor
    ) -> paddle.Tensor:
        # breakpoint()
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        cu_seqlens = paddle.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(axis=0, dtype="int32")
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for idx, blk in enumerate(self.blocks):
            if self.enable_recompute and self.training:
                hidden_states = self.recompute_training_full(
                    blk, hidden_states, cu_seqlens, rotary_pos_emb
                )
            else:
                hidden_states = blk(
                    hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb
                )

        return self.merger(hidden_states)


class Qwen2VLModel(Qwen2VLPreTrainedModel):
    def __init__(self, config: Qwen2VLConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        # Recompute defaults to False and is controlled by Trainer

        if (
            config.tensor_parallel_degree > 1
            and config.vocab_size % config.tensor_parallel_degree == 0
        ):
            self.embed_tokens = mpu.VocabParallelEmbedding(
                self.vocab_size,
                self.hidden_size,
                weight_attr=paddle.ParamAttr(initializer=nn.initializer.XavierNormal()),
            )
        else:
            self.embed_tokens = nn.Embedding(
                self.vocab_size,
                self.hidden_size,
            )

        # self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.LayerList(
            [
                Qwen2VLDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen2RMSNorm(config, config.hidden_size, eps=config.rms_norm_eps)

        self.enamble_recompute = False

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @staticmethod
    def _prepare_decoder_attention_mask(
        attention_mask, input_shape, past_key_values_length, dtype
    ):
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            if len(attention_mask.shape) == 2:
                expanded_attn_mask = _expand_2d_mask(
                    attention_mask, dtype, tgt_length=input_shape[-1]
                )
                # For decoding phase in generation, seq_length = 1, we don't need to add causal mask
                if input_shape[-1] > 1:
                    combined_attention_mask = _make_causal_mask(
                        input_shape,
                        past_key_values_length=past_key_values_length,
                    )
                    expanded_attn_mask = expanded_attn_mask & combined_attention_mask
            # [bsz, seq_len, seq_len] -> [bsz, 1, seq_len, seq_len]
            elif len(attention_mask.shape) == 3:
                expanded_attn_mask = attention_mask.unsqueeze(1).astype("bool")
            # if attention_mask is already 4-D, do nothing
            else:
                expanded_attn_mask = attention_mask
        else:
            expanded_attn_mask = _make_causal_mask(
                input_shape,
                past_key_values_length=past_key_values_length,
            )
        # Convert bool attention_mask to float attention mask, which will be added to attention_scores later
        expanded_attn_mask = paddle.where(
            expanded_attn_mask, 0.0, paddle.finfo(dtype).min
        ).astype(dtype)
        return expanded_attn_mask

    @paddle.jit.not_to_static
    def recompute_training_full(
        self,
        layer_module: nn.Layer,
        hidden_states: paddle.Tensor,
        attention_mask: paddle.Tensor,
        position_ids: Optional[paddle.Tensor],
        past_key_value: paddle.Tensor,
        output_attentions: bool,
        use_cache: bool,
        cache_position: Optional[paddle.Tensor] = None,
    ):
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        hidden_states = recompute(
            create_custom_forward(layer_module),
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            cache_position,
            use_reentrant=self.config.recompute_use_reentrant,
        )
        return hidden_states

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[paddle.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.layers))
        # NOTE: to make cache can be clear in-time
        past_key_values = list(past_key_values)

        seq_length_with_past = seq_length
        cache_length = 0
        if past_key_values[0] is not None:
            cache_length = past_key_values[0][0].shape[2]  # shape[1] in qwen2
            seq_length_with_past += cache_length

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # embed positions
        if attention_mask is None:
            # [bs, seq_len]
            attention_mask = paddle.ones(
                (batch_size, seq_length_with_past), dtype=paddle.bool
            )

        if flash_attn_varlen_func:
            causal_mask = attention_mask
        else:
            causal_mask = self._prepare_decoder_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                cache_length,
                inputs_embeds.dtype,
            )  # [bs, 1, seq_len, seq_len]

        if cache_position is None:
            past_seen_tokens = (
                past_key_values[0][0].shape[2] if past_key_values[0] is not None else 0
            )
            cache_position = paddle.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1]
            )

        if position_ids is None:
            # the hard coded `3` is for temporal, height and width.
            position_ids = cache_position.reshape([1, 1, -1]).expand(
                [3, inputs_embeds.shape[0], -1]
            )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = ()

        for idx, (decoder_layer) in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            if self.enamble_recompute and self.training:
                layer_outputs = self.recompute_training_full(
                    decoder_layer,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,  # False
                    use_cache=use_cache,  # True
                    cache_position=cache_position,
                )

            # NOTE: clear outdate cache after it has been used for memory saving
            past_key_value = past_key_values[idx] = None

            hidden_states = layer_outputs[0]

            next_decoder_cache = (
                next_decoder_cache + (layer_outputs[-1],) if use_cache else None
            )

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class Qwen2LMHead(nn.Layer):
    def __init__(self, config, embedding_weights=None, transpose_y=False):
        super(Qwen2LMHead, self).__init__()
        self.config = config
        if (
            config.tensor_parallel_degree > 1
            and config.vocab_size % config.tensor_parallel_degree == 0
        ):
            vocab_size = config.vocab_size // config.tensor_parallel_degree
        else:
            vocab_size = config.vocab_size

        self.transpose_y = transpose_y
        if transpose_y:
            # only for weight from embedding_weights
            if embedding_weights is not None:
                self.weight = embedding_weights
            else:
                self.weight = self.create_parameter(
                    shape=[vocab_size, config.hidden_size],
                    dtype=paddle.get_default_dtype(),
                )
        else:

            if vocab_size != config.vocab_size:
                with get_rng_state_tracker().rng_state():
                    self.weight = self.create_parameter(
                        shape=[config.hidden_size, vocab_size],
                        dtype=paddle.get_default_dtype(),
                    )
            else:
                self.weight = self.create_parameter(
                    shape=[config.hidden_size, vocab_size],
                    dtype=paddle.get_default_dtype(),
                )

        # Must set distributed attr for Tensor Parallel !
        self.weight.is_distributed = (
            True if (vocab_size != config.vocab_size) else False
        )
        if self.weight.is_distributed:
            # for tie_word_embeddings
            self.weight.split_axis = 0 if self.transpose_y else 1

    def forward(self, hidden_states, tensor_parallel_output=None):
        if tensor_parallel_output is None:
            tensor_parallel_output = self.config.tensor_parallel_output

        # 确保数据类型一致
        if self.weight.dtype != hidden_states.dtype:
            hidden_states = paddle.cast(hidden_states, self.weight.dtype)

        logits = parallel_matmul(
            hidden_states,
            self.weight,
            transpose_y=self.transpose_y,
            tensor_parallel_output=tensor_parallel_output,
        )
        return logits


class Qwen2VLForConditionalGeneration(Qwen2VLPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, attn_implementation="flash_attention_2"):
        super().__init__(config)
        config._attn_implementation = attn_implementation
        config.vision_config._attn_implementation = attn_implementation

        self.visual = Qwen2VisionTransformerPretrainedModel._from_config(
            config.vision_config
        )
        self.model = Qwen2VLModel(config)
        self.vocab_size = config.vocab_size

        if config.tie_word_embeddings:
            self.lm_head = Qwen2LMHead(
                config,
                embedding_weights=self.model.embed_tokens.weight,
                transpose_y=True,
            )
            self.tie_weights()
        else:
            self.lm_head = Qwen2LMHead(config)
        self.padding_side = "left"  # set it to left by default, user can use setter to change padding_sides

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @staticmethod
    def get_rope_index(
        spatial_merge_size,
        image_token_id,
        video_token_id,
        vision_start_token_id,
        input_ids: paddle.Tensor,
        image_grid_thw: Optional[paddle.Tensor] = None,
        video_grid_thw: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embedding for text part.
            Examples:
                Assume we have a video input with 3 temporal patches, 2 height patches and 2 width patches.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [3, 4, 5, 6, 7]
                text height position_ids: [3, 4, 5, 6, 7]
                text width position_ids: [3, 4, 5, 6, 7]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`paddle.Tensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`paddle.Tensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`paddle.Tensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            attention_mask (`paddle.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`paddle.Tensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`paddle.Tensor` of shape `(batch_size)`)
        """
        mrope_position_deltas = []
        if image_grid_thw is not None or video_grid_thw is not None:
            total_input_ids = input_ids
            position_ids = paddle.ones(
                [3, input_ids.shape[0], input_ids.shape[1]], dtype=input_ids.dtype
            )
            image_index, video_index = 0, 0
            for i, input_ids in enumerate(total_input_ids):
                # TODO: CUDA error in some paddle version
                if attention_mask is not None:
                    input_ids = paddle.to_tensor(
                        input_ids.cpu()[attention_mask[i].cpu() == 1]
                    )  # NOTE 原始写法

                image_nums, video_nums = 0, 0
                vision_start_indices = paddle.nonzero(
                    input_ids == vision_start_token_id
                ).squeeze(
                    1
                )  # NOTE 原始写法

                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (
                    (vision_tokens == image_token_id).sum()
                    if vision_tokens.numel() > 0
                    else 0
                )
                video_nums = (
                    (vision_tokens == video_token_id).sum()
                    if vision_tokens.numel() > 0
                    else 0
                )
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    llm_pos_ids_list.append(
                        paddle.arange(text_len).reshape([1, -1]).expand([3, -1])
                        + st_idx
                    )

                    t_index = (
                        paddle.arange(llm_grid_t)
                        .reshape([-1, 1])
                        .expand([-1, llm_grid_h * llm_grid_w])
                        .flatten()
                    )
                    h_index = (
                        paddle.arange(llm_grid_h)
                        .reshape([1, -1, 1])
                        .expand([llm_grid_t, -1, llm_grid_w])
                        .flatten()
                    )
                    w_index = (
                        paddle.arange(llm_grid_w)
                        .reshape([1, 1, -1])
                        .expand([llm_grid_t, llm_grid_h, -1])
                        .flatten()
                    )
                    llm_pos_ids_list.append(
                        paddle.stack([t_index, h_index, w_index]) + text_len + st_idx
                    )
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(
                        paddle.arange(text_len).reshape([1, -1]).expand([3, -1])
                        + st_idx
                    )

                llm_positions = paddle.concat(llm_pos_ids_list, axis=1).reshape([3, -1])
                if _IS_NPU:
                    bool_indices = (
                        (attention_mask[i] == 1)
                        .unsqueeze(0)
                        .tile([position_ids.shape[0], 1])
                    )
                    position_ids[:, i] = paddle.index_put(
                        position_ids[:, i], [bool_indices], llm_positions.reshape([-1])
                    )
                else:
                    position_ids[..., i, attention_mask[i] == 1] = llm_positions
                mrope_position_deltas.append(
                    llm_positions.max() + 1 - len(total_input_ids[i])
                )
            mrope_position_deltas = paddle.to_tensor(mrope_position_deltas).unsqueeze(1)
        else:
            if attention_mask is not None:
                position_ids = paddle.cast(attention_mask, dtype="int64").cumsum(-1) - 1
                position_ids.masked_fill_(mask=attention_mask == 0, value=1)
                position_ids = position_ids.unsqueeze(0).expand([3, -1, -1])
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(
                    -1, keepdim=True
                )[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    paddle.arange(input_ids.shape[1])
                    .reshape([1, 1, -1])
                    .expand(shape=[3, input_ids.shape[0], -1])
                )
                mrope_position_deltas = paddle.zeros(
                    [input_ids.shape[0], 1], dtype=input_ids.dtype
                )

        return position_ids, mrope_position_deltas

    def update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        # num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        model_kwargs = super().update_model_kwargs_for_generation(
            outputs=outputs,
            model_kwargs=model_kwargs,
            is_encoder_decoder=is_encoder_decoder,
        )

        if getattr(outputs, "rope_deltas", None) is not None:
            model_kwargs["rope_deltas"] = outputs.rope_deltas

        return model_kwargs

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[paddle.Tensor] = None,
        pixel_values_videos: Optional[paddle.Tensor] = None,
        image_grid_thw: Optional[paddle.Tensor] = None,
        video_grid_thw: Optional[paddle.Tensor] = None,
        rope_deltas: Optional[paddle.Tensor] = None,
    ):
        """
        Args:
            labels (`paddle.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states  # fmt:skip

        return_dict = True  # return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

            if pixel_values is not None:
                pixel_values = paddle.cast(pixel_values, inputs_embeds.dtype)

                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                image_embeds = paddle.cast(image_embeds, inputs_embeds.dtype)

                image_mask = input_ids == self.config.image_token_id
                if self.training:
                    inputs_embeds = inputs_embeds.clone()
                inputs_embeds[image_mask] = image_embeds
            if pixel_values_videos is not None:
                pixel_values_videos = paddle.cast(
                    pixel_values_videos, inputs_embeds.dtype
                )
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                video_embeds = paddle.cast(video_embeds, inputs_embeds.dtype)
                video_mask = input_ids == self.config.video_token_id
                inputs_embeds[video_mask] = video_embeds
            if attention_mask is not None:
                attention_mask = attention_mask

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        tensor_parallel_output = (
            self.config.tensor_parallel_output
            and self.config.tensor_parallel_degree > 1
        )

        logits = self.lm_head(
            hidden_states, tensor_parallel_output=tensor_parallel_output
        )

        logits = paddle.cast(logits, "float32")

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            # Flatten the tokens
            shift_logits = shift_logits.reshape([-1, self.config.vocab_size])
            shift_labels = shift_labels.reshape([-1])
            if _IS_NPU:
                tmp = F.log_softmax(shift_logits, axis=1)
                loss = F.nll_loss(tmp, shift_labels, reduction="sum")
            else:
                loss_fct = nn.CrossEntropyLoss(reduction="sum")
                loss = loss_fct(shift_logits, shift_labels)
            label_sum = paddle.sum(shift_labels != -100).cast("float32")
            loss = loss / label_sum

        if not return_dict:
            output = (logits,) + tuple(outputs[1:])
            return (loss,) + output if loss is not None else output

        return Qwen2VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=rope_deltas,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        **kwargs,
    ):

        batch_size, seq_length = input_ids.shape
        if past_key_values is None:
            cache_position = paddle.arange(input_ids.shape[1])
        else:
            cache_position = paddle.to_tensor([seq_length - 1])

        if past_key_values is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        rope_deltas = kwargs.get("rope_deltas", None)

        if attention_mask is not None and position_ids is None:
            if cache_position is None or (
                cache_position is not None and cache_position[0] == 0
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    self.config.vision_config.spatial_merge_size,
                    self.config.image_token_id,
                    self.config.video_token_id,
                    self.config.vision_start_token_id,
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask,
                )
            else:
                batch_size, seq_length = input_ids.shape
                delta = (
                    cache_position[0] + rope_deltas
                    if cache_position is not None and rope_deltas is not None
                    else 0
                )
                position_ids = paddle.arange(seq_length)
                position_ids = position_ids.reshape([1, -1]).expand([batch_size, -1])
                position_ids = position_ids + delta
                position_ids = position_ids.unsqueeze(axis=0).expand([3, -1, -1])

        if cache_position[0] != 0:
            pixel_values = None
            pixel_values_videos = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,  # [3, 1, 3602]
                "past_key_values": past_key_values,  # DynamicCache()
                "use_cache": use_cache,  # 1
                "attention_mask": attention_mask,  # [1, 3602]
                "pixel_values": pixel_values,  # [14308, 1176]
                "pixel_values_videos": pixel_values_videos,
                "image_grid_thw": image_grid_thw,  # [[  1,  98, 146]]
                "video_grid_thw": video_grid_thw,
                "rope_deltas": rope_deltas,  # [[-3504]]
            }
        )
        return model_inputs

    def gme_qwen2_vl_forward(
        self,
        input_ids: paddle.Tensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[paddle.Tensor] = None,
        pixel_values_videos: Optional[paddle.Tensor] = None,
        image_grid_thw: Optional[paddle.Tensor] = None,
        video_grid_thw: Optional[paddle.Tensor] = None,
        rope_deltas: Optional[paddle.Tensor] = None,
    ):

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = True  # return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                # 确保 pixel_values 和 inputs_embeds 使用相同的数据类型
                pixel_values = paddle.cast(pixel_values, inputs_embeds.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                # 确保 image_embeds 和 inputs_embeds 使用相同的数据类型
                image_embeds = paddle.cast(image_embeds, inputs_embeds.dtype)
                image_mask = input_ids == self.config.image_token_id
                if self.training:
                    inputs_embeds = inputs_embeds.clone()

                inputs_embeds[image_mask] = image_embeds

            if pixel_values_videos is not None:
                # 确保 pixel_values_videos 和 inputs_embeds 使用相同的数据类型
                pixel_values_videos = paddle.cast(
                    pixel_values_videos, inputs_embeds.dtype
                )
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                # 确保 video_embeds 和 inputs_embeds 使用相同的数据类型
                video_embeds = paddle.cast(video_embeds, inputs_embeds.dtype)
                video_mask = input_ids == self.config.video_token_id
                inputs_embeds[video_mask] = video_embeds
            if attention_mask is not None:
                attention_mask = attention_mask

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        # get last hidden state
        last_hidden_state = hidden_states[:, -1, :]
        return last_hidden_state


class PPDocBeeInference(Qwen2VLForConditionalGeneration):
    set_inference_operations(get_inference_operations() + ["docbee_generate"])

    @benchmark.timeit_with_options(name="docbee_generate")
    def generate(self, inputs, **kwargs):
        max_new_tokens = kwargs.get("max_new_tokens", 2048)
        temperature = kwargs.get("temperature", 0.1)
        top_p = kwargs.get("top_p", 0.001)
        top_k = kwargs.get("top_k", 1)
        with paddle.no_grad():
            generated_ids = super().generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )

        return generated_ids
