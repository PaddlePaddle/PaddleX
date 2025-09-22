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

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import List, Optional, Tuple, Union

import numpy as np

from .....utils.deps import is_dep_available

if all(map(is_dep_available, ("einops", "torch", "transformers", "vllm"))):

    import torch
    import torch.nn as nn
    from einops import rearrange
    from transformers import BatchFeature
    from transformers.activations import GELUActivation
    from transformers.modeling_outputs import (
        BaseModelOutput,
        BaseModelOutputWithPooling,
    )
    from transformers.utils import torch_int
    from vllm.compilation.decorators import support_torch_compile
    from vllm.config import VllmConfig
    from vllm.distributed import get_tensor_model_parallel_world_size
    from vllm.model_executor.layers.activation import get_act_fn
    from vllm.model_executor.layers.linear import (
        ColumnParallelLinear,
        QKVParallelLinear,
        RowParallelLinear,
    )
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.quantization import QuantizationConfig
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
    from vllm.model_executor.model_loader.weight_utils import (
        default_weight_loader,
        maybe_remap_kv_scale_name,
    )

    try:
        from vllm.model_executor.models.ernie45 import Ernie4_5_ForCausalLM
    except ImportError:
        from vllm.model_executor.models.ernie45 import (
            Ernie4_5ForCausalLM as Ernie4_5_ForCausalLM,
        )
    from vllm.model_executor.models.interfaces import SupportsMultiModal
    from vllm.model_executor.models.utils import (
        AutoWeightsLoader,
        PPMissingLayer,
        is_pp_missing_parameter,
        merge_multimodal_embeddings,
    )
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.multimodal.inputs import (
        MultiModalDataDict,
        MultiModalFieldConfig,
        MultiModalKwargs,
        NestedTensors,
    )
    from vllm.multimodal.parse import (
        ImageProcessorItems,
        ImageSize,
        MultiModalDataItems,
    )
    from vllm.multimodal.processing import (
        BaseMultiModalProcessor,
        BaseProcessingInfo,
        PromptReplacement,
        PromptUpdate,
    )
    from vllm.multimodal.profiling import BaseDummyInputsBuilder
    from vllm.platforms import _Backend
    from vllm.sequence import IntermediateTensors

    def smart_resize(
        height: int,
        width: int,
        factor: int = 28,
        min_pixels: int = 28 * 28 * 130,
        max_pixels: int = 28 * 28 * 1280,
    ):
        """Rescales the image so that the following conditions are met:

        1. Both dimensions (height and width) are divisible by 'factor'.

        2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

        3. The aspect ratio of the image is maintained as closely as possible.

        """
        # if height < factor or width < factor:
        #    raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
        # if int(height < factor//4) + int(width < factor//4):
        #     raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor//4}")

        if height < factor:
            print(
                f"smart_resize: height={height} < factor={factor}, reset height=factor"
            )
            width = round((width * factor) / height)
            height = factor

        if width < factor:
            print(f"smart_resize: width={width} < factor={factor}, reset width=factor")
            height = round((height * factor) / width)
            width = factor

        if max(height, width) / min(height, width) > 200:
            raise ValueError(
                f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
            )
        h_bar = round(height / factor) * factor
        w_bar = round(width / factor) * factor
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = math.floor(height / beta / factor) * factor
            w_bar = math.floor(width / beta / factor) * factor
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = math.ceil(height * beta / factor) * factor
            w_bar = math.ceil(width * beta / factor) * factor
        return h_bar, w_bar

    class PPOCRVLProcessingInfo(BaseProcessingInfo):

        def get_hf_config(self):
            return self.ctx.get_hf_config()

        def get_hf_processor(self, **kwargs: object):
            return self.ctx.get_hf_processor(**kwargs)

        def get_image_processor(self, **kwargs: object):
            return self.get_hf_processor(**kwargs).image_processor

        def get_supported_mm_limits(self):
            return {"image": None}

        def get_num_image_tokens(
            self,
            *,
            image_width: int,
            image_height: int,
        ) -> int:
            image_processor = self.get_image_processor()

            do_resize = True
            hf_config = self.get_hf_config()
            vision_config = hf_config.vision_config
            patch_size = vision_config.patch_size
            merge_size = vision_config.spatial_merge_size

            if do_resize:
                resized_height, resized_width = smart_resize(
                    height=image_height,
                    width=image_width,
                    factor=patch_size * merge_size,
                    min_pixels=image_processor.min_pixels,
                    max_pixels=image_processor.max_pixels,
                )
                preprocessed_size = ImageSize(
                    width=resized_width, height=resized_height
                )
            else:
                preprocessed_size = ImageSize(width=image_width, height=image_height)

            grid_t = 1
            grid_h = preprocessed_size.height // patch_size
            grid_w = preprocessed_size.width // patch_size

            num_patches = grid_t * grid_h * grid_w
            num_image_tokens = num_patches // (merge_size**2)

            return num_image_tokens

        def get_image_size_with_most_features(self) -> ImageSize:
            hf_config = self.get_hf_config()
            image_size = hf_config.vision_config.image_size
            return ImageSize(height=image_size, width=image_size)

    class PPOCRVLDummyInputsBuilder(BaseDummyInputsBuilder[PPOCRVLProcessingInfo]):

        def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
            num_images = mm_counts.get("image", 0)

            processor = self.info.get_hf_processor()
            image_token = processor.image_token

            return image_token * num_images

        def get_dummy_mm_data(
            self,
            seq_len: int,
            mm_counts: Mapping[str, int],
        ) -> MultiModalDataDict:
            num_images = mm_counts.get("image", 0)

            (target_width, target_height) = (
                self.info.get_image_size_with_most_features()
            )

            return {
                "image": self._get_dummy_images(
                    width=target_width, height=target_height, num_images=num_images
                )
            }

    class PPOCRVLMultiModalProcessor(BaseMultiModalProcessor[PPOCRVLProcessingInfo]):

        def _call_hf_processor(
            self,
            prompt: str,
            mm_data: Mapping[str, object],
            mm_kwargs: Mapping[str, object],
            tok_kwargs: Mapping[str, object],
        ) -> BatchFeature:
            if mm_data:
                processed_outputs = self.info.ctx.call_hf_processor(
                    self.info.get_hf_processor(**mm_kwargs),
                    dict(text=prompt, **mm_data),
                    dict(**mm_kwargs, **tok_kwargs),
                )
                processed_outputs["pixel_values"] = processed_outputs[
                    "pixel_values"
                ].unsqueeze(0)
            else:
                tokenizer = self.info.get_tokenizer()
                processed_outputs = tokenizer(
                    prompt, add_special_tokens=True, return_tensors="pt"
                )
            return processed_outputs

        def _get_mm_fields_config(
            self,
            hf_inputs: BatchFeature,
            hf_processor_mm_kwargs: Mapping[str, object],
        ) -> Mapping[str, MultiModalFieldConfig]:
            return dict(
                pixel_values=MultiModalFieldConfig.batched("image"),
                image_grid_thw=MultiModalFieldConfig.batched("image"),
            )

        def _get_prompt_updates(
            self,
            mm_items: MultiModalDataItems,
            hf_processor_mm_kwargs: Mapping[str, object],
            out_mm_kwargs: MultiModalKwargs,
        ) -> Sequence[PromptUpdate]:
            hf_config = self.info.get_hf_config()
            image_token_id = hf_config.image_token_id

            def get_replacement(item_idx: int):
                images = mm_items.get_items("image", ImageProcessorItems)

                image_size = images.get_image_size(item_idx)
                num_image_tokens = self.info.get_num_image_tokens(
                    image_width=image_size.width,
                    image_height=image_size.height,
                )

                return [image_token_id] * num_image_tokens

            return [
                PromptReplacement(
                    modality="image",
                    target=[image_token_id],
                    replacement=get_replacement,
                ),
            ]

    class Projector(nn.Module):

        def __init__(
            self,
            text_config,
            vision_config,
            prefix: str = "",
        ):
            super().__init__()
            self.text_config = text_config
            self.vision_config = vision_config
            self.merge_kernel_size = (2, 2)

            self.hidden_size = (
                self.vision_config.hidden_size
                * self.merge_kernel_size[0]
                * self.merge_kernel_size[1]
            )

            self.pre_norm = torch.nn.LayerNorm(
                self.vision_config.hidden_size, eps=1e-05
            )
            self.linear_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
            self.act = GELUActivation()
            self.linear_2 = nn.Linear(
                self.hidden_size, self.text_config.hidden_size, bias=True
            )

        def forward(
            self,
            image_features: torch.Tensor,
            image_grid_thw: List[Tuple[int, int, int]],
        ) -> torch.Tensor:
            m1, m2 = self.merge_kernel_size
            if isinstance(image_features, (list, tuple)):
                processed_features = list()
                for image_feature, image_grid in zip(image_features, image_grid_thw):
                    image_feature = self.pre_norm(image_feature)
                    t, h, w = image_grid
                    from einops import rearrange

                    image_feature = rearrange(
                        image_feature,
                        "(t h p1 w p2) d -> (t h w) (p1 p2 d)",
                        t=t,
                        h=h // m1,
                        p1=m1,
                        w=w // m2,
                        p2=m2,
                    )
                    hidden_states = self.linear_1(image_feature)
                    hidden_states = self.act(hidden_states)
                    hidden_states = self.linear_2(hidden_states)
                    processed_features.append(hidden_states)

                return processed_features

            dims = image_features.shape[:-1]
            dim = image_features.shape[-1]
            image_features = image_features.view(np.prod(dims), dim)
            hidden_states = self.pre_norm(image_features).view(-1, self.hidden_size)
            hidden_states = self.linear_1(hidden_states)
            hidden_states = self.act(hidden_states)
            hidden_states = self.linear_2(hidden_states)

            return hidden_states.view(*dims, -1)

    class SiglipVisionEmbeddings(nn.Module):

        def __init__(self, config):
            super().__init__()
            self.config = config
            self.embed_dim = config.hidden_size
            self.image_size = config.image_size
            self.patch_size = config.patch_size

            self.patch_embedding = nn.Conv2d(
                in_channels=config.num_channels,
                out_channels=self.embed_dim,
                kernel_size=self.patch_size,
                stride=self.patch_size,
                padding="valid",
            )

            self.num_patches = (self.image_size // self.patch_size) ** 2
            self.num_positions = self.num_patches
            self.cache_position_embedding = dict()
            self.cache_position_count = dict()
            self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
            self.packing_position_embedding = nn.Embedding(32768, self.embed_dim)

            self.register_buffer(
                "position_ids",
                torch.arange(self.num_positions).expand((1, -1)),
                persistent=False,
            )

        def interpolate_pos_encoding(
            self,
            embeddings: torch.Tensor,
            height: int,
            width: int,
            is_after_patchify: bool = False,
        ) -> torch.Tensor:

            num_positions = self.position_embedding.weight.shape[0]

            patch_pos_embed = self.position_embedding.weight.unsqueeze(0)

            dim = embeddings.shape[-1]

            if is_after_patchify:
                new_height = height
                new_width = width
            else:
                new_height = height // self.patch_size
                new_width = width // self.patch_size

            sqrt_num_positions = torch_int(num_positions**0.5)
            patch_pos_embed = patch_pos_embed.reshape(
                1, sqrt_num_positions, sqrt_num_positions, dim
            )
            patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

            patch_pos_embed = nn.functional.interpolate(
                patch_pos_embed,
                size=(new_height, new_width),
                mode="bilinear",
                align_corners=False,
            )

            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            return patch_pos_embed

        def fetch_position_embedding_lfu_cache(
            self, embeddings, h, w, max_cache: int = 20
        ):
            grid = (h, w)
            if grid in self.cache_position_embedding:
                self.cache_position_count[grid] += 1
                return self.cache_position_embedding[grid]

            if len(self.cache_position_embedding) >= max_cache:
                min_hit_grid = min(
                    self.cache_position_count,
                    key=self.cache_position_count.get,
                )
                self.cache_position_count.pop(min_hit_grid)
                self.cache_position_embedding.pop(min_hit_grid)

            position_embedding = self.interpolate_pos_encoding(embeddings, h, w, True)
            self.cache_position_count[grid] = 1
            self.cache_position_embedding[grid] = position_embedding
            return position_embedding

        def forward(
            self,
            pixel_values: torch.FloatTensor,
            position_ids: Optional[torch.Tensor] = None,
            image_grid_thw: Optional[
                list[
                    Union[
                        tuple[int, int, int],
                        list[tuple[int, int, int]],
                    ]
                ]
            ] = None,
            interpolate_pos_encoding=False,
        ) -> torch.Tensor:
            if pixel_values.dim() == 4:
                pixel_values = pixel_values.unsqueeze(0)
            if pixel_values.dim() == 5:
                if position_ids is None:
                    raise ValueError(
                        "position_ids cannot be None when pixel_values.dim() is 5."
                    )
                (
                    batch_size,
                    squence_len,
                    channel,
                    height,
                    width,
                ) = pixel_values.shape
                target_dtype = self.patch_embedding.weight.dtype
                pixel_values = rearrange(pixel_values, "b l c h w -> (b l) c h w")
                patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))
                embeddings = patch_embeds.flatten(-2).squeeze(-1)

                if interpolate_pos_encoding and image_grid_thw is not None:
                    start = 0
                    tmp_embeddings = list()
                    for image_grid in image_grid_thw:
                        t, h, w = image_grid
                        end = start + t * h * w
                        image_embeddings = embeddings[start:end, :]
                        position_embedding = (
                            self.interpolate_pos_encoding(image_embeddings, h, w, True)
                            .squeeze(0)
                            .repeat(t, 1)
                        )
                        image_embeddings = image_embeddings + position_embedding
                        tmp_embeddings.append(image_embeddings)
                        start = end
                    embeddings = torch.concat(tmp_embeddings, dim=0).unsqueeze(0)
                else:
                    embeddings = embeddings + self.packing_position_embedding(
                        position_ids
                    )
                return embeddings
            else:
                raise ValueError(
                    "Unsupported pixel_values dimension:"
                    f" {pixel_values.dim()}. Expected 4 or 5."
                )

    def apply_rotary_pos_emb_flashatt(
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos = cos.chunk(2, dim=-1)[0].contiguous()
        sin = sin.chunk(2, dim=-1)[0].contiguous()

        from vllm.vllm_flash_attn.layers.rotary import apply_rotary_emb

        q_embed = apply_rotary_emb(q.float(), cos.float(), sin.float()).type_as(q)
        k_embed = apply_rotary_emb(k.float(), cos.float(), sin.float()).type_as(k)
        return q_embed, k_embed

    class SiglipAttention(nn.Module):
        """Multi-headed attention from 'Attention Is All You
        Need' paper."""

        def __init__(
            self,
            config,
            quant_config: Optional[QuantizationConfig] = None,
            prefix: str = "",
        ):
            super().__init__()
            self.config = config

            hidden_size = config.hidden_size
            self.hidden_size = config.hidden_size
            tp_size = get_tensor_model_parallel_world_size()
            self.total_num_heads = config.num_attention_heads
            assert self.total_num_heads % tp_size == 0
            self.num_heads = self.total_num_heads // tp_size
            self.total_num_kv_heads = config.num_attention_heads
            if self.total_num_kv_heads >= tp_size:
                assert self.total_num_kv_heads % tp_size == 0
            else:
                assert tp_size % self.total_num_kv_heads == 0
            self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
            self.head_dim = config.hidden_size // self.total_num_heads
            self.q_size = self.num_heads * self.head_dim
            self.kv_size = self.num_kv_heads * self.head_dim
            self.scale = self.head_dim**-0.5

            self.qkv_proj = QKVParallelLinear(
                hidden_size,
                self.head_dim,
                self.total_num_heads,
                self.total_num_kv_heads,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.qkv_proj",
            )
            self.out_proj = RowParallelLinear(
                input_size=hidden_size,
                output_size=hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.out_proj",
            )

            # TODO: Detect attention implementation.
            self.attn_backend = _Backend.XFORMERS
            # self.attn_backend: _Backend = get_vit_attn_backend(support_fa=True)
            # if self.attn_backend not in {_Backend.XFORMERS}:
            #     raise RuntimeError(
            #         f"Keye-VL does not support {self.attn_backend} backend now."
            #     )

        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = False,
            cu_seqlens: Optional[list[torch.Tensor]] = None,
            rope_emb: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        ) -> torch.Tensor:
            qkv, _ = self.qkv_proj(hidden_states)
            q, k, v = qkv.split(
                [self.q_size, self.kv_size, self.kv_size],
                dim=-1,
            )

            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
            seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
            batch_size = q.shape[0]

            if rope_emb is None:
                q = q.view(*q.shape[:-1], self.num_heads, self.head_dim)
                k = k.view(
                    *k.shape[:-1],
                    self.num_kv_heads,
                    self.head_dim,
                )
                v = v.view(
                    *v.shape[:-1],
                    self.num_kv_heads,
                    self.head_dim,
                )
            else:
                if cu_seqlens is None:
                    raise ValueError(
                        "cu_seqlens cannot be None when rope_emb is not None."
                    )
                cos, sin = rope_emb
                q = q.view(*q.shape[:-1], self.num_heads, self.head_dim)
                k = k.view(
                    *k.shape[:-1],
                    self.num_kv_heads,
                    self.head_dim,
                )
                q, k = apply_rotary_pos_emb_flashatt(q, k, cos, sin)
                v = v.view(
                    *v.shape[:-1],
                    self.num_kv_heads,
                    self.head_dim,
                )

            if self.attn_backend == _Backend.XFORMERS:
                from xformers import ops as xops
                from xformers.ops.fmha.attn_bias import BlockDiagonalMask

                attn_bias = BlockDiagonalMask.from_seqlens(
                    q_seqlen=seqlens, kv_seqlen=None, device=q.device
                )

                context_layer = xops.memory_efficient_attention_forward(
                    q, k, v, attn_bias=attn_bias, p=0, scale=None
                )

            context_layer = rearrange(
                context_layer, "b s h d -> b s (h d)"
            ).contiguous()

            output, _ = self.out_proj(context_layer)
            return output

    class SigLIPRotaryEmbedding(nn.Module):

        def __init__(self, dim: int, theta: float = 10000.0) -> None:
            super().__init__()
            self.dim = dim
            self.theta = theta
            self.rope_init()

        def rope_init(self):
            inv_freq = 1.0 / (
                self.theta
                ** (torch.arange(0, self.dim, 2, dtype=torch.float) / self.dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        def forward(self, seqlen: int) -> torch.Tensor:
            seq = torch.arange(
                seqlen,
                device=self.inv_freq.device,
                dtype=self.inv_freq.dtype,
            )
            freqs = torch.outer(seq, self.inv_freq)
            return freqs

    class SiglipMLP(nn.Module):

        def __init__(
            self,
            config,
            quant_config: Optional[QuantizationConfig] = None,
            prefix: str = "",
        ) -> None:
            super().__init__()

            self.config = config
            self.activation_fn = get_act_fn(config.hidden_act)
            # Special handling for BNB and torchao quantization
            if quant_config and quant_config.get_name() in ["bitsandbytes", "torchao"]:
                quantizable = True
            else:
                # For other quantization, we require the hidden size to be a
                # multiple of 64
                quantizable = (
                    config.hidden_size % 64 == 0 and config.intermediate_size % 64 == 0
                )
            self.fc1 = ColumnParallelLinear(
                config.hidden_size,
                config.intermediate_size,
                quant_config=quant_config if quantizable else None,
                prefix=f"{prefix}.fc1",
            )
            self.fc2 = RowParallelLinear(
                config.intermediate_size,
                config.hidden_size,
                quant_config=quant_config if quantizable else None,
                prefix=f"{prefix}.fc2",
            )

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            hidden_states, _ = self.fc1(hidden_states)
            hidden_states = self.activation_fn(hidden_states)
            hidden_states, _ = self.fc2(hidden_states)
            return hidden_states

    class SiglipEncoderLayer(nn.Module):

        def __init__(
            self,
            config,
            quant_config: Optional[QuantizationConfig] = None,
            prefix: str = "",
        ):
            super().__init__()
            self.embed_dim = config.hidden_size
            self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
            self.self_attn = SiglipAttention(
                config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
            )
            self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
            self.mlp = SiglipMLP(
                config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )

        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor,
            output_attentions: Optional[bool] = False,
            cu_seqlens: Optional[list[torch.Tensor]] = None,
            rope_emb: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        ) -> tuple[torch.FloatTensor]:

            residual = hidden_states

            hidden_states = self.layer_norm1(hidden_states)
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                cu_seqlens=cu_seqlens,
                rope_emb=rope_emb,
            )

            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = self.layer_norm2(hidden_states)
            hidden_states = self.mlp(hidden_states)

            hidden_states = residual + hidden_states

            return hidden_states

    class SiglipEncoder(nn.Module):

        def __init__(
            self,
            config,
            quant_config: Optional[QuantizationConfig] = None,
            prefix: str = "",
        ):
            super().__init__()
            self.config = config
            embed_dim = config.hidden_size
            num_heads = config.num_attention_heads
            head_dim = embed_dim // num_heads
            self.layers = nn.ModuleList(
                [
                    SiglipEncoderLayer(
                        config,
                        quant_config=quant_config,
                        prefix=f"{prefix}.layers.{layer_idx}",
                    )
                    for layer_idx in range(config.num_hidden_layers)
                ]
            )
            self.rotary_pos_emb = SigLIPRotaryEmbedding(head_dim // 2)

        @staticmethod
        def flatten_list(image_grid_thw):
            tmp_image_grid_thw = list()
            for image_grid in image_grid_thw:
                if isinstance(image_grid, list):
                    tmp_image_grid_thw.extend(image_grid)
                else:
                    tmp_image_grid_thw.append(image_grid)
            return tmp_image_grid_thw

        def forward(
            self,
            inputs_embeds,
            attention_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            cu_seqlens: Optional[list[torch.Tensor]] = None,
            image_grid_thw: Optional[
                list[
                    Union[
                        tuple[int, int, int],
                        list[tuple[int, int, int]],
                    ]
                ]
            ] = None,
            height_position_ids: Optional[torch.Tensor] = None,
            width_position_ids: Optional[torch.Tensor] = None,
            use_rope: Optional[bool] = False,
            window_size: Optional[bool] = -1,
            vision_or_text: str = "vision",
        ) -> BaseModelOutput:
            device = inputs_embeds.device
            hidden_states = inputs_embeds
            if use_rope is True:
                flatten_image_grid_thw = self.flatten_list(image_grid_thw)

                if width_position_ids is None or height_position_ids is None:
                    split_hids = list()
                    split_wids = list()
                    for t, h, w in flatten_image_grid_thw:
                        image_pids = torch.arange(t * h * w, device=device) % (h * w)
                        sample_hids = image_pids // w
                        sample_wids = image_pids % w
                        split_hids.append(sample_hids)
                        split_wids.append(sample_wids)
                    width_position_ids = torch.concat(split_wids, dim=0)
                    height_position_ids = torch.concat(split_hids, dim=0)

                pids = torch.stack(
                    [height_position_ids, width_position_ids],
                    dim=-1,
                )
                max_grid_size = pids.max() + 1
                rope_emb_max_grid = self.rotary_pos_emb(max_grid_size)
                rope_emb = rope_emb_max_grid[pids].flatten(1)
                rope_emb = rope_emb.repeat(1, 2)
                rope_emb = (rope_emb.cos(), rope_emb.sin())
            else:
                rope_emb = None

            attn_cu_seqlens = cu_seqlens
            hidden_states = inputs_embeds
            assert attention_mask is None

            for encoder_layer in self.layers:
                hidden_states = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=output_attentions,
                    cu_seqlens=attn_cu_seqlens,
                    rope_emb=rope_emb,
                )
            return hidden_states

    class SiglipVisionTransformer(nn.Module):

        def __init__(
            self,
            config,
            quant_config: Optional[QuantizationConfig] = None,
            prefix: str = "",
        ):
            super().__init__()
            self.config = config
            embed_dim = config.hidden_size

            self.embeddings = SiglipVisionEmbeddings(config)
            self.encoder = SiglipEncoder(
                config,
                quant_config=quant_config,
                prefix=f"{prefix}.encoder",
            )
            self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

        def forward(
            self,
            pixel_values,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            interpolate_pos_encoding: Optional[bool] = False,
            attention_mask: Optional[torch.Tensor] = None,
            sample_indices: Optional[torch.Tensor] = None,
            image_indices: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            height_position_ids: Optional[torch.Tensor] = None,
            width_position_ids: Optional[torch.Tensor] = None,
            cu_seqlens: Optional[list[torch.Tensor]] = None,
            padding_mask: Optional[torch.Tensor] = None,
            vision_return_embed_list: Optional[bool] = False,
            image_grid_thw: Optional[
                list[
                    Union[
                        tuple[int, int, int],
                        list[tuple[int, int, int]],
                    ]
                ]
            ] = None,
            return_pooler_output: Optional[bool] = True,
            use_rope: Optional[bool] = False,
            window_size: Optional[bool] = -1,
        ) -> BaseModelOutputWithPooling:

            hidden_states = self.embeddings(
                pixel_values,
                interpolate_pos_encoding=interpolate_pos_encoding,
                position_ids=position_ids,
                image_grid_thw=image_grid_thw,
            )

            last_hidden_state = self.encoder(
                inputs_embeds=hidden_states,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                attention_mask=attention_mask,
                cu_seqlens=cu_seqlens,
                image_grid_thw=image_grid_thw,
                use_rope=use_rope,
                height_position_ids=height_position_ids,
                width_position_ids=width_position_ids,
                window_size=window_size,
                vision_or_text="vision",
            )

            last_hidden_state = self.post_layernorm(last_hidden_state)

            sample_hidden_state = list()
            if cu_seqlens is None:
                raise ValueError(
                    "cu_seqlens cannot be None for "
                    "SiglipVisionTransformer output processing."
                )
            for i in range(cu_seqlens.shape[0] - 1):
                start = cu_seqlens[i]
                end = cu_seqlens[i + 1]
                tensor = last_hidden_state[:, start:end, :].squeeze(0)
                sample_hidden_state.append(tensor)

            return sample_hidden_state

    class SiglipVisionModel(nn.Module):
        config_class = "PPOCRVisionConfig"
        main_input_name = "pixel_values"

        def __init__(
            self,
            config,
            quant_config: Optional[QuantizationConfig] = None,
            prefix: str = "",
        ):
            super().__init__()

            self.vision_model = SiglipVisionTransformer(
                config,
                quant_config=quant_config,
                prefix=f"{prefix}.vision_model",
            )
            self.quant_config = quant_config

        @property
        def dtype(self) -> torch.dtype:
            return self.vision_model.embeddings.patch_embedding.weight.dtype

        @property
        def device(self) -> torch.device:
            return self.vision_model.embeddings.patch_embedding.weight.device

        def get_input_embeddings(self) -> nn.Module:
            return self.vision_model.embeddings.patch_embedding

        def forward(
            self,
            pixel_values,
            sample_indices: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            interpolate_pos_encoding: bool = False,
            position_ids: Optional[torch.Tensor] = None,
            vision_return_embed_list: Optional[bool] = False,
            image_grid_thw: Optional[
                list[
                    Union[
                        tuple[int, int, int],
                        list[tuple[int, int, int]],
                    ]
                ]
            ] = None,
            cu_seqlens: Optional[list[torch.Tensor]] = None,
            return_pooler_output: Optional[bool] = True,
            use_rope: Optional[bool] = False,
            window_size: Optional[bool] = -1,
        ) -> BaseModelOutputWithPooling:

            return self.vision_model(
                pixel_values=pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                interpolate_pos_encoding=interpolate_pos_encoding,
                position_ids=position_ids,
                vision_return_embed_list=vision_return_embed_list,
                image_grid_thw=image_grid_thw,
                sample_indices=sample_indices,
                cu_seqlens=cu_seqlens,
                return_pooler_output=return_pooler_output,
                use_rope=use_rope,
                window_size=window_size,
            )

        def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
            stacked_params_mapping = [
                ("qkv_proj", "q_proj", "q"),
                ("qkv_proj", "k_proj", "k"),
                ("qkv_proj", "v_proj", "v"),
            ]
            params_dict = dict(self.named_parameters(remove_duplicate=False))
            loaded_params: set[str] = set()
            for name, loaded_weight in weights:
                if "rotary_emb.inv_freq" in name:
                    continue
                if "head.attention" in name or "head.layernorm" in name:
                    continue
                if "head.mlp" in name or "head.probe" in name:
                    continue
                if self.quant_config is not None and (
                    scale_name := self.quant_config.get_cache_scale(name)
                ):
                    param = params_dict[scale_name]
                    weight_loader = getattr(
                        param,
                        "weight_loader",
                        default_weight_loader,
                    )
                    loaded_weight = (
                        loaded_weight if loaded_weight.dim() == 0 else loaded_weight[0]
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(scale_name)
                    continue
                for (
                    param_name,
                    weight_name,
                    shard_id,
                ) in stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if is_pp_missing_parameter(name, self):
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id)
                    break
                else:
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue
                    if is_pp_missing_parameter(name, self):
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param,
                        "weight_loader",
                        default_weight_loader,
                    )
                    weight_loader(param, loaded_weight)
                loaded_params.add(name)
            return loaded_params

    @MULTIMODAL_REGISTRY.register_processor(
        PPOCRVLMultiModalProcessor,
        info=PPOCRVLProcessingInfo,
        dummy_inputs=PPOCRVLDummyInputsBuilder,
    )
    @support_torch_compile(
        # set dynamic_arg_dims to support mrope
        dynamic_arg_dims={
            "input_ids": 0,
            "positions": -1,
            "intermediate_tensors": 0,
            "inputs_embeds": 0,
        }
    )
    class PPOCRVLForConditionalGeneration(Ernie4_5_ForCausalLM, SupportsMultiModal):

        def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
            super().__init__(vllm_config=vllm_config, prefix=prefix)
            config = self.config

            self.mlp_AR = Projector(config, config.vision_config)
            self.visual = SiglipVisionModel(config=config.vision_config)
            self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
            self.logits_processor = LogitsProcessor(config.vocab_size)

            for layer in self.model.layers:
                if not isinstance(layer, PPMissingLayer):
                    layer.self_attn.rotary_emb.is_neox_style = True

        def compute_logits(
            self,
            hidden_states: torch.Tensor,
            sampling_metadata,
        ) -> Optional[torch.Tensor]:
            logits = self.logits_processor(
                self.lm_head, hidden_states, sampling_metadata
            )
            return logits

        @property
        def language_model(self):
            return self.model

        def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            intermediate_tensors: Optional[IntermediateTensors] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            **kwargs,
        ):
            if intermediate_tensors is not None:
                inputs_embeds = None

            elif inputs_embeds is None:
                vision_embeddings = self.get_multimodal_embeddings(**kwargs)
                inputs_embeds = self.get_input_embeddings(input_ids, vision_embeddings)
                input_ids = None

            return self.language_model(
                input_ids, positions, intermediate_tensors, inputs_embeds
            )

        @classmethod
        def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
            if modality.startswith("image"):
                return "<|vision_start|><|image_pad|><|vision_end|>"

            raise ValueError("Only image modality is supported")

        def encode_image(self, pixel_values, image_grid_thw):
            pixel_values = pixel_values.type(self.visual.dtype)
            siglip_position_ids = list()
            image_grid_hws = list()
            sample_indices = list()
            cu_seqlens = [0]

            for idx, thw in enumerate(image_grid_thw):
                thw_tuple = tuple(thw.detach().cpu().numpy().tolist())
                numel = np.prod(thw_tuple)
                image_grid_hws.append(thw_tuple)
                image_position_ids = torch.arange(numel) % np.prod(thw_tuple[1:])
                siglip_position_ids.append(image_position_ids)
                sample_indices.append(torch.full((numel,), idx, dtype=torch.int64))
                cu_seqlens.append(cu_seqlens[-1] + numel)

            siglip_position_ids = torch.concat(siglip_position_ids, dim=0).to(
                pixel_values.device
            )
            cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32).to(
                pixel_values.device
            )
            sample_indices = torch.concat(sample_indices, dim=0).to(pixel_values.device)

            vision_outputs = self.visual(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_hws,
                position_ids=siglip_position_ids,
                vision_return_embed_list=True,
                interpolate_pos_encoding=True,
                sample_indices=sample_indices,
                cu_seqlens=cu_seqlens,
                return_pooler_output=False,
                use_rope=True,
                window_size=-1,
            )
            image_embeds = self.mlp_AR(vision_outputs, image_grid_thw)

            return image_embeds

        def get_multimodal_embeddings(self, **kwargs):
            pixel_values = kwargs["pixel_values"]
            image_grid_thw = kwargs["image_grid_thw"]

            multimodal_embeddings = []
            for pv, ig in zip(pixel_values, image_grid_thw):
                if pv is not None:
                    image_embeds = self.encode_image(pv, ig)
                    multimodal_embeddings += image_embeds

            return multimodal_embeddings

        def get_input_embeddings(
            self,
            input_ids: torch.Tensor,
            multimodal_embeddings: Optional[NestedTensors] = None,
        ) -> torch.Tensor:
            inputs_embeds = self.language_model.get_input_embeddings(input_ids)

            if multimodal_embeddings is not None and len(multimodal_embeddings) != 0:
                inputs_embeds = merge_multimodal_embeddings(
                    input_ids,
                    inputs_embeds,
                    multimodal_embeddings,
                    self.config.image_token_id,
                )

            return inputs_embeds

        def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:

            loader = AutoWeightsLoader(self)
            autoloaded_weights = loader.load_weights(weights)
            return autoloaded_weights
