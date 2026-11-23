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

"""Opt-in vision RoPE adapter for vLLM's integrated PaddleOCR-VL.

Based on vLLM's Apache-2.0 PaddleOCR-VL implementation (vLLM contributors).
"""

import inspect

from .....utils import logging
from .....utils.deps import is_dep_available

if not all(map(is_dep_available, ("torch", "vllm", "triton"))):
    raise ImportError("Vision RoPE requires Torch, vLLM, and Triton.")
else:
    import torch
    from vllm.model_executor.models.paddleocr_vl import (
        PaddleOCRVLDummyInputsBuilder,
        PaddleOCRVLForConditionalGeneration,
        PaddleOCRVLMultiModalProcessor,
        PaddleOCRVLProcessingInfo,
        SiglipAttention,
        SiglipEncoder,
    )
    from vllm.multimodal import MULTIMODAL_REGISTRY

from ..paddleocr_vl_09b._vllm_batching import (
    PaddleXVisionBatchedPaddleOCRVLForConditionalGeneration,
)
from ._vision_rope import rotate_qk_inplace


class _FusedSiglipAttention(SiglipAttention):
    def forward(self, hidden_states, *, cu_seqlens, rotary_pos_emb, max_seqlen):
        if not isinstance(rotary_pos_emb, tuple):
            return super().forward(
                hidden_states,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb=rotary_pos_emb,
                max_seqlen=max_seqlen,
            )
        cos, sin = rotary_pos_emb
        x, _ = self.qkv_proj(hidden_states.transpose(0, 1))
        fused = rotate_qk_inplace(
            x,
            cos,
            sin,
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
        )
        q, k, v = (t.transpose(0, 1) for t in self.split_qkv(x))
        if not fused:
            q, k = self.apply_rotary_emb(torch.cat((q, k), dim=0), cos, sin).chunk(2)
        context = self.attn(
            query=q, key=k, value=v, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
        )
        output, _ = self.out_proj(context.flatten(-2))
        return output


def _accepts_keywords(fn, *names):
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return all(
        name in params
        and params[name].kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        for name in names
    )


def _vision_support(vllm_config, encoder):
    parallel = vllm_config.parallel_config
    if (
        torch.version.cuda is None
        or not torch.cuda.is_available()
        or torch.cuda.get_device_capability() != (8, 0)
        or vllm_config.model_config.dtype != torch.bfloat16
        or vllm_config.quant_config is not None
        or parallel.tensor_parallel_size != 1
        or parallel.pipeline_parallel_size != 1
        or parallel.data_parallel_size != 1
        or getattr(parallel, "enable_expert_parallel", False)
        or getattr(getattr(encoder, "attn_backend", None), "name", None) != "FLASH_ATTN"
    ):
        return (
            "requires one SM80 GPU, BF16, unquantized weights and vision FlashAttention"
        )
    if not isinstance(encoder, SiglipEncoder) or not (
        _accepts_keywords(
            encoder.forward,
            "inputs_embeds",
            "cu_seqlens",
            "image_grid_thw",
            "height_position_ids",
            "width_position_ids",
        )
        and getattr(encoder, "layers", None)
        and callable(getattr(encoder, "flatten_list", None))
        and callable(getattr(encoder, "rotary_pos_emb", None))
        and all(
            type(getattr(layer, "self_attn", None)) is SiglipAttention
            and _accepts_keywords(
                layer.forward, "cu_seqlens", "rotary_pos_emb", "max_seqlen"
            )
            for layer in encoder.layers
        )
    ):
        return "incompatible vLLM vision interface"
    return None


def _encode_vision_layers(encoder, inputs_embeds, cu_seqlens, image_grid_thw):
    device = inputs_embeds.device
    grids = encoder.flatten_list(image_grid_thw)
    split_hids, split_wids = [], []
    for t, h, w in grids:
        image_pids = torch.arange(t * h * w, device=device) % (h * w)
        split_hids.append(image_pids // w)
        split_wids.append(image_pids % w)
    height_position_ids = torch.cat(split_hids)
    width_position_ids = torch.cat(split_wids)
    pids = torch.stack(
        [height_position_ids, width_position_ids],
        dim=-1,
    )
    # The ids are built from the grids above, so `h`/`w` bound them
    # and the table size is known on the host.
    max_grid_size = max(max(h, w) for _, h, w in grids)
    rope_emb_max_grid = encoder.rotary_pos_emb(max_grid_size)
    rotary_pos_emb = rope_emb_max_grid[pids].flatten(1)
    if cu_seqlens is None:
        raise ValueError("cu_seqlens cannot be None for SiglipEncoder.")
    if not isinstance(cu_seqlens, torch.Tensor):
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32)
    # Computed before the H2D copy: the attention wrappers call `.item()`
    # on it, which would sync on every layer for a device tensor.
    max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
    if cu_seqlens.is_cpu and device.type == "cuda" and not cu_seqlens.is_pinned():
        cu_seqlens = cu_seqlens.pin_memory()
    cu_seqlens = cu_seqlens.to(device=device, non_blocking=True)
    # Reuse the same FP32 phase tensors within this forward only.
    rotary_pos_emb = (
        rotary_pos_emb.cos().contiguous(),
        rotary_pos_emb.sin().contiguous(),
    )
    hidden_states = inputs_embeds
    for encoder_layer in encoder.layers:
        hidden_states = encoder_layer(
            hidden_states,
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            max_seqlen=max_seqlen,
        )
    return hidden_states


class _PaddleXSiglipEncoder(SiglipEncoder):
    def forward(
        self,
        inputs_embeds,
        cu_seqlens=None,
        image_grid_thw=None,
        height_position_ids=None,
        width_position_ids=None,
    ):
        if (
            inputs_embeds.dtype != torch.bfloat16
            or inputs_embeds.ndim != 3
            or inputs_embeds.shape[0] != 1
            or not inputs_embeds.is_cuda
            or image_grid_thw is None
            or height_position_ids is not None
            or width_position_ids is not None
            or torch.compiler.is_compiling()
        ):
            return super().forward(
                inputs_embeds,
                cu_seqlens,
                image_grid_thw,
                height_position_ids,
                width_position_ids,
            )
        return _encode_vision_layers(
            self,
            inputs_embeds,
            cu_seqlens,
            image_grid_thw,
        )


class _VisionRoPEMixin:
    def __init__(self, *, vllm_config, prefix=""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        encoder = getattr(getattr(self.visual, "vision_model", None), "encoder", None)
        reason = _vision_support(vllm_config, encoder)
        if reason:
            raise ValueError(f"PaddleX vision RoPE unavailable: {reason}.")
        # Mutate only these instances. No global vLLM patch or new batching path.
        for layer in encoder.layers:
            layer.self_attn.__class__ = _FusedSiglipAttention
        encoder.__class__ = _PaddleXSiglipEncoder
        logging.info("PaddleX fused vision RoPE enabled (SM80 BF16)")


@MULTIMODAL_REGISTRY.register_processor(
    PaddleOCRVLMultiModalProcessor,
    info=PaddleOCRVLProcessingInfo,
    dummy_inputs=PaddleOCRVLDummyInputsBuilder,
)
class PaddleXVisionRoPEPaddleOCRVLForConditionalGeneration(
    _VisionRoPEMixin, PaddleOCRVLForConditionalGeneration
):
    pass


@MULTIMODAL_REGISTRY.register_processor(
    PaddleOCRVLMultiModalProcessor,
    info=PaddleOCRVLProcessingInfo,
    dummy_inputs=PaddleOCRVLDummyInputsBuilder,
)
class PaddleXVisionBatchedRoPEPaddleOCRVLForConditionalGeneration(
    _VisionRoPEMixin, PaddleXVisionBatchedPaddleOCRVLForConditionalGeneration
):
    pass
