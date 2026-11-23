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

"""Fused PaddleOCR-VL vision RoPE with no vLLM dependency."""

from .....utils.deps import is_dep_available

if not all(map(is_dep_available, ("torch", "triton"))):
    raise ImportError("Vision RoPE requires Torch and Triton.")
else:
    import torch
    import triton
    import triton.language as tl


@triton.jit
def _rotate_qk(
    qkv,
    cos,
    sin,
    size,
    stride,
    HALF: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_J: tl.constexpr,
):
    s = tl.program_id(0) * BLOCK_S + tl.arange(0, BLOCK_S)[:, None]
    j = tl.arange(0, BLOCK_J)[None, :]
    head, dim = j // HALF, j % HALF
    mask = (s < size) & (j < HEADS * HALF)
    c = tl.load(cos + s * HALF + dim, mask, other=0.0).to(tl.float32)
    sn = tl.load(sin + s * HALF + dim, mask, other=0.0).to(tl.float32)
    for part in tl.static_range(2):
        p = qkv + s * stride + part * HEADS * DIM + head * DIM + dim
        x = tl.load(p, mask, other=0.0).to(tl.float32)
        y = tl.load(p + HALF, mask, other=0.0).to(tl.float32)
        tl.store(p, x * c - y * sn, mask)
        tl.store(p + HALF, y * c + x * sn, mask)


def rotate_qk_inplace(qkv, cos, sin, heads, dim):
    """Rotate packed Q/K in FP32 and leave V intact; return False for other layouts."""
    if (
        qkv.ndim != 3
        or qkv.shape[0] == 0
        or qkv.shape[1:] != (1, 3 * heads * dim)
        or qkv.dtype != torch.bfloat16
        or not qkv.is_cuda
        or qkv.stride(-1) != 1
        or qkv.stride(0) < qkv.shape[-1]
        or heads <= 0
        or dim <= 0
        or dim % 2
        or any(
            x.shape != (qkv.shape[0], dim // 2)
            or x.dtype != torch.float32
            or x.device != qkv.device
            or not x.is_contiguous()
            for x in (cos, sin)
        )
    ):
        return False
    seq = qkv.shape[0]
    _rotate_qk[(triton.cdiv(seq, 8),)](
        qkv,
        cos,
        sin,
        seq,
        qkv.stride(0),
        HALF=dim // 2,
        HEADS=heads,
        DIM=dim,
        BLOCK_S=8,
        BLOCK_J=triton.next_power_of_2(heads * dim // 2),
        num_warps=8,
        enable_fp_fusion=False,
    )
    return True
