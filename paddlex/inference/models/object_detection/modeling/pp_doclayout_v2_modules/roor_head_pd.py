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

import paddle
import paddle.nn as nn

INF = 1e4


class RotaryPositionEmbeddingPD(nn.Layer):
    def __init__(self, dim, max_seq_len=1024):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (paddle.arange(0, dim, 2, dtype="float32") / dim))
        t = paddle.arange(max_seq_len, dtype="float32")
        freqs = paddle.einsum("n,d->nd", t, inv_freq)
        self.register_buffer("sin", paddle.sin(freqs), persistable=False)
        self.register_buffer("cos", paddle.cos(freqs), persistable=False)

    def forward(self, x, seqlen, seq_axis=-2):
        nd = x.ndim
        if seq_axis < 0:
            seq_axis = nd + seq_axis
        if seq_axis != nd - 2:
            perm = list(range(nd))
            perm[seq_axis], perm[-2] = perm[-2], perm[seq_axis]
            x = x.transpose(perm)

        Dh = x.shape[-1]
        x1, x2 = x[..., 0::2], x[..., 1::2]
        sin = self.sin[:seqlen].reshape([1] * (nd - 2) + [seqlen, Dh // 2])
        cos = self.cos[:seqlen].reshape([1] * (nd - 2) + [seqlen, Dh // 2])

        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        y = paddle.stack([y1, y2], axis=-1).reshape(x.shape)

        if seq_axis != nd - 2:
            inv = list(range(nd))
            inv[seq_axis], inv[-2] = inv[-2], inv[seq_axis]
            y = y.transpose(inv)
        return y


class GlobalPointerPD(nn.Layer):
    def __init__(
        self,
        hidden_size,
        heads=1,
        head_size=64,
        use_rope=True,
        tril_mask=False,
        max_length=1024,
    ):
        super().__init__()
        self.heads = heads
        self.head_size = head_size
        self.use_rope = use_rope
        self.tril_mask = tril_mask
        self.dense = nn.Linear(hidden_size, heads * 2 * head_size)
        self.rotary = (
            RotaryPositionEmbeddingPD(head_size, max_length) if use_rope else None
        )

    def forward(self, inputs, attn_mask_1d):
        B, N, _ = inputs.shape
        proj = self.dense(inputs).reshape([B, N, self.heads, 2, self.head_size])
        qw, kw = proj[..., 0, :], proj[..., 1, :]

        if self.use_rope:
            qw = self.rotary(qw, N, seq_axis=1)
            kw = self.rotary(kw, N, seq_axis=1)

        qw_t = qw.transpose([0, 2, 1, 3])
        kw_t = kw.transpose([0, 2, 1, 3])
        logits = paddle.einsum("bhmd,bhnd->bhmn", qw_t, kw_t) / (self.head_size**0.5)

        a = attn_mask_1d.astype("float32")
        pair_mask = 1.0 - (a.unsqueeze(1).unsqueeze(2) * a.unsqueeze(1).unsqueeze(3))
        logits = logits - pair_mask * INF

        if self.tril_mask:
            lower = paddle.tril(paddle.ones([N, N], dtype="float32"))
            lower = lower.astype("bool").unsqueeze(0).unsqueeze(0)
            logits = logits - lower.astype(logits.dtype) * INF
            pair_mask = paddle.logical_or(pair_mask.astype("bool"), lower)

        return logits, pair_mask.astype("bool")
