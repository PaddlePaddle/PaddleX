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

import os

import paddle
import paddle.nn.functional as F

try:
    from paddle.incubate.nn.functional import fused_rotary_position_embedding
except ImportError:
    fused_rotary_position_embedding = None

try:
    from paddle.incubate.nn.functional import swiglu
except ImportError:

    def swiglu(x, y=None):
        if y is None:
            x, y = paddle.chunk(x, chunks=2, axis=-1)
        return F.silu(x) * y


from paddle.utils import try_import


def get_env_device():
    """
    Return the device name of running environment.
    """
    if paddle.is_compiled_with_cuda():
        return "gpu"
    elif "npu" in paddle.device.get_all_custom_device_type():
        return "npu"
    elif "mlu" in paddle.device.get_all_custom_device_type():
        return "mlu"
    elif "gcu" in paddle.device.get_all_custom_device_type():
        return "gcu"
    elif "intel_hpu" in paddle.device.get_all_custom_device_type():
        return "intel_hpu"
    elif "iluvatar_gpu" in paddle.device.get_all_custom_device_type():
        return "iluvatar_gpu"
    elif paddle.is_compiled_with_rocm():
        return "rocm"
    elif paddle.is_compiled_with_xpu():
        return "xpu"
    return "cpu"


try:
    from paddle.incubate.nn.functional import fused_rotary_position_embedding
except ImportError:
    fused_rotary_position_embedding = None
try:
    if get_env_device() in ["npu", "mlu", "gcu", "iluvatar_gpu"]:
        from paddle.base import core

        for lib in os.listdir(os.getenv("CUSTOM_DEVICE_ROOT")):
            if lib.endswith(".so"):
                paddle.utils.cpp_extension.extension_utils.load_op_meta_info_and_register_op(
                    lib
                )
    from paddle.nn.functional.flash_attention import flash_attention
except:
    flash_attention = None


def fusion_rope(
    query_states,
    key_states,
    value_states,
    hidden_states,
    position_ids,
    past_key_value,
    rotary_emb,
    context_parallel_degree=-1,
):
    if get_env_device() not in ["gcu", "intel_hpu", "iluvatar_gpu"]:
        assert past_key_value is None, "fuse rotary not support cache kv for now"
    batch_size, seq_length, num_heads, head_dim = query_states.shape
    _, kv_seq_len, num_key_value_heads, _ = key_states.shape
    if context_parallel_degree > 1:
        assert (
            get_env_device() == "gpu"
        ), "context parallel only support cuda device for now"
        kv_seq_len *= context_parallel_degree
    if get_env_device() not in ["gcu", "intel_hpu", "iluvatar_gpu"]:
        cos, sin = rotary_emb(value_states, seq_len=kv_seq_len)
    if get_env_device() == "npu":
        query_states = core.eager._run_custom_op("fused_rope", query_states, cos, sin)[
            0
        ]
        key_states = core.eager._run_custom_op("fused_rope", key_states, cos, sin)[0]
    elif get_env_device() == "intel_hpu":
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-3]
        cos, sin = rotary_emb(value_states, seq_len=kv_seq_len)
        cos = cos.squeeze().unsqueeze(0).unsqueeze(0)
        sin = sin.squeeze().unsqueeze(0).unsqueeze(0)
        query_states, _, _ = (
            paddle.incubate.nn.functional.fused_rotary_position_embedding(
                paddle.transpose(query_states, [0, 2, 1, 3]),
                None,
                None,
                sin=sin,
                cos=cos,
                position_ids=position_ids,
            )
        )
        key_states, _, _ = (
            paddle.incubate.nn.functional.fused_rotary_position_embedding(
                paddle.transpose(key_states, [0, 2, 1, 3]),
                None,
                None,
                sin=sin,
                cos=cos,
                position_ids=position_ids,
            )
        )
        query_states = paddle.transpose(query_states, [0, 2, 1, 3])
        key_states = paddle.transpose(key_states, [0, 2, 1, 3])
    elif get_env_device() == "gcu":
        cos_sin = rotary_emb.get_fused_cos_sin(value_states, seq_len=kv_seq_len)
        query_states, key_states = core.eager._run_custom_op(
            "fused_rotary_embedding_gcu",
            query_states,
            key_states,
            cos_sin,
            position_ids,
            True,
        )
    else:
        # paddle version > 2.6 or develop support q and k/v with different num_heads
        paddle_version = float(paddle.__version__[:3])
        if ((paddle_version != 0.0) and (paddle_version <= 2.6)) and (
            num_heads != num_key_value_heads
        ):
            query_states, _, _ = fused_rotary_position_embedding(
                query_states,
                None,
                None,
                sin=sin,
                cos=cos,
                position_ids=position_ids,
                use_neox_rotary_style=False,
            )
            key_states, _, _ = fused_rotary_position_embedding(
                key_states,
                None,
                None,
                sin=sin,
                cos=cos,
                position_ids=position_ids,
                use_neox_rotary_style=False,
            )
        else:
            query_states, key_states, _ = fused_rotary_position_embedding(
                query_states,
                key_states,
                v=None,
                sin=sin,
                cos=cos,
                position_ids=position_ids,
                use_neox_rotary_style=False,
            )
    return query_states, key_states


def rms_norm_fused(x_in, w, eps, use_fast_ln=False):
    if use_fast_ln:
        fast_ln = try_import("fast_ln")
        return fast_ln.fast_rms_norm(x_in, w, eps)[0]
    else:
        fused_ln = try_import("fused_ln")
        return fused_ln.fused_rms_norm(x_in, w, eps)[0]


def fusion_rms_norm(hidden_states, weight, variance_epsilon, use_fast_ln=False):
    if get_env_device() == "npu":
        return core.eager._run_custom_op(
            "rms_norm_npu", hidden_states, weight, variance_epsilon
        )[0]
    if get_env_device() == "mlu":
        return core.eager._run_custom_op(
            "rms_norm_mlu", hidden_states, weight, variance_epsilon
        )[0]
    elif get_env_device() == "gcu":
        return core.eager._run_custom_op(
            "rms_norm_gcu", hidden_states, weight, variance_epsilon
        )[0]
    elif get_env_device() == "intel_hpu":
        return paddle.incubate.nn.functional.fused_rms_norm(
            hidden_states, weight, None, variance_epsilon, hidden_states.dim() - 1
        )[0]

    return rms_norm_fused(hidden_states, weight, variance_epsilon, use_fast_ln)
