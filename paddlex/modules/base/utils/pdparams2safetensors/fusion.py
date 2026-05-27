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

"""Reparameterization fusion for PP-OCRv6 detection pdparams.

Training-time graphs use multi-branch RepDWConv and DilatedReparamBlock
modules whose forward pass is mathematically equivalent to a single
Conv2d. The HF inference modules are written as that fused single
Conv2d, so converting a training pdparams to inference safetensors
requires folding the branches back into one (weight, bias) pair before
regex key mapping.

NumPy port of the per-model torch scripts the upstream team produced
(``convert_small_det.py`` / ``convert_medium_det.py``).
"""

import re

import numpy as np


_REP_DW_PATTERN = re.compile(
    r"^backbone\.blocks_s(\d+)\.(\d+)\.token_mixer\.rep_dw\.(.+)$"
)
# Rec models use a different prefix (``blocks{N}`` without ``_s``) and a
# different zero-based offset (``N-2`` instead of ``N-1``).
_REP_DW_PATTERN_REC = re.compile(
    r"^backbone\.blocks(\d+)\.(\d+)\.token_mixer\.rep_dw\.(.+)$"
)
_NECK_INPUT_DW_PATTERN_SMALL = re.compile(r"^neck\.inp_conv_dw\.(\d+)\.(.+)$")
_NECK_REPARAM_PATTERN_MEDIUM = re.compile(
    r"^neck\.(inp_conv|pan_lat_conv)\.(\d+)\.(.+)$"
)


def _get_bn_value(params, name):
    """Look up a BN parameter, accepting either HF or Paddle naming."""
    if name in params:
        return params[name]
    paddle_alias = {
        "running_mean": "_mean",
        "running_var": "_variance",
    }.get(name)
    if paddle_alias is not None and paddle_alias in params:
        return params[paddle_alias]
    raise KeyError(
        f"Missing BN parameter: {name!r}. Available: {sorted(params.keys())!r}"
    )


def _fuse_conv_bn(conv_w, bn_w, bn_b, bn_mean, bn_var, eps=1e-5):
    """Absorb a BatchNorm into the preceding conv.

    ``(conv_w, 0) -> bn`` becomes ``(w, b)`` where
    ``w = conv_w * (bn_w / sqrt(bn_var+eps))``
    ``b = bn_b - bn_mean * (bn_w / sqrt(bn_var+eps))``
    """
    std = np.sqrt(bn_var + eps)
    scale = bn_w / std
    w = conv_w * scale.reshape(-1, 1, 1, 1)
    b = bn_b - bn_mean * scale
    return w, b


def _fuse_rep_dw(params):
    """Fuse a RepDWConv block (3 branches) into a single depthwise Conv2d.

    The training-time block is ``conv (KxK + BN) + conv1 (1x1) + identity``,
    all summed and then passed through an outer BN. Fold into
    ``Conv2d(weight, bias)`` matching the HF inference ``token_conv``.
    """
    w, b = _fuse_conv_bn(
        params["conv.conv.weight"],
        params["conv.bn.weight"],
        params["conv.bn.bias"],
        _get_bn_value(params, "conv.bn.running_mean"),
        _get_bn_value(params, "conv.bn.running_var"),
    )

    pad = w.shape[2] // 2
    w = w + np.pad(params["conv1.weight"], ((0, 0), (0, 0), (pad, pad), (pad, pad)))

    identity = np.zeros_like(w)
    identity[:, 0, pad, pad] = 1.0
    w = w + identity

    std = np.sqrt(_get_bn_value(params, "bn.running_var") + 1e-5)
    scale = params["bn.weight"] / std
    w = w * scale.reshape(-1, 1, 1, 1)
    b = params["bn.bias"] + (b - _get_bn_value(params, "bn.running_mean")) * scale

    return w, b


def _get_dilated_reparam_branches(kernel_size):
    """Branch kernel/dilation pairs for a DilatedReparamBlock with the given
    origin kernel size.

    Values match the upstream PaddleOCR PP-OCRv6 DilatedReparamBlock
    definition.
    """
    if kernel_size == 5:
        return [(3, 1), (3, 2)]
    if kernel_size == 7:
        return [(5, 1), (3, 2), (3, 3)]
    if kernel_size == 9:
        return [(5, 1), (5, 2), (3, 3), (3, 4)]
    if kernel_size == 11:
        return [(5, 1), (5, 2), (3, 3), (3, 4), (3, 5)]
    if kernel_size == 13:
        return [(5, 1), (7, 2), (3, 3), (3, 4), (3, 5)]
    raise ValueError(f"Unsupported DilatedReparamBlock origin kernel size: {kernel_size}")


def _add_dilated_kernel(base, branch, dilation):
    """Accumulate a dilated branch kernel into the origin kernel in-place."""
    branch_k = branch.shape[2]
    center = base.shape[2] // 2
    start = center - dilation * (branch_k // 2)
    for i in range(branch_k):
        for j in range(branch_k):
            base[:, :, start + i * dilation, start + j * dilation] += branch[:, :, i, j]
    return base


def _fuse_dilated_reparam_branch_set(params, prefix=""):
    """Fuse a DilatedReparamBlock (origin + dilated branches, each BN-fused)
    into a single depthwise (conv_w, conv_b).

    ``prefix`` lets the same logic serve both ``params["lk_origin..."]``
    (small_det neck) and ``params["dw.lk_origin..."]`` (medium_det neck).
    """
    origin_w_key = f"{prefix}lk_origin.weight"
    w, b = _fuse_conv_bn(
        params[origin_w_key],
        params[f"{prefix}origin_bn.weight"],
        params[f"{prefix}origin_bn.bias"],
        _get_bn_value(params, f"{prefix}origin_bn.running_mean"),
        _get_bn_value(params, f"{prefix}origin_bn.running_var"),
    )

    origin_k = params[origin_w_key].shape[2]
    for branch_k, dilation in _get_dilated_reparam_branches(origin_k):
        conv_key = f"{prefix}dil_conv_k{branch_k}_{dilation}.weight"
        bn_prefix = f"{prefix}dil_bn_k{branch_k}_{dilation}"
        if conv_key not in params:
            raise KeyError(
                f"Expected {conv_key!r} for origin kernel {origin_k}, "
                f"but it is missing. Available: {sorted(params.keys())!r}"
            )
        branch_w, branch_b = _fuse_conv_bn(
            params[conv_key],
            params[f"{bn_prefix}.weight"],
            params[f"{bn_prefix}.bias"],
            _get_bn_value(params, f"{bn_prefix}.running_mean"),
            _get_bn_value(params, f"{bn_prefix}.running_var"),
        )
        w = _add_dilated_kernel(w, branch_w, dilation)
        b = b + branch_b
    return w, b


def _fuse_neck_input_depthwise_reparam_small(params):
    """small_det neck.inp_conv_dw fusion -> Conv2d(weight, bias).

    The training block is just the DilatedReparamBlock (no pointwise/BN
    after); the HF ``input_conv.{i}.depthwise_convolution`` is a single
    depthwise Conv2d.
    """
    return _fuse_dilated_reparam_branch_set(params, prefix="")


def _fuse_depthwise_pointwise_bn_medium(params):
    """medium_det neck.{inp_conv,pan_lat_conv} fusion -> Conv2d(weight, bias).

    Training-time block:
      ``DilatedReparamBlock (dw)`` -> ``1x1 conv (pw)`` -> ``BN (bn)``
    Fuse into a single conv2d, which is what the HF
    ``input_feature_projection_convolution`` / ``path_aggregation_lateral_convolution``
    expect.
    """
    dw_w, dw_b = _fuse_dilated_reparam_branch_set(params, prefix="dw.")
    pw_w = params["pw.weight"].squeeze(-1).squeeze(-1)
    w = pw_w[:, :, None, None] * dw_w[:, 0, :, :][None]
    b = np.matmul(pw_w, dw_b)
    if "bn.weight" in params:
        std = np.sqrt(_get_bn_value(params, "bn.running_var") + 1e-5)
        scale = params["bn.weight"] / std
        w = w * scale.reshape(-1, 1, 1, 1)
        b = params["bn.bias"] + (b - _get_bn_value(params, "bn.running_mean")) * scale
    return w, b


def _collect_groups(state_dict, group_patterns):
    """Iterate state_dict once and bucket keys into named groups.

    ``group_patterns`` is a list of (name, compiled regex) tuples;
    matching keys get their (sub-key, tensor) tucked into
    ``groups[name][group_id]``. ``group_id`` is derived from regex groups
    1..N-1 joined by ``"\\x00"`` (so callers split it cheaply); the final
    regex group is the per-group sub-key.

    Returns ``(groups, passthrough)``: keys that matched no group remain
    in ``passthrough`` with their values unchanged. ``drop_prefixes``
    filtering is expected to have happened upstream.
    """
    groups = {name: {} for name, _ in group_patterns}
    passthrough = {}
    for key, value in state_dict.items():
        matched = False
        for name, pattern in group_patterns:
            match = pattern.match(key)
            if match is None:
                continue
            group_id = "\x00".join(match.groups()[:-1])
            sub_key = match.groups()[-1]
            groups[name].setdefault(group_id, {})[sub_key] = value
            matched = True
            break
        if not matched:
            passthrough[key] = value
    return groups, passthrough


def fuse_v6_small_det_state_dict(state_dict):
    """Fuse all training-time reparam blocks for v6 small/tiny det.

    Output keys for fused blocks are already in HF inference naming;
    the remaining ``passthrough`` keys still carry PaddleOCR-style names
    and are handled downstream by the regex mapping.
    """
    groups, passthrough = _collect_groups(
        state_dict,
        group_patterns=[
            ("rep_dw", _REP_DW_PATTERN),
            ("inp_dw", _NECK_INPUT_DW_PATTERN_SMALL),
        ],
    )

    for group_id, params in groups["rep_dw"].items():
        stage_one_based, layer = group_id.split("\x00")
        stage = int(stage_one_based) - 1
        prefix = f"model.backbone.encoder.blocks.{stage}.blocks.{layer}.token_conv"
        w, b = _fuse_rep_dw(params)
        passthrough[f"{prefix}.weight"] = w
        passthrough[f"{prefix}.bias"] = b

    for group_id, params in groups["inp_dw"].items():
        layer = group_id  # only one capture group besides the sub-key
        prefix = f"model.neck.input_conv.{layer}.depthwise_convolution"
        w, b = _fuse_neck_input_depthwise_reparam_small(params)
        passthrough[f"{prefix}.weight"] = w
        passthrough[f"{prefix}.bias"] = b

    return passthrough


def fuse_v6_rec_state_dict(state_dict):
    """Fuse RepDWConv backbone blocks for v6 rec models (small/medium/tiny).

    The rec head is regex-mappable end-to-end (no reparam fusion needed),
    so only the backbone ``token_mixer.rep_dw`` groups get collapsed here.
    Stage indices are one-based and offset by ``-2`` (training uses
    ``blocks2..blocks5``, HF uses ``encoder.blocks.0..3``).
    """
    groups, passthrough = _collect_groups(
        state_dict,
        group_patterns=[("rep_dw", _REP_DW_PATTERN_REC)],
    )

    for group_id, params in groups["rep_dw"].items():
        stage_one_based, layer = group_id.split("\x00")
        stage = int(stage_one_based) - 2
        prefix = f"model.backbone.encoder.blocks.{stage}.blocks.{layer}.token_conv"
        w, b = _fuse_rep_dw(params)
        passthrough[f"{prefix}.weight"] = w
        passthrough[f"{prefix}.bias"] = b

    return passthrough


def fuse_v6_medium_det_state_dict(state_dict):
    """Fuse all training-time reparam blocks for v6 medium det."""
    groups, passthrough = _collect_groups(
        state_dict,
        group_patterns=[
            ("rep_dw", _REP_DW_PATTERN),
            ("neck_reparam", _NECK_REPARAM_PATTERN_MEDIUM),
        ],
    )

    for group_id, params in groups["rep_dw"].items():
        stage_one_based, layer = group_id.split("\x00")
        stage = int(stage_one_based) - 1
        prefix = f"model.backbone.encoder.blocks.{stage}.blocks.{layer}.token_conv"
        w, b = _fuse_rep_dw(params)
        passthrough[f"{prefix}.weight"] = w
        passthrough[f"{prefix}.bias"] = b

    for group_id, params in groups["neck_reparam"].items():
        name, layer = group_id.split("\x00")
        if name == "inp_conv":
            prefix = f"model.neck.input_feature_projection_convolution.{layer}"
        else:
            prefix = f"model.neck.path_aggregation_lateral_convolution.{layer}"
        w, b = _fuse_depthwise_pointwise_bn_medium(params)
        passthrough[f"{prefix}.weight"] = w
        passthrough[f"{prefix}.bias"] = b

    return passthrough
