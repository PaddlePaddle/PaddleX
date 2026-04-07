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

"""Weight converter: pdparams -> safetensors.

Conversion flow:
  1. paddle.load() -> state dict with OLD PaddleOCR/PaddleDetection key names
  2. Rename BatchNorm keys: _mean -> running_mean, _variance -> running_var
  3. Apply per-architecture regex key mappings (old keys -> HF keys)
  4. Transpose linear weight keys (Paddle [in, out] -> HF [out, in])
  5. Save as safetensors via safetensors.numpy.save_file()
"""

import json
import os
import re
from pathlib import Path

from ...utils import logging
from ...utils.config import AttrDict


def build_weight_converter(config: AttrDict) -> "WeightConverter":
    """Build a weight converter from PaddleX config.

    Args:
        config (AttrDict): PaddleX pipeline config.

    Returns:
        WeightConverter: the converter instance.
    """
    return WeightConverter(config)


# ---------------------------------------------------------------------------
# BatchNorm key renaming (pdparams2safetensors.py step)
# ---------------------------------------------------------------------------
# Paddle's BatchNorm uses _mean/_variance; HF uses running_mean/running_var.

_BN_KEY_RULES = [
    (re.compile(r"(.+)\._mean$"), r"\1.running_mean"),
    (re.compile(r"(.+)\._variance$"), r"\1.running_var"),
]


def _rename_bn_keys(state_dict):
    """Rename Paddle BatchNorm keys to HF convention."""
    new_sd = {}
    for k, v in state_dict.items():
        new_key = k
        for pattern, replacement in _BN_KEY_RULES:
            new_key, n = pattern.subn(replacement, new_key)
            if n > 0:
                break
        new_sd[new_key] = v
    return new_sd


# ---------------------------------------------------------------------------
# Per-architecture key mappings (from convert.py)
# ---------------------------------------------------------------------------
# Each mapping is an ordered list of (regex_pattern, replacement) tuples.
# Applied iteratively: after each substitution, the full rule set is re-scanned
# until no more substitutions occur (matching convert.py's convert_key logic).

# PPLCNet (PP-LCNet_x1_0_doc_ori, PP-LCNet_x1_0_table_cls,
#           PP-LCNet_x0_25_textline_ori, PP-LCNet_x1_0_textline_ori)
_PPLCNET_MAPPING = [
    (r"^conv1\.conv\.weight$", r"encoder.convolution.convolution.weight"),
    (r"^conv1\.bn\.weight$", r"encoder.convolution.normalization.weight"),
    (r"^conv1\.bn\.bias$", r"encoder.convolution.normalization.bias"),
    (r"^conv1\.bn\.running_mean$", r"encoder.convolution.normalization.running_mean"),
    (r"^conv1\.bn\.running_var$", r"encoder.convolution.normalization.running_var"),
    (r"^last_conv\.weight$", r"last_convolution.weight"),
    (r"^fc\.weight$", r"head.weight"),
    (r"^fc\.bias$", r"head.bias"),
    # blocks2-6 -> encoder.blocks.0-4
    (r"^blocks2\.(\d+)\.dw_conv\.conv\.(\w+)$", r"encoder.blocks.0.layers.\1.depthwise_convolution.convolution.\2"),
    (r"^blocks2\.(\d+)\.dw_conv\.bn\.(\w+)$", r"encoder.blocks.0.layers.\1.depthwise_convolution.normalization.\2"),
    (r"^blocks3\.(\d+)\.dw_conv\.conv\.(\w+)$", r"encoder.blocks.1.layers.\1.depthwise_convolution.convolution.\2"),
    (r"^blocks3\.(\d+)\.dw_conv\.bn\.(\w+)$", r"encoder.blocks.1.layers.\1.depthwise_convolution.normalization.\2"),
    (r"^blocks4\.(\d+)\.dw_conv\.conv\.(\w+)$", r"encoder.blocks.2.layers.\1.depthwise_convolution.convolution.\2"),
    (r"^blocks4\.(\d+)\.dw_conv\.bn\.(\w+)$", r"encoder.blocks.2.layers.\1.depthwise_convolution.normalization.\2"),
    (r"^blocks5\.(\d+)\.dw_conv\.conv\.(\w+)$", r"encoder.blocks.3.layers.\1.depthwise_convolution.convolution.\2"),
    (r"^blocks5\.(\d+)\.dw_conv\.bn\.(\w+)$", r"encoder.blocks.3.layers.\1.depthwise_convolution.normalization.\2"),
    (r"^blocks6\.(\d+)\.dw_conv\.conv\.(\w+)$", r"encoder.blocks.4.layers.\1.depthwise_convolution.convolution.\2"),
    (r"^blocks6\.(\d+)\.dw_conv\.bn\.(\w+)$", r"encoder.blocks.4.layers.\1.depthwise_convolution.normalization.\2"),
    (r"^blocks2\.(\d+)\.pw_conv\.conv\.(\w+)$", r"encoder.blocks.0.layers.\1.pointwise_convolution.convolution.\2"),
    (r"^blocks2\.(\d+)\.pw_conv\.bn\.(\w+)$", r"encoder.blocks.0.layers.\1.pointwise_convolution.normalization.\2"),
    (r"^blocks3\.(\d+)\.pw_conv\.conv\.(\w+)$", r"encoder.blocks.1.layers.\1.pointwise_convolution.convolution.\2"),
    (r"^blocks3\.(\d+)\.pw_conv\.bn\.(\w+)$", r"encoder.blocks.1.layers.\1.pointwise_convolution.normalization.\2"),
    (r"^blocks4\.(\d+)\.pw_conv\.conv\.(\w+)$", r"encoder.blocks.2.layers.\1.pointwise_convolution.convolution.\2"),
    (r"^blocks4\.(\d+)\.pw_conv\.bn\.(\w+)$", r"encoder.blocks.2.layers.\1.pointwise_convolution.normalization.\2"),
    (r"^blocks5\.(\d+)\.pw_conv\.conv\.(\w+)$", r"encoder.blocks.3.layers.\1.pointwise_convolution.convolution.\2"),
    (r"^blocks5\.(\d+)\.pw_conv\.bn\.(\w+)$", r"encoder.blocks.3.layers.\1.pointwise_convolution.normalization.\2"),
    (r"^blocks6\.(\d+)\.pw_conv\.conv\.(\w+)$", r"encoder.blocks.4.layers.\1.pointwise_convolution.convolution.\2"),
    (r"^blocks6\.(\d+)\.pw_conv\.bn\.(\w+)$", r"encoder.blocks.4.layers.\1.pointwise_convolution.normalization.\2"),
    # Squeeze-excitation in blocks6
    (r"^blocks6\.(\d+)\.se\.conv1\.(\w+)$", r"encoder.blocks.4.layers.\1.squeeze_excitation_module.convolutions.0.\2"),
    (r"^blocks6\.(\d+)\.se\.conv2\.(\w+)$", r"encoder.blocks.4.layers.\1.squeeze_excitation_module.convolutions.2.\2"),
]

# PP-OCRv5_mobile_det
_PPOCRV5_MOBILE_DET_MAPPING = [
    # Neck
    (r"neck\.ins_conv\.(\d+)\.se_block", r"model.neck.insert_conv.\1.squeeze_excitation_block"),
    (r"neck\.ins_conv\.", r"model.neck.insert_conv."),
    (r"neck\.inp_conv\.(\d+)\.se_block", r"model.neck.input_conv.\1.squeeze_excitation_block"),
    (r"neck\.inp_conv\.", r"model.neck.input_conv."),
    # Head binarize
    (r"^head\.binarize\.conv1\.weight", r"head.conv_down.convolution.weight"),
    (r"^head\.binarize\.conv2\.weight", r"head.conv_up.convolution.weight"),
    (r"^head\.binarize\.conv2\.bias", r"head.conv_up.convolution.bias"),
    (r"^head\.binarize\.conv3\.weight", r"head.conv_final.weight"),
    (r"^head\.binarize\.conv3\.bias", r"head.conv_final.bias"),
    (r"^head\.binarize\.conv_bn1\.weight", r"head.conv_down.norm.weight"),
    (r"^head\.binarize\.conv_bn1\.bias", r"head.conv_down.norm.bias"),
    (r"^head\.binarize\.conv_bn1\.running_mean", r"head.conv_down.norm.running_mean"),
    (r"^head\.binarize\.conv_bn1\.running_var", r"head.conv_down.norm.running_var"),
    (r"^head\.binarize\.conv_bn2\.weight", r"head.conv_up.norm.weight"),
    (r"^head\.binarize\.conv_bn2\.bias", r"head.conv_up.norm.bias"),
    (r"^head\.binarize\.conv_bn2\.running_mean", r"head.conv_up.norm.running_mean"),
    (r"^head\.binarize\.conv_bn2\.running_var", r"head.conv_up.norm.running_var"),
    # Backbone conv1
    (r"^backbone\.conv1\.conv\.weight$", r"model.backbone.encoder.convolution.convolution.weight"),
    (r"^backbone\.conv1\.bn\.weight$", r"model.backbone.encoder.convolution.normalization.weight"),
    (r"^backbone\.conv1\.bn\.bias$", r"model.backbone.encoder.convolution.normalization.bias"),
    (r"^backbone\.conv1\.bn\.running_mean$", r"model.backbone.encoder.convolution.normalization.running_mean"),
    (r"^backbone\.conv1\.bn\.running_var$", r"model.backbone.encoder.convolution.normalization.running_var"),
    # Backbone layer_list
    (r"^backbone\.layer_list\.(\d+)\.weight$", r"model.layer.\1.weight"),
    (r"^backbone\.layer_list\.(\d+)\.bias$", r"model.layer.\1.bias"),
    # Backbone blocks
    (r"^backbone\.blocks2\.0\.", r"model.backbone.encoder.blocks.0.layers.0."),
    (r"^backbone\.blocks3\.(\d+)\.", r"model.backbone.encoder.blocks.1.layers.\1."),
    (r"^backbone\.blocks4\.(\d+)\.", r"model.backbone.encoder.blocks.2.layers.\1."),
    (r"^backbone\.blocks5\.(\d+)\.", r"model.backbone.encoder.blocks.3.layers.\1."),
    (r"^backbone\.blocks6\.(\d+)\.", r"model.backbone.encoder.blocks.4.layers.\1."),
    # Sub-module renaming (applied after block mapping)
    (r"\.dw_conv\.", r".depthwise_convolution."),
    (r"\.pw_conv\.", r".pointwise_convolution."),
    (r"conv_1x1\.conv\.", r"conv_small_symmetric.convolution."),
    (r"conv_1x1\.bn\.", r"conv_small_symmetric.normalization."),
    (r"conv_kxk\.(\d+)\.conv\.", r"conv_symmetric.\1.convolution."),
    (r"conv_kxk\.(\d+)\.bn\.", r"conv_symmetric.\1.normalization."),
    (r"\.se\.conv1\.", r".squeeze_excitation_module.convolutions.0."),
    (r"\.se\.conv2\.", r".squeeze_excitation_module.convolutions.2."),
]

# PP-OCRv5_server_det
_PPOCRV5_SERVER_DET_MAPPING = [
    # Backbone stages
    (r"^backbone\.stages\.(\d+)\.blocks\.(\d+)\.layers\.(\d+)\.conv(\d)\.bn\.(\w+)$",
     r"model.backbone.encoder.stages.\1.blocks.\2.layers.\3.conv\4.normalization.\5"),
    (r"^backbone\.stages\.(\d+)\.blocks\.(\d+)\.layers\.(\d+)\.conv(\d)\.conv\.(\w+)$",
     r"model.backbone.encoder.stages.\1.blocks.\2.layers.\3.conv\4.convolution.\5"),
    (r"^backbone\.stages\.(\d+)\.blocks\.(\d+)\.layers\.(\d+)\.bn\.(\w+)$",
     r"model.backbone.encoder.stages.\1.blocks.\2.layers.\3.normalization.\4"),
    (r"^backbone\.stages\.(\d+)\.blocks\.(\d+)\.layers\.(\d+)\.conv\.(\w+)$",
     r"model.backbone.encoder.stages.\1.blocks.\2.layers.\3.convolution.\4"),
    # Downsample
    (r"^backbone\.stages\.(\d+)\.downsample\.bn\.(\w+)$",
     r"model.backbone.encoder.stages.\1.downsample.normalization.\2"),
    (r"^backbone\.stages\.(\d+)\.downsample\.conv\.(\w+)$",
     r"model.backbone.encoder.stages.\1.downsample.convolution.\2"),
    # Aggregation
    (r"^backbone\.stages\.(\d+)\.blocks\.(\d+)\.aggregation_squeeze_conv\.bn\.(\w+)$",
     r"model.backbone.encoder.stages.\1.blocks.\2.aggregation.0.normalization.\3"),
    (r"^backbone\.stages\.(\d+)\.blocks\.(\d+)\.aggregation_squeeze_conv\.conv\.(\w+)$",
     r"model.backbone.encoder.stages.\1.blocks.\2.aggregation.0.convolution.\3"),
    (r"^backbone\.stages\.(\d+)\.blocks\.(\d+)\.aggregation_excitation_conv\.bn\.(\w+)$",
     r"model.backbone.encoder.stages.\1.blocks.\2.aggregation.1.normalization.\3"),
    (r"^backbone\.stages\.(\d+)\.blocks\.(\d+)\.aggregation_excitation_conv\.conv\.(\w+)$",
     r"model.backbone.encoder.stages.\1.blocks.\2.aggregation.1.convolution.\3"),
    # Stem
    (r"^backbone\.stem\.stem(\d+[ab]?)\.bn\.(\w+)$", r"model.backbone.embedder.stem\1.normalization.\2"),
    (r"^backbone\.stem\.stem(\d+[ab]?)\.conv\.(\w+)$", r"model.backbone.embedder.stem\1.convolution.\2"),
    # Head
    (r"^head\.binarize\.conv1\.(\w+)$", r"head.binarize_head.conv_down.convolution.\1"),
    (r"^head\.binarize\.conv2\.(\w+)$", r"head.binarize_head.conv_up.convolution.\1"),
    (r"^head\.binarize\.conv3\.(\w+)$", r"head.binarize_head.conv_final.\1"),
    (r"^head\.binarize\.conv_bn1\.(\w+)$", r"head.binarize_head.conv_down.norm.\1"),
    (r"^head\.binarize\.conv_bn2\.(\w+)$", r"head.binarize_head.conv_up.norm.\1"),
    # Local Refinement Module
    (r"^head\.cbn_layer\.last_3\.conv\.(\w+)$", r"head.local_refinement_module.convolution_backbone.convolution.\1"),
    (r"^head\.cbn_layer\.last_3\.bn\.(\w+)$", r"head.local_refinement_module.convolution_backbone.norm.\1"),
    (r"^head\.cbn_layer\.last_1\.(\w+)$", r"head.local_refinement_module.convolution_final.\1"),
    # Neck
    (r"^neck\.incl(\d+)\.conv1x1_reduce_channel\.(\w+)$",
     lambda m: f"model.neck.intraclass_blocks.{int(m.group(1))-1}.conv_reduce_channel.{m.group(2)}"),
    (r"^neck\.incl(\d+)\.conv1x1_return_channel\.(\w+)$",
     lambda m: f"model.neck.intraclass_blocks.{int(m.group(1))-1}.conv_final.convolution.{m.group(2)}"),
    (r"^neck\.incl(\d+)\.bn\.(\w+)$",
     lambda m: f"model.neck.intraclass_blocks.{int(m.group(1))-1}.conv_final.norm.{m.group(2)}"),
    (r"^neck\.incl(\d+)\.v_layer_7x1\.(\w+)$",
     lambda m: f"model.neck.intraclass_blocks.{int(m.group(1))-1}.vertical_long_to_small_conv_longratio.{m.group(2)}"),
    (r"^neck\.incl(\d+)\.v_layer_5x1\.(\w+)$",
     lambda m: f"model.neck.intraclass_blocks.{int(m.group(1))-1}.vertical_long_to_small_conv_midratio.{m.group(2)}"),
    (r"^neck\.incl(\d+)\.v_layer_3x1\.(\w+)$",
     lambda m: f"model.neck.intraclass_blocks.{int(m.group(1))-1}.vertical_long_to_small_conv_shortratio.{m.group(2)}"),
    (r"^neck\.incl(\d+)\.q_layer_1x7\.(\w+)$",
     lambda m: f"model.neck.intraclass_blocks.{int(m.group(1))-1}.horizontal_small_to_long_conv_longratio.{m.group(2)}"),
    (r"^neck\.incl(\d+)\.q_layer_1x5\.(\w+)$",
     lambda m: f"model.neck.intraclass_blocks.{int(m.group(1))-1}.horizontal_small_to_long_conv_midratio.{m.group(2)}"),
    (r"^neck\.incl(\d+)\.q_layer_1x3\.(\w+)$",
     lambda m: f"model.neck.intraclass_blocks.{int(m.group(1))-1}.horizontal_small_to_long_conv_shortratio.{m.group(2)}"),
    (r"^neck\.incl(\d+)\.c_layer_7x7\.(\w+)$",
     lambda m: f"model.neck.intraclass_blocks.{int(m.group(1))-1}.symmetric_conv_long_longratio.{m.group(2)}"),
    (r"^neck\.incl(\d+)\.c_layer_5x5\.(\w+)$",
     lambda m: f"model.neck.intraclass_blocks.{int(m.group(1))-1}.symmetric_conv_long_midratio.{m.group(2)}"),
    (r"^neck\.incl(\d+)\.c_layer_3x3\.(\w+)$",
     lambda m: f"model.neck.intraclass_blocks.{int(m.group(1))-1}.symmetric_conv_long_shortratio.{m.group(2)}"),
    # Neck convolutions
    (r"^neck\.inp_conv\.(\d+)\.weight$", r"model.neck.input_feature_projection_convolution.\1.weight"),
    (r"^neck\.ins_conv\.(\d+)\.weight$", r"model.neck.input_channel_adjustment_convolution.\1.weight"),
    (r"^neck\.pan_lat_conv\.(\d+)\.weight$", r"model.neck.path_aggregation_lateral_convolution.\1.weight"),
    (r"^neck\.pan_head_conv\.(\d+)\.weight$", r"model.neck.path_aggregation_head_convolution.\1.weight"),
]

# RT-DETR family (RT-DETR-L_*, PP-DocLayout_plus-L, PP-DocBlockLayout,
#                  PP-DocLayoutV2, PP-DocLayoutV3)
# From _apply_rt_detr_key_conversion() in pp_doclayout_v2.py:39
_RTDETR_MAPPING = [
    (r"out_proj", r"o_proj"),
    (r"layers\.(\d+)\.fc1", r"layers.\1.mlp.fc1"),
    (r"layers\.(\d+)\.fc2", r"layers.\1.mlp.fc2"),
    (r"encoder\.encoder\.(\d+)\.layers", r"encoder.aifi.\1.layers"),
]

# UVDoc
_UVDOC_MAPPING = [
    # ResNet Down layers
    (r"^resnet_down\.layer1\.(\d+)\.conv1\.(\w+)$", r"backbone.resnet.resnet_down.0.layers.\1.conv_start.convolution.\2"),
    (r"^resnet_down\.layer1\.(\d+)\.conv2\.(\w+)$", r"backbone.resnet.resnet_down.0.layers.\1.conv_final.convolution.\2"),
    (r"^resnet_down\.layer1\.(\d+)\.conv1\.0\.(\w+)$", r"backbone.resnet.resnet_down.0.layers.\1.conv_start.convolution.\2"),
    (r"^resnet_down\.layer1\.(\d+)\.conv2\.0\.(\w+)$", r"backbone.resnet.resnet_down.0.layers.\1.conv_final.convolution.\2"),
    (r"^resnet_down\.layer1\.(\d+)\.bn1\.(\w+)$", r"backbone.resnet.resnet_down.0.layers.\1.conv_start.normalization.\2"),
    (r"^resnet_down\.layer1\.(\d+)\.bn2\.(\w+)$", r"backbone.resnet.resnet_down.0.layers.\1.conv_final.normalization.\2"),
    (r"^resnet_down\.layer2\.(\d+)\.conv1\.(\w+)$", r"backbone.resnet.resnet_down.1.layers.\1.conv_start.convolution.\2"),
    (r"^resnet_down\.layer2\.(\d+)\.conv2\.(\w+)$", r"backbone.resnet.resnet_down.1.layers.\1.conv_final.convolution.\2"),
    (r"^resnet_down\.layer2\.(\d+)\.conv1\.0\.(\w+)$", r"backbone.resnet.resnet_down.1.layers.\1.conv_start.convolution.\2"),
    (r"^resnet_down\.layer2\.(\d+)\.conv2\.0\.(\w+)$", r"backbone.resnet.resnet_down.1.layers.\1.conv_final.convolution.\2"),
    (r"^resnet_down\.layer2\.(\d+)\.bn1\.(\w+)$", r"backbone.resnet.resnet_down.1.layers.\1.conv_start.normalization.\2"),
    (r"^resnet_down\.layer2\.(\d+)\.bn2\.(\w+)$", r"backbone.resnet.resnet_down.1.layers.\1.conv_final.normalization.\2"),
    (r"^resnet_down\.layer2\.(\d+)\.downsample\.0\.(\w+)$", r"backbone.resnet.resnet_down.1.layers.\1.conv_down.convolution.\2"),
    (r"^resnet_down\.layer2\.(\d+)\.downsample\.1\.(\w+)$", r"backbone.resnet.resnet_down.1.layers.\1.conv_down.normalization.\2"),
    (r"^resnet_down\.layer3\.(\d+)\.conv1\.(\w+)$", r"backbone.resnet.resnet_down.2.layers.\1.conv_start.convolution.\2"),
    (r"^resnet_down\.layer3\.(\d+)\.conv2\.(\w+)$", r"backbone.resnet.resnet_down.2.layers.\1.conv_final.convolution.\2"),
    (r"^resnet_down\.layer3\.(\d+)\.conv1\.0\.(\w+)$", r"backbone.resnet.resnet_down.2.layers.\1.conv_start.convolution.\2"),
    (r"^resnet_down\.layer3\.(\d+)\.conv2\.0\.(\w+)$", r"backbone.resnet.resnet_down.2.layers.\1.conv_final.convolution.\2"),
    (r"^resnet_down\.layer3\.(\d+)\.bn1\.(\w+)$", r"backbone.resnet.resnet_down.2.layers.\1.conv_start.normalization.\2"),
    (r"^resnet_down\.layer3\.(\d+)\.bn2\.(\w+)$", r"backbone.resnet.resnet_down.2.layers.\1.conv_final.normalization.\2"),
    (r"^resnet_down\.layer3\.(\d+)\.downsample\.0\.(\w+)$", r"backbone.resnet.resnet_down.2.layers.\1.conv_down.convolution.\2"),
    (r"^resnet_down\.layer3\.(\d+)\.downsample\.1\.(\w+)$", r"backbone.resnet.resnet_down.2.layers.\1.conv_down.normalization.\2"),
    # ResNet Head
    (r"^resnet_head\.0\.", r"backbone.resnet.resnet_head.0.convolution."),
    (r"^resnet_head\.1\.", r"backbone.resnet.resnet_head.0.normalization."),
    (r"^resnet_head\.3\.", r"backbone.resnet.resnet_head.1.convolution."),
    (r"^resnet_head\.4\.", r"backbone.resnet.resnet_head.1.normalization."),
    # Bridge layers
    (r"^bridge_1\.(\d+)\.0\.", r"backbone.bridge.bridge.0.blocks.\1.convolution."),
    (r"^bridge_1\.(\d+)\.1\.", r"backbone.bridge.bridge.0.blocks.\1.normalization."),
    (r"^bridge_2\.(\d+)\.0\.", r"backbone.bridge.bridge.1.blocks.\1.convolution."),
    (r"^bridge_2\.(\d+)\.1\.", r"backbone.bridge.bridge.1.blocks.\1.normalization."),
    (r"^bridge_3\.(\d+)\.0\.", r"backbone.bridge.bridge.2.blocks.\1.convolution."),
    (r"^bridge_3\.(\d+)\.1\.", r"backbone.bridge.bridge.2.blocks.\1.normalization."),
    (r"^bridge_4\.(\d+)\.0\.", r"backbone.bridge.bridge.3.blocks.\1.convolution."),
    (r"^bridge_4\.(\d+)\.1\.", r"backbone.bridge.bridge.3.blocks.\1.normalization."),
    (r"^bridge_5\.(\d+)\.0\.", r"backbone.bridge.bridge.4.blocks.\1.convolution."),
    (r"^bridge_5\.(\d+)\.1\.", r"backbone.bridge.bridge.4.blocks.\1.normalization."),
    (r"^bridge_6\.(\d+)\.0\.", r"backbone.bridge.bridge.5.blocks.\1.convolution."),
    (r"^bridge_6\.(\d+)\.1\.", r"backbone.bridge.bridge.5.blocks.\1.normalization."),
    # Output heads
    (r"^out_point_positions2D\.0\.", r"head.out_point_positions2D.conv_down.convolution."),
    (r"^out_point_positions2D\.1\.", r"head.out_point_positions2D.conv_down.normalization."),
    (r"^out_point_positions2D\.2\.", r"head.out_point_positions2D.conv_down.activation."),
    (r"^out_point_positions2D\.3\.", r"head.out_point_positions2D.conv_up."),
    # Bridge connector
    (r"^bridge_concat\.0\.", r"head.bridge_connector.convolution."),
    (r"^bridge_concat\.1\.", r"head.bridge_connector.normalization."),
]

# ---------------------------------------------------------------------------
# Shared SVTR encoder + CTC head mapping (used by both mobile and server rec)
# ---------------------------------------------------------------------------
# OLD PaddleOCR MultiHead structure:
#   head.ctc_encoder.encoder.{conv1,conv2,conv3,conv4,conv1x1} -> EncoderWithSVTR
#   head.ctc_encoder.encoder.svtr_block.{0,1} -> Block (norm1/mixer/norm2/mlp)
#   head.ctc_encoder.encoder.norm -> LayerNorm
#   head.ctc_head.fc -> CTCHead linear
# NEW HF structure:
#   head.encoder.conv_block.{0..4} -> ConvLayer
#   head.encoder.svtr_block.{0,1} -> Block (layer_norm1/self_attn/layer_norm2/mlp)
#   head.encoder.norm -> LayerNorm
#   head.head -> Linear

_SVTR_CTC_HEAD_MAPPING = [
    # SVTR encoder conv blocks: conv1->0, conv2->1, conv3->2, conv4->3, conv1x1->4
    (r"^head\.ctc_encoder\.encoder\.conv1\.conv\.", r"head.encoder.conv_block.0.convolution."),
    (r"^head\.ctc_encoder\.encoder\.conv1\.norm\.", r"head.encoder.conv_block.0.normalization."),
    (r"^head\.ctc_encoder\.encoder\.conv2\.conv\.", r"head.encoder.conv_block.1.convolution."),
    (r"^head\.ctc_encoder\.encoder\.conv2\.norm\.", r"head.encoder.conv_block.1.normalization."),
    (r"^head\.ctc_encoder\.encoder\.conv3\.conv\.", r"head.encoder.conv_block.2.convolution."),
    (r"^head\.ctc_encoder\.encoder\.conv3\.norm\.", r"head.encoder.conv_block.2.normalization."),
    (r"^head\.ctc_encoder\.encoder\.conv4\.conv\.", r"head.encoder.conv_block.3.convolution."),
    (r"^head\.ctc_encoder\.encoder\.conv4\.norm\.", r"head.encoder.conv_block.3.normalization."),
    (r"^head\.ctc_encoder\.encoder\.conv1x1\.conv\.", r"head.encoder.conv_block.4.convolution."),
    (r"^head\.ctc_encoder\.encoder\.conv1x1\.norm\.", r"head.encoder.conv_block.4.normalization."),
    # SVTR transformer blocks
    (r"^head\.ctc_encoder\.encoder\.svtr_block\.(\d+)\.norm1\.", r"head.encoder.svtr_block.\1.layer_norm1."),
    (r"^head\.ctc_encoder\.encoder\.svtr_block\.(\d+)\.norm2\.", r"head.encoder.svtr_block.\1.layer_norm2."),
    (r"^head\.ctc_encoder\.encoder\.svtr_block\.(\d+)\.mixer\.qkv\.", r"head.encoder.svtr_block.\1.self_attn.qkv."),
    (r"^head\.ctc_encoder\.encoder\.svtr_block\.(\d+)\.mixer\.proj\.", r"head.encoder.svtr_block.\1.self_attn.projection."),
    (r"^head\.ctc_encoder\.encoder\.svtr_block\.(\d+)\.mlp\.", r"head.encoder.svtr_block.\1.mlp."),
    # Final norm
    (r"^head\.ctc_encoder\.encoder\.norm\.", r"head.encoder.norm."),
    # CTC head: head.ctc_head.fc -> head.head
    (r"^head\.ctc_head\.fc\.", r"head.head."),
]

# PP-OCRv5_mobile_rec (PPLCNetV3 backbone + SVTR encoder + CTC head)
_PPOCRV5_MOBILE_REC_MAPPING = [
    # Backbone conv1 (stem)
    (r"^backbone\.conv1\.conv\.(\w+)$", r"model.backbone.encoder.convolution.convolution.\1"),
    (r"^backbone\.conv1\.bn\.(\w+)$", r"model.backbone.encoder.convolution.normalization.\1"),
    # Block index mapping: blocks2->blocks.0, blocks3->blocks.1, etc.
    (r"^backbone\.blocks2\.(\d+)\.", r"model.backbone.encoder.blocks.0.layers.\1."),
    (r"^backbone\.blocks3\.(\d+)\.", r"model.backbone.encoder.blocks.1.layers.\1."),
    (r"^backbone\.blocks4\.(\d+)\.", r"model.backbone.encoder.blocks.2.layers.\1."),
    (r"^backbone\.blocks5\.(\d+)\.", r"model.backbone.encoder.blocks.3.layers.\1."),
    (r"^backbone\.blocks6\.(\d+)\.", r"model.backbone.encoder.blocks.4.layers.\1."),
    # Sub-module renaming (applied after block mapping)
    (r"\.dw_conv\.", r".depthwise_convolution."),
    (r"\.pw_conv\.", r".pointwise_convolution."),
    (r"\.conv_kxk\.(\d+)\.conv\.", r".conv_symmetric.\1.convolution."),
    (r"\.conv_kxk\.(\d+)\.bn\.", r".conv_symmetric.\1.normalization."),
    (r"\.conv_1x1\.conv\.", r".conv_small_symmetric.convolution."),
    (r"\.conv_1x1\.bn\.", r".conv_small_symmetric.normalization."),
    (r"\.se\.conv1\.", r".squeeze_excitation_module.convolutions.0."),
    (r"\.se\.conv2\.", r".squeeze_excitation_module.convolutions.2."),
] + _SVTR_CTC_HEAD_MAPPING

# PP-OCRv5_server_rec (HGNetV2 backbone + SVTR encoder + CTC head)
_PPOCRV5_SERVER_REC_MAPPING = [
    # Stem: backbone.stem.stemX.conv/bn -> model.backbone.embedder.stemX.convolution/normalization
    (r"^backbone\.stem\.(\w+)\.conv\.(\w+)$", r"model.backbone.embedder.\1.convolution.\2"),
    (r"^backbone\.stem\.(\w+)\.bn\.(\w+)$", r"model.backbone.embedder.\1.normalization.\2"),
    # Stages prefix: backbone.stages -> model.backbone.encoder.stages
    (r"^backbone\.stages\.", r"model.backbone.encoder.stages."),
    # Aggregation (must come before generic conv/bn rules)
    (r"\.aggregation_squeeze_conv\.conv\.", r".aggregation.0.convolution."),
    (r"\.aggregation_squeeze_conv\.bn\.", r".aggregation.0.normalization."),
    (r"\.aggregation_excitation_conv\.conv\.", r".aggregation.1.convolution."),
    (r"\.aggregation_excitation_conv\.bn\.", r".aggregation.1.normalization."),
    # Downsample
    (r"\.downsample\.conv\.", r".downsample.convolution."),
    (r"\.downsample\.bn\.", r".downsample.normalization."),
    # Light block layers (conv1/conv2 sub-layers) — must come before generic
    (r"\.layers\.(\d+)\.conv(\d)\.conv\.", r".layers.\1.conv\2.convolution."),
    (r"\.layers\.(\d+)\.conv(\d)\.bn\.", r".layers.\1.conv\2.normalization."),
    # Non-light block layers
    (r"\.layers\.(\d+)\.conv\.", r".layers.\1.convolution."),
    (r"\.layers\.(\d+)\.bn\.", r".layers.\1.normalization."),
] + _SVTR_CTC_HEAD_MAPPING

# Keys to drop during rec model conversion (NRTR head + unused backbone layers)
_REC_DROP_PREFIXES = [
    "head.before_gtc.",   # NRTR preprocessing (FCTranspose)
    "head.gtc_head.",     # NRTR decoder (Transformer)
    "head.encoder_reshape.",  # Im2Seq (no params, but just in case)
]
_SERVER_REC_DROP_PREFIXES = _REC_DROP_PREFIXES + [
    "backbone.fc.",        # Classification head (not used in rec inference)
    "backbone.last_conv.", # Last conv (not used in rec inference)
]

# SLANeXt — deferred to Phase 2
_SLANEXT_MAPPING = []  # SLANeXt_wired, SLANeXt_wireless


def _apply_key_mapping(state_dict, mapping):
    """Apply regex key mappings to state dict, matching convert.py logic.

    Each key is tested against all rules iteratively until no more
    substitutions occur.
    """
    new_sd = {}
    for key, value in state_dict.items():
        current_key = key
        while True:
            replaced = False
            for pattern, replacement in mapping:
                if callable(replacement):
                    new_key = re.sub(pattern, replacement, current_key)
                else:
                    new_key, n = re.subn(pattern, replacement, current_key)
                    if n > 0:
                        current_key = new_key
                        replaced = True
                        break
                if new_key != current_key:
                    current_key = new_key
                    replaced = True
                    break
            if not replaced:
                break
        new_sd[current_key] = value
    return new_sd


# ---------------------------------------------------------------------------
# Per-model inference metadata (for inference.yml output)
# ---------------------------------------------------------------------------
# inference.yml is load-bearing: predictors read label_list, character_dict,
# image_shape, etc. from it at runtime. Missing required fields cause failures.

# Classification label lists
_LABEL_DOC_ORI = ["0", "90", "180", "270"]
_LABEL_TABLE_CLS = ["wired_table", "wireless_table"]
_LABEL_TEXTLINE_ORI = ["0_degree", "180_degree"]

# Detection/layout label lists
_LABEL_TABLE_CELL_DET = ["cell"]
_LABEL_DOC_BLOCK_LAYOUT = ["Region"]
_LABEL_DOC_LAYOUT_PLUS = [
    "paragraph_title", "image", "text", "number", "abstract", "content",
    "figure_title", "formula", "table", "reference", "doc_title", "footnote",
    "header", "algorithm", "footer", "seal", "chart", "formula_number",
    "aside_text", "reference_content",
]
_LABEL_DOC_LAYOUT_V2V3 = [
    "abstract", "algorithm", "aside_text", "chart", "content",
    "display_formula", "doc_title", "figure_title", "footer", "footer_image",
    "footnote", "formula_number", "header", "header_image", "image",
    "inline_formula", "number", "paragraph_title", "reference",
    "reference_content", "seal", "table", "text", "title", "vision_footnote",
]

# DBPostProcess defaults for text detection
_DET_POSTPROCESS = {
    "PostProcess": {
        "name": "DBPostProcess",
        "thresh": 0.3,
        "box_thresh": 0.6,
        "max_candidates": 1000,
        "unclip_ratio": 1.5,
    }
}

# Text recognition: image_shape and character_dict are required.
# character_dict (18383 chars) is loaded at runtime from a known file.
_REC_IMAGE_SHAPE = [3, 48, 320]


def _build_rec_inference_meta(model_name):
    """Build inference_meta for rec models, loading character_dict at runtime."""
    meta = {
        "PreProcess": {
            "transform_ops": [
                {"DecodeImage": {"channel_first": False, "img_mode": "BGR"}},
                {"RecResizeImg": {"image_shape": _REC_IMAGE_SHAPE}},
            ]
        },
        "PostProcess": {
            "name": "CTCLabelDecode",
        },
    }
    return meta


_BUNDLED_DICT_PATH = Path(__file__).resolve().parent / "res" / "ppocrv5_dict.txt"


def _load_character_dict():
    """Load PP-OCRv5 character dict.

    Uses the bundled dict file shipped with PaddleX
    (paddlex/modules/base/res/ppocrv5_dict.txt).
    """
    if not _BUNDLED_DICT_PATH.exists():
        raise FileNotFoundError(
            f"Bundled character dict not found at {_BUNDLED_DICT_PATH}. "
            "This file is required for rec model conversion."
        )
    chars = _BUNDLED_DICT_PATH.read_text("utf-8").strip().split("\n")
    logging.info(f"Loaded character dict ({len(chars)} chars)")
    return chars


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# model_name -> (key_mapping, config_overrides, inference_meta, drop_key_prefixes)
# drop_key_prefixes: list of key prefixes to drop from pdparams before conversion
#   (e.g., NRTR head keys in rec models that are not used for CTC inference)
#
# inference_meta can be a dict (static) or a callable returning a dict (dynamic).
# Rec models use a callable to load the character_dict at runtime.

_META_CLS_DOC_ORI = {
    "PostProcess": {"Topk": {"topk": 1, "label_list": _LABEL_DOC_ORI}},
}
_META_CLS_TABLE = {
    "PostProcess": {"Topk": {"topk": 5, "label_list": _LABEL_TABLE_CLS}},
}
_META_CLS_TEXTLINE = {
    "PostProcess": {"Topk": {"topk": 1, "label_list": _LABEL_TEXTLINE_ORI}},
}

_META_DET_RTDETR = lambda labels: {
    "label_list": labels,
    "draw_threshold": 0.5,
}

_MODEL_REGISTRY = {
    # PPLCNet family
    "PP-LCNet_x1_0_doc_ori": (
        _PPLCNET_MAPPING, {}, _META_CLS_DOC_ORI, [],
    ),
    "PP-LCNet_x1_0_table_cls": (
        _PPLCNET_MAPPING, {"class_num": 2}, _META_CLS_TABLE, [],
    ),
    "PP-LCNet_x0_25_textline_ori": (
        _PPLCNET_MAPPING, {"scale": 0.25, "class_num": 2}, _META_CLS_TEXTLINE, [],
    ),
    "PP-LCNet_x1_0_textline_ori": (
        _PPLCNET_MAPPING, {"class_num": 2}, _META_CLS_TEXTLINE, [],
    ),
    # Text detection
    "PP-OCRv5_mobile_det": (_PPOCRV5_MOBILE_DET_MAPPING, {}, _DET_POSTPROCESS, []),
    "PP-OCRv5_server_det": (_PPOCRV5_SERVER_DET_MAPPING, {}, _DET_POSTPROCESS, []),
    # Text recognition — inference_meta built dynamically (character_dict loaded at runtime)
    "PP-OCRv5_mobile_rec": (
        _PPOCRV5_MOBILE_REC_MAPPING, {},
        _build_rec_inference_meta, _REC_DROP_PREFIXES,
    ),
    "PP-OCRv5_server_rec": (
        _PPOCRV5_SERVER_REC_MAPPING, {},
        _build_rec_inference_meta, _SERVER_REC_DROP_PREFIXES,
    ),
    # Table structure recognition — deferred to Phase 2
    "SLANeXt_wired": (_SLANEXT_MAPPING, {}, {}, []),
    "SLANeXt_wireless": (_SLANEXT_MAPPING, {}, {}, []),
    # Layout analysis (RT-DETR based, 25 labels)
    "PP-DocLayoutV2": (
        _RTDETR_MAPPING, {},
        _META_DET_RTDETR(_LABEL_DOC_LAYOUT_V2V3), [],
    ),
    "PP-DocLayoutV3": (
        _RTDETR_MAPPING, {},
        _META_DET_RTDETR(_LABEL_DOC_LAYOUT_V2V3), [],
    ),
    # Object detection (RT-DETR based)
    "RT-DETR-L_wired_table_cell_det": (
        _RTDETR_MAPPING, {"num_labels": 1},
        _META_DET_RTDETR(_LABEL_TABLE_CELL_DET), [],
    ),
    "RT-DETR-L_wireless_table_cell_det": (
        _RTDETR_MAPPING, {"num_labels": 1},
        _META_DET_RTDETR(_LABEL_TABLE_CELL_DET), [],
    ),
    "PP-DocLayout_plus-L": (
        _RTDETR_MAPPING, {"num_labels": 20},
        _META_DET_RTDETR(_LABEL_DOC_LAYOUT_PLUS), [],
    ),
    "PP-DocBlockLayout": (
        _RTDETR_MAPPING, {"num_labels": 1},
        _META_DET_RTDETR(_LABEL_DOC_BLOCK_LAYOUT), [],
    ),
    # Image unwarping
    "UVDoc": (_UVDOC_MAPPING, {}, {}, []),
}


# ---------------------------------------------------------------------------
# Tensor preprocessing (from pdparams2safetensors.py)
# ---------------------------------------------------------------------------
# Applied on OLD key names, BEFORE regex key mapping.

# Substring patterns for 2D weight tensors that need transposition.
# Paddle Linear stores [in_features, out_features]; HF uses [out_features, in_features].
# From pdparams2safetensors.py t_layers list.
_TRANSPOSE_SUBSTRINGS = [
    "fc",
    "channelwise",
    "mapper_crp",
    "mapper_sca",
    ".mapper.",
    "txt_mapper",
    "txt_pooled_mapper",
    "clip_img_mapper",
    "kv_mapper",
    "clip_mapper",
    "out_proj",
    "q_proj",
    "k_proj",
    "v_proj",
    "lm_head",
    "gate_proj",
    "up_proj",
    "down_proj",
    "o_proj",
    "linear_1",
    "linear_2",
    "attn.qkv",
    "mlp.lin1",
    "mlp.lin2",
    "attn.proj",
    "mm_projector_vary",
]


def _preprocess_tensors(state_dict):
    """Preprocess Paddle tensors: dtype cast, transpose, channelwise reshape.

    Matches pdparams2safetensors.py logic. Applied on OLD key names before
    key mapping.

    Returns dict of {key: numpy_array}.
    """
    import numpy as np

    result = {}
    for key, tensor in state_dict.items():
        # Convert to numpy, casting bf16/fp16 to fp32
        if hasattr(tensor, "numpy"):
            import paddle
            if tensor.dtype in (paddle.bfloat16, paddle.float16):
                tensor = tensor.astype(paddle.float32)
            np_weight = tensor.cpu().numpy()
        elif isinstance(tensor, np.ndarray):
            np_weight = tensor
        else:
            np_weight = np.array(tensor)

        # Channelwise gamma/beta reshape
        if "channelwise" in key and ("gamma" in key or "beta" in key):
            np_weight = np_weight.reshape((-1, 1, 1, 1))

        # Transpose 2D linear weights (skip bias)
        if "bias" not in key:
            if any(t in key for t in _TRANSPOSE_SUBSTRINGS):
                if np_weight.ndim == 2:
                    np_weight = np_weight.transpose()

        # Handle in_proj splitting (fused qkv -> separate q, k, v)
        if "attention.in_proj" in key or "attention.attn.in_proj" in key:
            split_size = np_weight.shape[0] // 3
            if "weight" in key:
                # Already transposed above, split along first axis
                result[key.replace("in_proj_weight", "q_proj.weight")] = np_weight[:split_size]
                result[key.replace("in_proj_weight", "k_proj.weight")] = np_weight[split_size:2*split_size]
                result[key.replace("in_proj_weight", "v_proj.weight")] = np_weight[2*split_size:]
            elif "bias" in key:
                result[key.replace("in_proj_bias", "q_proj.bias")] = np_weight[:split_size]
                result[key.replace("in_proj_bias", "k_proj.bias")] = np_weight[split_size:2*split_size]
                result[key.replace("in_proj_bias", "v_proj.bias")] = np_weight[2*split_size:]
            continue

        result[key] = np_weight

    return result


# ---------------------------------------------------------------------------
# input_path resolution
# ---------------------------------------------------------------------------

def _resolve_input_path(input_path):
    """Resolve input_path to a concrete .pdparams file.

    Accepted forms:
      - Direct file path ending with .pdparams
      - Checkpoint/inference directory containing .pdparams file(s)

    Directory resolution order:
      1. model_state.pdparams   (inference-style, per model_paths.py:70)
      2. inference.pdparams     (inference-style, per model_paths.py:72)
      3. best_model.pdparams    (training checkpoint, per Evaluate.weight_path
                                 in PP-DocLayoutV3.yaml:28)
      4. best_accuracy.pdparams (training checkpoint, per Evaluate.weight_path
                                 in PP-OCRv5_mobile_det.yaml:29)
      5. Single *.pdparams file in directory
    """
    p = Path(input_path)

    if p.is_file():
        if not p.name.endswith(".pdparams"):
            raise ValueError(
                f"input_path file must end with .pdparams, got: {p}"
            )
        return str(p)

    if p.is_dir():
        candidates = [
            "model_state.pdparams",
            "inference.pdparams",
            "best_model.pdparams",
            "best_accuracy.pdparams",
        ]
        for name in candidates:
            candidate = p / name
            if candidate.exists():
                return str(candidate)

        pdparams_files = list(p.glob("*.pdparams"))
        if len(pdparams_files) == 1:
            return str(pdparams_files[0])
        elif len(pdparams_files) > 1:
            names = [f.name for f in pdparams_files]
            raise ValueError(
                f"Multiple .pdparams files found in {p}: {names}. "
                "Please specify the exact file path."
            )
        else:
            raise FileNotFoundError(
                f"No .pdparams files found in directory: {p}"
            )

    raise FileNotFoundError(f"input_path does not exist: {p}")


# ---------------------------------------------------------------------------
# WeightConverter
# ---------------------------------------------------------------------------

class WeightConverter:
    """Converts Paddle .pdparams weights to safetensors format.

    Phase 1 scope: 17 official-label-space models.
    Hardcodes official metadata (label_list, character_dict, etc.) per model.
    """

    def __init__(self, config):
        self.global_config = config.Global
        self.convert_config = config.Pdparams2safetensors
        self.model_name = config.Global.model

        self.input_path = self.convert_config.input_path
        self.output_dir = self.convert_config.output_dir

        if self.input_path is None:
            raise ValueError(
                "Pdparams2safetensors.input_path is required. "
                "Specify a .pdparams file or a directory containing one."
            )
        if self.output_dir is None:
            raise ValueError("Pdparams2safetensors.output_dir is required.")

        if self.model_name not in _MODEL_REGISTRY:
            supported = ", ".join(sorted(_MODEL_REGISTRY.keys()))
            raise ValueError(
                f"Model '{self.model_name}' is not supported for "
                f"pdparams2safetensors conversion. Supported models: {supported}"
            )

    def convert(self):
        """Execute the pdparams -> safetensors conversion.

        Flow (matches Stage 1 pdparams2safetensors.py + convert.py order):
          1. Load pdparams, drop unused keys, BN rename, preprocess tensors,
             apply per-architecture regex key mapping
          2. Save model.safetensors, config.json, preprocess_config.json,
             inference.yml (and llm config for Chart2Table models)
        """
        from ...inference.models.doc_vlm.constants import PP_CHART2TABLE_MODELS

        key_mapping, config_overrides, inference_meta_or_fn, drop_prefixes = (
            _MODEL_REGISTRY[self.model_name]
        )

        if callable(inference_meta_or_fn):
            inference_meta = inference_meta_or_fn(self.model_name)
        else:
            inference_meta = inference_meta_or_fn

        numpy_sd = self._convert_weights(key_mapping, drop_prefixes)

        os.makedirs(self.output_dir, exist_ok=True)
        self._save_safetensors(numpy_sd)
        self._save_model_config(config_overrides)
        self._save_preprocess_config()
        self._save_inference_yml(inference_meta)

        if self.model_name in PP_CHART2TABLE_MODELS:
            self._save_llm_config()

        logging.info(
            f"Conversion complete. Output saved to: {self.output_dir}"
        )

    def _convert_weights(self, key_mapping, drop_prefixes):
        """Load pdparams and convert to numpy state dict with HF key names.

        Performs: load → drop unused keys → BN rename → tensor preprocessing
        (dtype cast, transpose, in_proj split) → regex key mapping.
        Returns dict of {hf_key: numpy_array}.
        """
        import paddle

        resolved_path = _resolve_input_path(self.input_path)
        logging.info(f"Loading weights from: {resolved_path}")
        state_dict = paddle.load(resolved_path)

        if drop_prefixes:
            dropped = [
                k for k in state_dict
                if any(k.startswith(p) for p in drop_prefixes)
            ]
            for k in dropped:
                del state_dict[k]
            if dropped:
                logging.info(
                    f"Dropped {len(dropped)} keys not needed for inference"
                )

        state_dict = _rename_bn_keys(state_dict)
        numpy_sd = _preprocess_tensors(state_dict)

        if key_mapping:
            numpy_sd = _apply_key_mapping(numpy_sd, key_mapping)
        else:
            logging.warning(
                f"No key mapping defined for {self.model_name}. "
                "Keys will be saved as-is from pdparams."
            )

        return numpy_sd

    def _save_safetensors(self, numpy_sd):
        """Save numpy state dict as model.safetensors."""
        from safetensors.numpy import save_file

        out_path = os.path.join(self.output_dir, "model.safetensors")
        save_file(numpy_sd, out_path)
        logging.info(f"Saved model.safetensors to: {out_path}")

    def _save_model_config(self, config_overrides):
        """Save model config as config.json."""
        out_path = os.path.join(self.output_dir, "config.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(config_overrides, f, indent=2, ensure_ascii=False)
        logging.info(f"Saved config.json to: {out_path}")

    def _save_preprocess_config(self):
        """Save preprocess_config.json (placeholder for HF image processor)."""
        out_path = os.path.join(self.output_dir, "preprocess_config.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({}, f, indent=2)
        logging.info(f"Saved preprocess_config.json to: {out_path}")

    def _save_inference_yml(self, inference_meta):
        """Save inference.yml from per-model metadata template.

        This file is load-bearing: predictors read label_list, character_dict,
        image_shape, etc. from it at runtime. Missing required fields cause
        runtime failures.

        For rec models, the character_dict (18K+ chars) is loaded from a
        known file location rather than hardcoded.
        """
        import yaml

        inference_config = {"Global": {"model_name": self.model_name}}
        inference_config.update(inference_meta)

        if self.model_name in ("PP-OCRv5_mobile_rec", "PP-OCRv5_server_rec"):
            char_dict = _load_character_dict()
            inference_config.setdefault("PostProcess", {})["character_dict"] = char_dict

        out_path = os.path.join(self.output_dir, "inference.yml")
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.dump(
                inference_config, f,
                default_flow_style=False,
                allow_unicode=True,
            )
        logging.info(f"Saved inference.yml to: {out_path}")

    def _save_llm_config(self):
        """Save LLM config for Chart2Table models (tokenizer assets, etc.).

        TODO: implement in Phase 2 (requires tokenizer asset handling from
        doc_vlm/predictor.py).
        """
        raise NotImplementedError(
            f"LLM config saving is not yet implemented for {self.model_name}. "
            "This will be added in Phase 2."
        )
