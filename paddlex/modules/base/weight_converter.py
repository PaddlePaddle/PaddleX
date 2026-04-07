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

# TODO: derive from model code
_PPOCRV5_MOBILE_REC_MAPPING = []  # PP-OCRv5_mobile_rec
_PPOCRV5_SERVER_REC_MAPPING = []  # PP-OCRv5_server_rec
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
# Model registry
# ---------------------------------------------------------------------------
# model_name -> (key_mapping, config_overrides, inference_meta)

_MODEL_REGISTRY = {
    # PPLCNet family
    "PP-LCNet_x1_0_doc_ori": (_PPLCNET_MAPPING, {}, {}),
    "PP-LCNet_x1_0_table_cls": (_PPLCNET_MAPPING, {"class_num": 2}, {}),
    "PP-LCNet_x0_25_textline_ori": (_PPLCNET_MAPPING, {"scale": 0.25, "class_num": 2}, {}),
    "PP-LCNet_x1_0_textline_ori": (_PPLCNET_MAPPING, {"class_num": 2}, {}),
    # Text detection
    "PP-OCRv5_mobile_det": (_PPOCRV5_MOBILE_DET_MAPPING, {}, {}),
    "PP-OCRv5_server_det": (_PPOCRV5_SERVER_DET_MAPPING, {}, {}),
    # Text recognition
    "PP-OCRv5_mobile_rec": (_PPOCRV5_MOBILE_REC_MAPPING, {}, {}),  # TODO: mapping
    "PP-OCRv5_server_rec": (_PPOCRV5_SERVER_REC_MAPPING, {}, {}),  # TODO: mapping
    # Table structure recognition
    "SLANeXt_wired": (_SLANEXT_MAPPING, {}, {}),  # TODO: mapping
    "SLANeXt_wireless": (_SLANEXT_MAPPING, {}, {}),  # TODO: mapping
    # Layout analysis / Object detection (RT-DETR based)
    "PP-DocLayoutV2": (_RTDETR_MAPPING, {}, {}),
    "PP-DocLayoutV3": (_RTDETR_MAPPING, {}, {}),
    "RT-DETR-L_wired_table_cell_det": (_RTDETR_MAPPING, {"num_labels": 1}, {}),
    "RT-DETR-L_wireless_table_cell_det": (_RTDETR_MAPPING, {"num_labels": 1}, {}),
    "PP-DocLayout_plus-L": (_RTDETR_MAPPING, {"num_labels": 11}, {}),
    "PP-DocBlockLayout": (_RTDETR_MAPPING, {"num_labels": 11}, {}),
    # Image unwarping
    "UVDoc": (_UVDOC_MAPPING, {}, {}),
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

        Flow matches Stage 1 (pdparams2safetensors.py + convert.py):
          1. paddle.load() -> state dict with OLD keys
          2. BN rename (_mean -> running_mean, _variance -> running_var)
          3. Preprocess tensors: dtype cast, transpose, in_proj split (on OLD keys)
          4. Apply per-architecture regex key mapping (old keys -> HF keys)
          5. Save outputs
        """
        import paddle

        key_mapping, config_overrides, inference_meta = _MODEL_REGISTRY[
            self.model_name
        ]

        # 1. Resolve input path
        resolved_path = _resolve_input_path(self.input_path)
        logging.info(f"Loading weights from: {resolved_path}")

        # 2. Load pdparams state dict
        state_dict = paddle.load(resolved_path)

        # 3. Rename BatchNorm keys (_mean -> running_mean, _variance -> running_var)
        state_dict = _rename_bn_keys(state_dict)

        # 4. Preprocess: dtype cast, transpose linears, in_proj split (on OLD keys)
        numpy_sd = _preprocess_tensors(state_dict)

        # 5. Apply per-architecture regex key mapping (old keys -> HF keys)
        if key_mapping:
            numpy_sd = _apply_key_mapping(numpy_sd, key_mapping)
        else:
            logging.warning(
                f"No key mapping defined for {self.model_name}. "
                "Keys will be saved as-is from pdparams."
            )

        # 6. Save outputs
        os.makedirs(self.output_dir, exist_ok=True)
        self._save_safetensors(numpy_sd)
        self._save_config_json(config_overrides)
        self._save_preprocess_config()
        self._save_inference_yml(inference_meta)

        logging.info(
            f"Conversion complete. Output saved to: {self.output_dir}"
        )

    def _save_safetensors(self, numpy_sd):
        """Save numpy state dict as model.safetensors."""
        from safetensors.numpy import save_file

        out_path = os.path.join(self.output_dir, "model.safetensors")
        save_file(numpy_sd, out_path)
        logging.info(f"Saved model.safetensors to: {out_path}")

    def _save_config_json(self, config_overrides):
        """Save model config as config.json."""
        # TODO: use model config class defaults + overrides for full config
        out_path = os.path.join(self.output_dir, "config.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(config_overrides, f, indent=2, ensure_ascii=False)
        logging.info(f"Saved config.json to: {out_path}")

    def _save_preprocess_config(self):
        """Save preprocess_config.json from per-architecture template."""
        # TODO: populate per-architecture preprocess templates
        out_path = os.path.join(self.output_dir, "preprocess_config.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({}, f, indent=2)
        logging.info(f"Saved preprocess_config.json to: {out_path}")

    def _save_inference_yml(self, inference_meta):
        """Save inference.yml from per-model template.

        This file is load-bearing: predictors read label_list, character_dict,
        image_shape, and other fields from it at runtime.

        Required fields by architecture:
        - Detection/layout: label_list (REQUIRED at object_detection/predictor.py:350
          and layout_analysis/predictor.py:176), draw_threshold
        - Text recognition: RecResizeImg.image_shape (at text_recognition/predictor.py:151),
          PostProcess.character_dict (at text_recognition/predictor.py:199)
        - Table structure: TableLabelEncode.merge_no_span_structure
          (at table_structure_recognition/predictor.py:59),
          PostProcess.character_dict (at table_structure_recognition/predictor.py:185)
        - Classification: label_list NOT required (Topk falls back to numeric ids
          at image_classification/processors.py:72)
        """
        # TODO: populate per-model required fields
        import yaml

        inference_config = {"Global": {"model_name": self.model_name}}
        inference_config.update(inference_meta)

        out_path = os.path.join(self.output_dir, "inference.yml")
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.dump(inference_config, f, default_flow_style=False, allow_unicode=True)
        logging.info(f"Saved inference.yml to: {out_path}")
