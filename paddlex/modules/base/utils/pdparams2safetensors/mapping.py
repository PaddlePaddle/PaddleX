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

"""Key mappings for pdparams -> safetensors conversion.

Each mapping is an ordered list of (regex_pattern, replacement) tuples.
Applied iteratively: after each substitution, the full rule set is re-scanned
until no more substitutions occur (matching convert.py's convert_key logic).
"""

import re

# BatchNorm key renaming
_BN_KEY_RULES = [
    (re.compile(r"(.+)\._mean$"), r"\1.running_mean"),
    (re.compile(r"(.+)\._variance$"), r"\1.running_var"),
]


def rename_bn_keys(state_dict):
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


def apply_key_mapping(state_dict, mapping):
    """Apply regex key mappings to state dict.

    Each key is tested against all rules iteratively until no more
    substitutions occur, matching convert.py's convert_key logic.
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


# PPLCNet
PPLCNET_MAPPING = [
    # Initial conv + BN
    (r"^conv1\.conv\.", r"encoder.convolution.convolution."),
    (r"^conv1\.bn\.", r"encoder.convolution.normalization."),
    # Block stage mapping: blocks{S} -> encoder.blocks.{S-2}
    (r"^blocks2\.(\d+)\.", lambda m: f"encoder.blocks.0.layers.{m.group(1)}."),
    (r"^blocks3\.(\d+)\.", lambda m: f"encoder.blocks.1.layers.{m.group(1)}."),
    (r"^blocks4\.(\d+)\.", lambda m: f"encoder.blocks.2.layers.{m.group(1)}."),
    (r"^blocks5\.(\d+)\.", lambda m: f"encoder.blocks.3.layers.{m.group(1)}."),
    (r"^blocks6\.(\d+)\.", lambda m: f"encoder.blocks.4.layers.{m.group(1)}."),
    # Sub-module renaming (applied after block mapping)
    (r"\.dw_conv\.conv\.", r".depthwise_convolution.convolution."),
    (r"\.dw_conv\.bn\.", r".depthwise_convolution.normalization."),
    (r"\.pw_conv\.conv\.", r".pointwise_convolution.convolution."),
    (r"\.pw_conv\.bn\.", r".pointwise_convolution.normalization."),
    (r"\.se\.conv1\.", r".squeeze_excitation_module.convolutions.0."),
    (r"\.se\.conv2\.", r".squeeze_excitation_module.convolutions.2."),
    # Head + final conv
    (r"^fc\.", r"head."),
    (r"^last_conv\.", r"last_convolution."),
]


# PP-OCRv5_mobile_det
PPOCRV5_MOBILE_DET_MAPPING = [
    # Neck
    (
        r"neck\.ins_conv\.(\d+)\.se_block",
        r"model.neck.insert_conv.\1.squeeze_excitation_block",
    ),
    (r"neck\.ins_conv\.", r"model.neck.insert_conv."),
    (
        r"neck\.inp_conv\.(\d+)\.se_block",
        r"model.neck.input_conv.\1.squeeze_excitation_block",
    ),
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
    (
        r"^backbone\.conv1\.conv\.weight$",
        r"model.backbone.encoder.convolution.convolution.weight",
    ),
    (
        r"^backbone\.conv1\.bn\.weight$",
        r"model.backbone.encoder.convolution.normalization.weight",
    ),
    (
        r"^backbone\.conv1\.bn\.bias$",
        r"model.backbone.encoder.convolution.normalization.bias",
    ),
    (
        r"^backbone\.conv1\.bn\.running_mean$",
        r"model.backbone.encoder.convolution.normalization.running_mean",
    ),
    (
        r"^backbone\.conv1\.bn\.running_var$",
        r"model.backbone.encoder.convolution.normalization.running_var",
    ),
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
PPOCRV5_SERVER_DET_MAPPING = [
    # Backbone stages
    (
        r"^backbone\.stages\.(\d+)\.blocks\.(\d+)\.layers\.(\d+)\.conv(\d)\.bn\.(\w+)$",
        r"model.backbone.encoder.stages.\1.blocks.\2.layers.\3.conv\4.normalization.\5",
    ),
    (
        r"^backbone\.stages\.(\d+)\.blocks\.(\d+)\.layers\.(\d+)\.conv(\d)\.conv\.(\w+)$",
        r"model.backbone.encoder.stages.\1.blocks.\2.layers.\3.conv\4.convolution.\5",
    ),
    (
        r"^backbone\.stages\.(\d+)\.blocks\.(\d+)\.layers\.(\d+)\.bn\.(\w+)$",
        r"model.backbone.encoder.stages.\1.blocks.\2.layers.\3.normalization.\4",
    ),
    (
        r"^backbone\.stages\.(\d+)\.blocks\.(\d+)\.layers\.(\d+)\.conv\.(\w+)$",
        r"model.backbone.encoder.stages.\1.blocks.\2.layers.\3.convolution.\4",
    ),
    # Downsample
    (
        r"^backbone\.stages\.(\d+)\.downsample\.bn\.(\w+)$",
        r"model.backbone.encoder.stages.\1.downsample.normalization.\2",
    ),
    (
        r"^backbone\.stages\.(\d+)\.downsample\.conv\.(\w+)$",
        r"model.backbone.encoder.stages.\1.downsample.convolution.\2",
    ),
    # Aggregation
    (
        r"^backbone\.stages\.(\d+)\.blocks\.(\d+)\.aggregation_squeeze_conv\.bn\.(\w+)$",
        r"model.backbone.encoder.stages.\1.blocks.\2.aggregation.0.normalization.\3",
    ),
    (
        r"^backbone\.stages\.(\d+)\.blocks\.(\d+)\.aggregation_squeeze_conv\.conv\.(\w+)$",
        r"model.backbone.encoder.stages.\1.blocks.\2.aggregation.0.convolution.\3",
    ),
    (
        r"^backbone\.stages\.(\d+)\.blocks\.(\d+)\.aggregation_excitation_conv\.bn\.(\w+)$",
        r"model.backbone.encoder.stages.\1.blocks.\2.aggregation.1.normalization.\3",
    ),
    (
        r"^backbone\.stages\.(\d+)\.blocks\.(\d+)\.aggregation_excitation_conv\.conv\.(\w+)$",
        r"model.backbone.encoder.stages.\1.blocks.\2.aggregation.1.convolution.\3",
    ),
    # Stem
    (
        r"^backbone\.stem\.stem(\d+[ab]?)\.bn\.(\w+)$",
        r"model.backbone.embedder.stem\1.normalization.\2",
    ),
    (
        r"^backbone\.stem\.stem(\d+[ab]?)\.conv\.(\w+)$",
        r"model.backbone.embedder.stem\1.convolution.\2",
    ),
    # Head
    (r"^head\.binarize\.conv1\.(\w+)$", r"head.binarize_head.conv_down.convolution.\1"),
    (r"^head\.binarize\.conv2\.(\w+)$", r"head.binarize_head.conv_up.convolution.\1"),
    (r"^head\.binarize\.conv3\.(\w+)$", r"head.binarize_head.conv_final.\1"),
    (r"^head\.binarize\.conv_bn1\.(\w+)$", r"head.binarize_head.conv_down.norm.\1"),
    (r"^head\.binarize\.conv_bn2\.(\w+)$", r"head.binarize_head.conv_up.norm.\1"),
    # Local Refinement Module
    (
        r"^head\.cbn_layer\.last_3\.conv\.(\w+)$",
        r"head.local_refinement_module.convolution_backbone.convolution.\1",
    ),
    (
        r"^head\.cbn_layer\.last_3\.bn\.(\w+)$",
        r"head.local_refinement_module.convolution_backbone.norm.\1",
    ),
    (
        r"^head\.cbn_layer\.last_1\.(\w+)$",
        r"head.local_refinement_module.convolution_final.\1",
    ),
    # Neck
    (
        r"^neck\.incl(\d+)\.conv1x1_reduce_channel\.(\w+)$",
        lambda m: f"model.neck.intraclass_blocks.{int(m.group(1))-1}.conv_reduce_channel.{m.group(2)}",
    ),
    (
        r"^neck\.incl(\d+)\.conv1x1_return_channel\.(\w+)$",
        lambda m: f"model.neck.intraclass_blocks.{int(m.group(1))-1}.conv_final.convolution.{m.group(2)}",
    ),
    (
        r"^neck\.incl(\d+)\.bn\.(\w+)$",
        lambda m: f"model.neck.intraclass_blocks.{int(m.group(1))-1}.conv_final.norm.{m.group(2)}",
    ),
    (
        r"^neck\.incl(\d+)\.v_layer_7x1\.(\w+)$",
        lambda m: f"model.neck.intraclass_blocks.{int(m.group(1))-1}.vertical_long_to_small_conv_longratio.{m.group(2)}",
    ),
    (
        r"^neck\.incl(\d+)\.v_layer_5x1\.(\w+)$",
        lambda m: f"model.neck.intraclass_blocks.{int(m.group(1))-1}.vertical_long_to_small_conv_midratio.{m.group(2)}",
    ),
    (
        r"^neck\.incl(\d+)\.v_layer_3x1\.(\w+)$",
        lambda m: f"model.neck.intraclass_blocks.{int(m.group(1))-1}.vertical_long_to_small_conv_shortratio.{m.group(2)}",
    ),
    (
        r"^neck\.incl(\d+)\.q_layer_1x7\.(\w+)$",
        lambda m: f"model.neck.intraclass_blocks.{int(m.group(1))-1}.horizontal_small_to_long_conv_longratio.{m.group(2)}",
    ),
    (
        r"^neck\.incl(\d+)\.q_layer_1x5\.(\w+)$",
        lambda m: f"model.neck.intraclass_blocks.{int(m.group(1))-1}.horizontal_small_to_long_conv_midratio.{m.group(2)}",
    ),
    (
        r"^neck\.incl(\d+)\.q_layer_1x3\.(\w+)$",
        lambda m: f"model.neck.intraclass_blocks.{int(m.group(1))-1}.horizontal_small_to_long_conv_shortratio.{m.group(2)}",
    ),
    (
        r"^neck\.incl(\d+)\.c_layer_7x7\.(\w+)$",
        lambda m: f"model.neck.intraclass_blocks.{int(m.group(1))-1}.symmetric_conv_long_longratio.{m.group(2)}",
    ),
    (
        r"^neck\.incl(\d+)\.c_layer_5x5\.(\w+)$",
        lambda m: f"model.neck.intraclass_blocks.{int(m.group(1))-1}.symmetric_conv_long_midratio.{m.group(2)}",
    ),
    (
        r"^neck\.incl(\d+)\.c_layer_3x3\.(\w+)$",
        lambda m: f"model.neck.intraclass_blocks.{int(m.group(1))-1}.symmetric_conv_long_shortratio.{m.group(2)}",
    ),
    # Neck convolutions
    (
        r"^neck\.inp_conv\.(\d+)\.weight$",
        r"model.neck.input_feature_projection_convolution.\1.weight",
    ),
    (
        r"^neck\.ins_conv\.(\d+)\.weight$",
        r"model.neck.input_channel_adjustment_convolution.\1.weight",
    ),
    (
        r"^neck\.pan_lat_conv\.(\d+)\.weight$",
        r"model.neck.path_aggregation_lateral_convolution.\1.weight",
    ),
    (
        r"^neck\.pan_head_conv\.(\d+)\.weight$",
        r"model.neck.path_aggregation_head_convolution.\1.weight",
    ),
]


# RT-DETR family
RTDETR_MAPPING = [
    # === Backbone: HGNetV2 ===
    # Stem
    (
        r"^backbone\.stem\.(\w+)\.conv\.(\w+)$",
        r"model.backbone.model.embedder.\1.convolution.\2",
    ),
    (
        r"^backbone\.stem\.(\w+)\.bn\.(\w+)$",
        r"model.backbone.model.embedder.\1.normalization.\2",
    ),
    # Stages: aggregation
    (
        r"^backbone\.stages\.(\d+)\.blocks\.(\d+)\.aggregation_squeeze_conv\.conv\.(\w+)$",
        r"model.backbone.model.encoder.stages.\1.blocks.\2.aggregation.0.convolution.\3",
    ),
    (
        r"^backbone\.stages\.(\d+)\.blocks\.(\d+)\.aggregation_squeeze_conv\.bn\.(\w+)$",
        r"model.backbone.model.encoder.stages.\1.blocks.\2.aggregation.0.normalization.\3",
    ),
    (
        r"^backbone\.stages\.(\d+)\.blocks\.(\d+)\.aggregation_excitation_conv\.conv\.(\w+)$",
        r"model.backbone.model.encoder.stages.\1.blocks.\2.aggregation.1.convolution.\3",
    ),
    (
        r"^backbone\.stages\.(\d+)\.blocks\.(\d+)\.aggregation_excitation_conv\.bn\.(\w+)$",
        r"model.backbone.model.encoder.stages.\1.blocks.\2.aggregation.1.normalization.\3",
    ),
    # Stages: layers with convN sub-modules (conv1.conv, conv1.bn, etc.)
    (
        r"^backbone\.stages\.(\d+)\.blocks\.(\d+)\.layers\.(\d+)\.conv(\d+)\.conv\.(\w+)$",
        r"model.backbone.model.encoder.stages.\1.blocks.\2.layers.\3.conv\4.convolution.\5",
    ),
    (
        r"^backbone\.stages\.(\d+)\.blocks\.(\d+)\.layers\.(\d+)\.conv(\d+)\.bn\.(\w+)$",
        r"model.backbone.model.encoder.stages.\1.blocks.\2.layers.\3.conv\4.normalization.\5",
    ),
    # Stages: layers with direct conv/bn (no sub-number)
    (
        r"^backbone\.stages\.(\d+)\.blocks\.(\d+)\.layers\.(\d+)\.conv\.(\w+)$",
        r"model.backbone.model.encoder.stages.\1.blocks.\2.layers.\3.convolution.\4",
    ),
    (
        r"^backbone\.stages\.(\d+)\.blocks\.(\d+)\.layers\.(\d+)\.bn\.(\w+)$",
        r"model.backbone.model.encoder.stages.\1.blocks.\2.layers.\3.normalization.\4",
    ),
    # Stages: downsample
    (
        r"^backbone\.stages\.(\d+)\.downsample\.conv\.(\w+)$",
        r"model.backbone.model.encoder.stages.\1.downsample.convolution.\2",
    ),
    (
        r"^backbone\.stages\.(\d+)\.downsample\.bn\.(\w+)$",
        r"model.backbone.model.encoder.stages.\1.downsample.normalization.\2",
    ),
    # === Neck → model.encoder ===
    # Encoder input_proj (neck side)
    (r"^neck\.input_proj\.", r"model.encoder_input_proj."),
    # Encoder attention layers
    (
        r"^neck\.encoder\.(\d+)\.layers\.(\d+)\.linear1\.",
        r"model.encoder.encoder.\1.layers.\2.fc1.",
    ),
    (
        r"^neck\.encoder\.(\d+)\.layers\.(\d+)\.linear2\.",
        r"model.encoder.encoder.\1.layers.\2.fc2.",
    ),
    (
        r"^neck\.encoder\.(\d+)\.layers\.(\d+)\.norm1\.",
        r"model.encoder.encoder.\1.layers.\2.self_attn_layer_norm.",
    ),
    (
        r"^neck\.encoder\.(\d+)\.layers\.(\d+)\.norm2\.",
        r"model.encoder.encoder.\1.layers.\2.final_layer_norm.",
    ),
    (r"^neck\.encoder\.(\d+)\.layers\.", r"model.encoder.encoder.\1.layers."),
    # Downsample convs (.bn → .norm)
    (
        r"^neck\.downsample_convs\.(\d+)\.bn\.",
        r"model.encoder.downsample_convs.\1.norm.",
    ),
    (r"^neck\.downsample_convs\.", r"model.encoder.downsample_convs."),
    # Lateral convs (.bn → .norm)
    (r"^neck\.lateral_convs\.(\d+)\.bn\.", r"model.encoder.lateral_convs.\1.norm."),
    (r"^neck\.lateral_convs\.", r"model.encoder.lateral_convs."),
    # FPN blocks (.bn → .norm)
    (
        r"^neck\.fpn_blocks\.(\d+)\.((?:bottlenecks\.\d+\.)?conv\d)\.bn\.",
        r"model.encoder.fpn_blocks.\1.\2.norm.",
    ),
    (r"^neck\.fpn_blocks\.", r"model.encoder.fpn_blocks."),
    # PAN blocks (.bn → .norm)
    (
        r"^neck\.pan_blocks\.(\d+)\.((?:bottlenecks\.\d+\.)?conv\d)\.bn\.",
        r"model.encoder.pan_blocks.\1.\2.norm.",
    ),
    (r"^neck\.pan_blocks\.", r"model.encoder.pan_blocks."),
    # Mask feature head (DocLayoutV3 only)
    (
        r"^neck\.mask_feat_head\.scale_heads\.(\d+)\.(\d+)\.0\.conv\.",
        r"model.encoder.mask_feature_head.scale_heads.\1.layers.\2.convolution.",
    ),
    (
        r"^neck\.mask_feat_head\.scale_heads\.(\d+)\.(\d+)\.0\.bn\.",
        r"model.encoder.mask_feature_head.scale_heads.\1.layers.\2.normalization.",
    ),
    (
        r"^neck\.mask_feat_head\.output_conv\.conv\.",
        r"model.encoder.mask_feature_head.output_conv.convolution.",
    ),
    (
        r"^neck\.mask_feat_head\.output_conv\.bn\.",
        r"model.encoder.mask_feature_head.output_conv.normalization.",
    ),
    # Encoder mask lateral/output (DocLayoutV3 only)
    (
        r"^neck\.enc_mask_lateral\.conv\.",
        r"model.encoder.encoder_mask_lateral.convolution.",
    ),
    (
        r"^neck\.enc_mask_lateral\.bn\.",
        r"model.encoder.encoder_mask_lateral.normalization.",
    ),
    (
        r"^neck\.enc_mask_output\.0\.conv\.",
        r"model.encoder.encoder_mask_output.base_conv.convolution.",
    ),
    (
        r"^neck\.enc_mask_output\.0\.bn\.",
        r"model.encoder.encoder_mask_output.base_conv.normalization.",
    ),
    (r"^neck\.enc_mask_output\.1\.", r"model.encoder.encoder_mask_output.conv."),
    # === Transformer → model.decoder + heads ===
    # Decoder input_proj
    (
        r"^transformer\.input_proj\.(\d+)\.conv\.(\w+)$",
        r"model.decoder_input_proj.\1.0.\2",
    ),
    (
        r"^transformer\.input_proj\.(\d+)\.norm\.(\w+)$",
        r"model.decoder_input_proj.\1.1.\2",
    ),
    # Decoder layers: cross_attn → encoder_attn
    (
        r"^transformer\.decoder\.layers\.(\d+)\.cross_attn\.",
        r"model.decoder.layers.\1.encoder_attn.",
    ),
    # Decoder layers: linear1/2 → fc1/2
    (
        r"^transformer\.decoder\.layers\.(\d+)\.linear1\.",
        r"model.decoder.layers.\1.fc1.",
    ),
    (
        r"^transformer\.decoder\.layers\.(\d+)\.linear2\.",
        r"model.decoder.layers.\1.fc2.",
    ),
    # Decoder layers: norm1/2/3 → named layer norms
    (
        r"^transformer\.decoder\.layers\.(\d+)\.norm1\.",
        r"model.decoder.layers.\1.self_attn_layer_norm.",
    ),
    (
        r"^transformer\.decoder\.layers\.(\d+)\.norm2\.",
        r"model.decoder.layers.\1.encoder_attn_layer_norm.",
    ),
    (
        r"^transformer\.decoder\.layers\.(\d+)\.norm3\.",
        r"model.decoder.layers.\1.final_layer_norm.",
    ),
    # Decoder layers: self_attn (passthrough after prefix)
    (r"^transformer\.decoder\.", r"model.decoder."),
    # Query pos head
    (r"^transformer\.query_pos_head\.", r"model.decoder.query_pos_head."),
    # Heads (DocLayoutV3 naming)
    (r"^transformer\.bbox_head\.", r"model.enc_bbox_head."),
    (r"^transformer\.score_head\.", r"model.enc_score_head."),
    # Heads (standard RT-DETR naming)
    (r"^transformer\.enc_bbox_head\.", r"model.enc_bbox_head."),
    (r"^transformer\.enc_score_head\.", r"model.enc_score_head."),
    (r"^transformer\.dec_bbox_head\.", r"model.decoder.bbox_embed."),
    (r"^transformer\.dec_score_head\.", r"model.decoder.class_embed."),
    # Common heads
    (r"^transformer\.enc_output\.", r"model.enc_output."),
    (r"^transformer\.mask_query_head\.", r"model.mask_query_head."),
    (r"^transformer\.dec_global_pointer\.", r"model.decoder_global_pointer."),
    (r"^transformer\.dec_norm\.", r"model.decoder_norm."),
    (r"^transformer\.dec_order_head\.", r"model.decoder_order_head."),
    (r"^transformer\.denoising_class_embed\.", r"model.denoising_class_embed."),
]

# PP-DocLayoutV2 extends RTDETR with reading_order mapping
PP_DOCLAYOUTV2_MAPPING = RTDETR_MAPPING + [
    # Reading order: prefix rename (applied first via iterative mapping)
    (
        r"^transformer\.reading_order_predictor\.",
        r"reading_order.",
    ),
    # Reading order: LayerNorm → norm (applied on second iteration)
    (r"reading_order\.embeddings\.LayerNorm\.", r"reading_order.embeddings.norm."),
    (
        r"reading_order\.encoder\.layer\.(\d+)\.attention\.output\.LayerNorm\.",
        r"reading_order.encoder.layer.\1.attention.output.norm.",
    ),
    (
        r"reading_order\.encoder\.layer\.(\d+)\.output\.LayerNorm\.",
        r"reading_order.encoder.layer.\1.output.norm.",
    ),
]


# UVDoc
UVDOC_MAPPING = [
    # ResNet Down layers
    (
        r"^resnet_down\.layer1\.(\d+)\.conv1\.(\w+)$",
        r"backbone.resnet.resnet_down.0.layers.\1.conv_start.convolution.\2",
    ),
    (
        r"^resnet_down\.layer1\.(\d+)\.conv2\.(\w+)$",
        r"backbone.resnet.resnet_down.0.layers.\1.conv_final.convolution.\2",
    ),
    (
        r"^resnet_down\.layer1\.(\d+)\.conv1\.0\.(\w+)$",
        r"backbone.resnet.resnet_down.0.layers.\1.conv_start.convolution.\2",
    ),
    (
        r"^resnet_down\.layer1\.(\d+)\.conv2\.0\.(\w+)$",
        r"backbone.resnet.resnet_down.0.layers.\1.conv_final.convolution.\2",
    ),
    (
        r"^resnet_down\.layer1\.(\d+)\.bn1\.(\w+)$",
        r"backbone.resnet.resnet_down.0.layers.\1.conv_start.normalization.\2",
    ),
    (
        r"^resnet_down\.layer1\.(\d+)\.bn2\.(\w+)$",
        r"backbone.resnet.resnet_down.0.layers.\1.conv_final.normalization.\2",
    ),
    (
        r"^resnet_down\.layer2\.(\d+)\.conv1\.(\w+)$",
        r"backbone.resnet.resnet_down.1.layers.\1.conv_start.convolution.\2",
    ),
    (
        r"^resnet_down\.layer2\.(\d+)\.conv2\.(\w+)$",
        r"backbone.resnet.resnet_down.1.layers.\1.conv_final.convolution.\2",
    ),
    (
        r"^resnet_down\.layer2\.(\d+)\.conv1\.0\.(\w+)$",
        r"backbone.resnet.resnet_down.1.layers.\1.conv_start.convolution.\2",
    ),
    (
        r"^resnet_down\.layer2\.(\d+)\.conv2\.0\.(\w+)$",
        r"backbone.resnet.resnet_down.1.layers.\1.conv_final.convolution.\2",
    ),
    (
        r"^resnet_down\.layer2\.(\d+)\.bn1\.(\w+)$",
        r"backbone.resnet.resnet_down.1.layers.\1.conv_start.normalization.\2",
    ),
    (
        r"^resnet_down\.layer2\.(\d+)\.bn2\.(\w+)$",
        r"backbone.resnet.resnet_down.1.layers.\1.conv_final.normalization.\2",
    ),
    (
        r"^resnet_down\.layer2\.(\d+)\.downsample\.0\.(\w+)$",
        r"backbone.resnet.resnet_down.1.layers.\1.conv_down.convolution.\2",
    ),
    (
        r"^resnet_down\.layer2\.(\d+)\.downsample\.1\.(\w+)$",
        r"backbone.resnet.resnet_down.1.layers.\1.conv_down.normalization.\2",
    ),
    (
        r"^resnet_down\.layer3\.(\d+)\.conv1\.(\w+)$",
        r"backbone.resnet.resnet_down.2.layers.\1.conv_start.convolution.\2",
    ),
    (
        r"^resnet_down\.layer3\.(\d+)\.conv2\.(\w+)$",
        r"backbone.resnet.resnet_down.2.layers.\1.conv_final.convolution.\2",
    ),
    (
        r"^resnet_down\.layer3\.(\d+)\.conv1\.0\.(\w+)$",
        r"backbone.resnet.resnet_down.2.layers.\1.conv_start.convolution.\2",
    ),
    (
        r"^resnet_down\.layer3\.(\d+)\.conv2\.0\.(\w+)$",
        r"backbone.resnet.resnet_down.2.layers.\1.conv_final.convolution.\2",
    ),
    (
        r"^resnet_down\.layer3\.(\d+)\.bn1\.(\w+)$",
        r"backbone.resnet.resnet_down.2.layers.\1.conv_start.normalization.\2",
    ),
    (
        r"^resnet_down\.layer3\.(\d+)\.bn2\.(\w+)$",
        r"backbone.resnet.resnet_down.2.layers.\1.conv_final.normalization.\2",
    ),
    (
        r"^resnet_down\.layer3\.(\d+)\.downsample\.0\.(\w+)$",
        r"backbone.resnet.resnet_down.2.layers.\1.conv_down.convolution.\2",
    ),
    (
        r"^resnet_down\.layer3\.(\d+)\.downsample\.1\.(\w+)$",
        r"backbone.resnet.resnet_down.2.layers.\1.conv_down.normalization.\2",
    ),
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
    (
        r"^out_point_positions2D\.0\.",
        r"head.out_point_positions2D.conv_down.convolution.",
    ),
    (
        r"^out_point_positions2D\.1\.",
        r"head.out_point_positions2D.conv_down.normalization.",
    ),
    (
        r"^out_point_positions2D\.2\._weight$",
        r"head.out_point_positions2D.conv_down.activation.weight",
    ),
    (r"^out_point_positions2D\.3\.", r"head.out_point_positions2D.conv_up."),
    # Bridge connector
    (r"^bridge_concat\.0\.", r"head.bridge_connector.convolution."),
    (r"^bridge_concat\.1\.", r"head.bridge_connector.normalization."),
]

# PP-Chart2Table (GOT-OCR2 + Qwen2 VLM)
PP_CHART2TABLE_MAPPING = [
    # Multi-modal projector: specific renames before general vision_tower_high rule
    (
        r"^qwen2\.vision_tower_high\.net_2\.weight$",
        "model.multi_modal_projector.conv_upsampler1.weight",
    ),
    (
        r"^qwen2\.vision_tower_high\.net_3\.weight$",
        "model.multi_modal_projector.conv_upsampler2.weight",
    ),
    (
        r"^qwen2\.mm_projector_vary\.",
        "model.multi_modal_projector.multimodal_projector.",
    ),
    # Vision tower: norm1/norm2 must be renamed before the general blocks rule
    (
        r"^qwen2\.vision_tower_high\.blocks\.(\d+)\.norm1\.",
        r"model.vision_tower.layers.\1.layer_norm1.",
    ),
    (
        r"^qwen2\.vision_tower_high\.blocks\.(\d+)\.norm2\.",
        r"model.vision_tower.layers.\1.layer_norm2.",
    ),
    (r"^qwen2\.vision_tower_high\.blocks\.(\d+)\.", r"model.vision_tower.layers.\1."),
    (
        r"^qwen2\.vision_tower_high\.patch_embed\.proj\.",
        "model.vision_tower.patch_embed.projection.",
    ),
    (r"^qwen2\.vision_tower_high\.pos_embed$", "model.vision_tower.pos_embed"),
    (r"^qwen2\.vision_tower_high\.neck\.0\.", "model.vision_tower.neck.conv1."),
    (r"^qwen2\.vision_tower_high\.neck\.1\.", "model.vision_tower.neck.layer_norm1."),
    (r"^qwen2\.vision_tower_high\.neck\.2\.", "model.vision_tower.neck.conv2."),
    (r"^qwen2\.vision_tower_high\.neck\.3\.", "model.vision_tower.neck.layer_norm2."),
    # Language model
    (r"^qwen2\.layers\.(\d+)\.", r"model.language_model.layers.\1."),
    (r"^qwen2\.norm\.", "model.language_model.norm."),
]

# Shared SVTR encoder + CTC head mapping (used by both mobile and server rec)
_SVTR_CTC_HEAD_MAPPING = [
    (
        r"^head\.ctc_encoder\.encoder\.conv1\.conv\.",
        r"head.encoder.conv_block.0.convolution.",
    ),
    (
        r"^head\.ctc_encoder\.encoder\.conv1\.norm\.",
        r"head.encoder.conv_block.0.normalization.",
    ),
    (
        r"^head\.ctc_encoder\.encoder\.conv2\.conv\.",
        r"head.encoder.conv_block.1.convolution.",
    ),
    (
        r"^head\.ctc_encoder\.encoder\.conv2\.norm\.",
        r"head.encoder.conv_block.1.normalization.",
    ),
    (
        r"^head\.ctc_encoder\.encoder\.conv3\.conv\.",
        r"head.encoder.conv_block.2.convolution.",
    ),
    (
        r"^head\.ctc_encoder\.encoder\.conv3\.norm\.",
        r"head.encoder.conv_block.2.normalization.",
    ),
    (
        r"^head\.ctc_encoder\.encoder\.conv4\.conv\.",
        r"head.encoder.conv_block.3.convolution.",
    ),
    (
        r"^head\.ctc_encoder\.encoder\.conv4\.norm\.",
        r"head.encoder.conv_block.3.normalization.",
    ),
    (
        r"^head\.ctc_encoder\.encoder\.conv1x1\.conv\.",
        r"head.encoder.conv_block.4.convolution.",
    ),
    (
        r"^head\.ctc_encoder\.encoder\.conv1x1\.norm\.",
        r"head.encoder.conv_block.4.normalization.",
    ),
    (
        r"^head\.ctc_encoder\.encoder\.svtr_block\.(\d+)\.norm1\.",
        r"head.encoder.svtr_block.\1.layer_norm1.",
    ),
    (
        r"^head\.ctc_encoder\.encoder\.svtr_block\.(\d+)\.norm2\.",
        r"head.encoder.svtr_block.\1.layer_norm2.",
    ),
    (
        r"^head\.ctc_encoder\.encoder\.svtr_block\.(\d+)\.mixer\.qkv\.",
        r"head.encoder.svtr_block.\1.self_attn.qkv.",
    ),
    (
        r"^head\.ctc_encoder\.encoder\.svtr_block\.(\d+)\.mixer\.proj\.",
        r"head.encoder.svtr_block.\1.self_attn.projection.",
    ),
    (
        r"^head\.ctc_encoder\.encoder\.svtr_block\.(\d+)\.mlp\.",
        r"head.encoder.svtr_block.\1.mlp.",
    ),
    (r"^head\.ctc_encoder\.encoder\.norm\.", r"head.encoder.norm."),
    (r"^head\.ctc_head\.fc\.", r"head.head."),
]


# PP-OCRv5 rec models
PPOCRV5_MOBILE_REC_MAPPING = [
    (
        r"^backbone\.conv1\.conv\.(\w+)$",
        r"model.backbone.encoder.convolution.convolution.\1",
    ),
    (
        r"^backbone\.conv1\.bn\.(\w+)$",
        r"model.backbone.encoder.convolution.normalization.\1",
    ),
    (r"^backbone\.blocks2\.(\d+)\.", r"model.backbone.encoder.blocks.0.layers.\1."),
    (r"^backbone\.blocks3\.(\d+)\.", r"model.backbone.encoder.blocks.1.layers.\1."),
    (r"^backbone\.blocks4\.(\d+)\.", r"model.backbone.encoder.blocks.2.layers.\1."),
    (r"^backbone\.blocks5\.(\d+)\.", r"model.backbone.encoder.blocks.3.layers.\1."),
    (r"^backbone\.blocks6\.(\d+)\.", r"model.backbone.encoder.blocks.4.layers.\1."),
    (r"\.dw_conv\.", r".depthwise_convolution."),
    (r"\.pw_conv\.", r".pointwise_convolution."),
    (r"\.conv_kxk\.(\d+)\.conv\.", r".conv_symmetric.\1.convolution."),
    (r"\.conv_kxk\.(\d+)\.bn\.", r".conv_symmetric.\1.normalization."),
    (r"\.conv_1x1\.conv\.", r".conv_small_symmetric.convolution."),
    (r"\.conv_1x1\.bn\.", r".conv_small_symmetric.normalization."),
    (r"\.se\.conv1\.", r".squeeze_excitation_module.convolutions.0."),
    (r"\.se\.conv2\.", r".squeeze_excitation_module.convolutions.2."),
] + _SVTR_CTC_HEAD_MAPPING

PPOCRV5_SERVER_REC_MAPPING = [
    (
        r"^backbone\.stem\.(\w+)\.conv\.(\w+)$",
        r"model.backbone.embedder.\1.convolution.\2",
    ),
    (
        r"^backbone\.stem\.(\w+)\.bn\.(\w+)$",
        r"model.backbone.embedder.\1.normalization.\2",
    ),
    (r"^backbone\.stages\.", r"model.backbone.encoder.stages."),
    (r"\.aggregation_squeeze_conv\.conv\.", r".aggregation.0.convolution."),
    (r"\.aggregation_squeeze_conv\.bn\.", r".aggregation.0.normalization."),
    (r"\.aggregation_excitation_conv\.conv\.", r".aggregation.1.convolution."),
    (r"\.aggregation_excitation_conv\.bn\.", r".aggregation.1.normalization."),
    (r"\.downsample\.conv\.", r".downsample.convolution."),
    (r"\.downsample\.bn\.", r".downsample.normalization."),
    (r"\.layers\.(\d+)\.conv(\d)\.conv\.", r".layers.\1.conv\2.convolution."),
    (r"\.layers\.(\d+)\.conv(\d)\.bn\.", r".layers.\1.conv\2.normalization."),
    (r"\.layers\.(\d+)\.conv\.", r".layers.\1.convolution."),
    (r"\.layers\.(\d+)\.bn\.", r".layers.\1.normalization."),
] + _SVTR_CTC_HEAD_MAPPING


# SLANeXt
SLANEXT_MAPPING = [
    # net_2 must be renamed before the general vision_tower_high rule
    (r"backbone\.vision_tower_high\.net_2\.", "backbone.post_conv."),
    (r"vision_tower_high\.", "vision_tower."),
    (r"\.blocks\.", ".layers."),
    (r"\.norm1\.", ".layer_norm1."),
    (r"\.norm2\.", ".layer_norm2."),
    (r"patch_embed\.proj\.", "patch_embed.projection."),
    (r"\.neck\.0\.", ".neck.conv1."),
    (r"\.neck\.1\.", ".neck.layer_norm1."),
    (r"\.neck\.2\.", ".neck.conv2."),
    (r"\.neck\.3\.", ".neck.layer_norm2."),
    (r"structure_generator\.0\.", "structure_generator.fc1."),
    (r"structure_generator\.1\.", "structure_generator.fc2."),
    (r"\.i2h\.", ".input_to_hidden."),
    (r"\.h2h\.", ".hidden_to_hidden."),
]


# Keys to drop during conversion (training-only / tied weights)
UVDOC_DROP_PREFIXES = [
    "out_point_positions3D.",
]

PP_CHART2TABLE_DROP_PREFIXES = [
    "qwen2.embed_tokens.",  # tied to lm_head.weight (tie_word_embeddings=True)
]

PP_DOCLAYOUTV2_DROP_PREFIXES = [
    "transformer.reading_order_predictor.global_agg.",
    "transformer.reading_order_predictor.global_gate.",
    "transformer.reading_order_predictor.global_visual_proj.",
    "transformer.reading_order_predictor.visual_features_projection.",
]

SLANEXT_DROP_PREFIXES = [
    "backbone.vision_tower_high.net_3.",
    "head.loc_generator.",
]

REC_DROP_PREFIXES = [
    "head.before_gtc.",
    "head.gtc_head.",
    "head.encoder_reshape.",
]

SERVER_REC_DROP_PREFIXES = REC_DROP_PREFIXES + [
    "backbone.fc.",
    "backbone.last_conv.",
]

MOBILE_DET_DROP_PREFIXES = [
    "head.thresh.",
]

SERVER_DET_DROP_PREFIXES = [
    "head.thresh.",
    "backbone.last_conv.",
]
