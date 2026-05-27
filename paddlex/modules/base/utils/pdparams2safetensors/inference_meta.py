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

"""Inference metadata for pdparams2safetensors conversion.

Provides per-model inference.yml content (label_list, PreProcess,
PostProcess, etc.) and preprocessor_config.json content.
"""

from pathlib import Path

from .....utils import logging

# Label lists
_LABEL_DOC_ORI = ["0", "90", "180", "270"]
_LABEL_TABLE_CLS = ["wired_table", "wireless_table"]
_LABEL_TEXTLINE_ORI = ["0_degree", "180_degree"]

_LABEL_TABLE_CELL_DET = ["cell"]
_LABEL_DOC_BLOCK_LAYOUT = ["Region"]
_LABEL_DOC_LAYOUT_PLUS = [
    "paragraph_title",
    "image",
    "text",
    "number",
    "abstract",
    "content",
    "figure_title",
    "formula",
    "table",
    "reference",
    "doc_title",
    "footnote",
    "header",
    "algorithm",
    "footer",
    "seal",
    "chart",
    "formula_number",
    "aside_text",
    "reference_content",
]
_LABEL_DOC_LAYOUT_V2V3 = [
    "abstract",
    "algorithm",
    "aside_text",
    "chart",
    "content",
    "display_formula",
    "doc_title",
    "figure_title",
    "footer",
    "footer_image",
    "footnote",
    "formula_number",
    "header",
    "header_image",
    "image",
    "inline_formula",
    "number",
    "paragraph_title",
    "reference",
    "reference_content",
    "seal",
    "table",
    "text",
    "vertical_text",
    "vision_footnote",
]


# PreProcess / PostProcess templates
_CLS_PREPROCESS_224 = {
    "PreProcess": {
        "transform_ops": [
            {"ResizeImage": {"resize_short": 256}},
            {"CropImage": {"size": 224}},
            {
                "NormalizeImage": {
                    "channel_num": 3,
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                    "scale": 1.0 / 255,
                    "order": "",
                }
            },
            {"ToCHWImage": None},
        ]
    },
}

_CLS_PREPROCESS_TEXTLINE = {
    "PreProcess": {
        "transform_ops": [
            {"ResizeImage": {"size": [160, 80]}},
            {
                "NormalizeImage": {
                    "channel_num": 3,
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                    "scale": 1.0 / 255,
                    "order": "",
                }
            },
            {"ToCHWImage": None},
        ]
    },
}

_DET_PREPROCESS = {
    "transform_ops": [
        {"DecodeImage": {"channel_first": False, "img_mode": "BGR"}},
        {"DetLabelEncode": None},
        {"DetResizeForTest": {"resize_long": 960}},
        {
            "NormalizeImage": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "scale": "1./255.",
                "order": "hwc",
            }
        },
        {"ToCHWImage": None},
        {"KeepKeys": {"keep_keys": ["image", "shape", "polys", "ignore_tags"]}},
    ]
}

_DET_POSTPROCESS = {
    "PreProcess": _DET_PREPROCESS,
    "PostProcess": {
        "name": "DBPostProcess",
        "thresh": 0.3,
        "box_thresh": 0.6,
        "max_candidates": 1000,
        "unclip_ratio": 1.5,
    },
}

_RTDETR_PREPROCESS_800 = [
    {"type": "Resize", "target_size": [800, 800], "keep_ratio": False, "interp": 2},
    {
        "type": "NormalizeImage",
        "mean": [0.0, 0.0, 0.0],
        "std": [1.0, 1.0, 1.0],
        "norm_type": "none",
    },
    {"type": "Permute"},
]

_RTDETR_PREPROCESS_640 = [
    {"type": "Resize", "target_size": [640, 640], "keep_ratio": False, "interp": 2},
    {
        "type": "NormalizeImage",
        "mean": [0.0, 0.0, 0.0],
        "std": [1.0, 1.0, 1.0],
        "norm_type": "none",
    },
    {"type": "Permute"},
]

_REC_IMAGE_SHAPE = [3, 48, 320]


# Hpi (TensorRT dynamic shape) templates

def _hpi_simple(input_name, shapes):
    """Build Hpi section with a single input (cls/det/rec/uvdoc)."""
    tds = {input_name: shapes}
    return {
        "backend_configs": {
            "paddle_infer": {"trt_dynamic_shapes": tds},
            "tensorrt": {"dynamic_shapes": tds},
        }
    }


def _hpi_rtdetr(image_size):
    """Build Hpi for RT-DETR models (im_shape + image + scale_factor)."""
    b = 8
    tds = {
        "im_shape": [[1, 2], [1, 2], [b, 2]],
        "image": [
            [1, 3, image_size, image_size],
            [1, 3, image_size, image_size],
            [b, 3, image_size, image_size],
        ],
        "scale_factor": [[1, 2], [1, 2], [b, 2]],
    }
    sf_data = {
        "im_shape": [
            [image_size, image_size],
            [image_size, image_size],
            [image_size] * (2 * b),
        ],
        "scale_factor": [[2, 2], [1, 1], [0.67] * (2 * b)],
    }
    return {
        "backend_configs": {
            "paddle_infer": {
                "trt_dynamic_shapes": tds,
                "trt_dynamic_shape_input_data": sf_data,
            },
            "tensorrt": {"dynamic_shapes": tds},
        }
    }


def _hpi_doclayoutv3(image_size):
    """Build Hpi for PP-DocLayoutV3 (image + scale_factor, no im_shape)."""
    b = 8
    tds = {
        "image": [
            [1, 3, image_size, image_size],
            [1, 3, image_size, image_size],
            [b, 3, image_size, image_size],
        ],
        "scale_factor": [[1, 2], [1, 2], [b, 2]],
    }
    sf_data = {
        "scale_factor": [[2, 2], [1, 1], [0.67] * (2 * b)],
    }
    return {
        "backend_configs": {
            "paddle_infer": {
                "trt_dynamic_shapes": tds,
                "trt_dynamic_shape_input_data": sf_data,
            },
            "tensorrt": {"dynamic_shapes": tds},
        }
    }


# RT-DETR/DocLayout extra top-level keys
_RTDETR_EXTRA_KEYS = {
    "mode": "paddle",
    "metric": "COCO",
    "use_dynamic_shape": False,
    "arch": "DETR",
    "min_subgraph_size": 3,
}


# Per-model inference metadata
def _meta_cls_doc_ori():
    return {
        "Hpi": _hpi_simple("x", [[1, 3, 224, 224], [1, 3, 224, 224], [8, 3, 224, 224]]),
        **_CLS_PREPROCESS_224,
        "PostProcess": {"Topk": {"topk": 1, "label_list": _LABEL_DOC_ORI}},
    }


def _meta_cls_table():
    return {
        "Hpi": _hpi_simple("x", [[1, 3, 224, 224], [1, 3, 224, 224], [8, 3, 224, 224]]),
        **_CLS_PREPROCESS_224,
        "PostProcess": {"Topk": {"topk": 5, "label_list": _LABEL_TABLE_CLS}},
    }


def _meta_cls_textline():
    return {
        "Hpi": _hpi_simple("x", [[1, 3, 80, 160], [1, 3, 80, 160], [8, 3, 80, 160]]),
        **_CLS_PREPROCESS_TEXTLINE,
        "PostProcess": {"Topk": {"topk": 1, "label_list": _LABEL_TEXTLINE_ORI}},
    }


def _meta_det():
    return {
        "Hpi": _hpi_simple("x", [[1, 3, 32, 32], [1, 3, 736, 736], [1, 3, 4000, 4000]]),
        **_DET_POSTPROCESS,
    }


def _meta_det_rtdetr(labels, preprocess, image_size):
    return {
        **_RTDETR_EXTRA_KEYS,
        "draw_threshold": 0.5,
        "Preprocess": preprocess,
        "label_list": labels,
        "Hpi": _hpi_rtdetr(image_size),
    }


def _meta_doclayoutv3():
    return {
        **_RTDETR_EXTRA_KEYS,
        "draw_threshold": 0.5,
        "Preprocess": _RTDETR_PREPROCESS_800,
        "label_list": _LABEL_DOC_LAYOUT_V2V3,
        "Hpi": _hpi_doclayoutv3(800),
    }


def _meta_rec():
    return {
        "Hpi": _hpi_simple("x", [[1, 3, 48, 160], [1, 3, 48, 320], [8, 3, 48, 3200]]),
        "PreProcess": {
            "transform_ops": [
                {"DecodeImage": {"channel_first": False, "img_mode": "BGR"}},
                {"MultiLabelEncode": {"gtc_encode": "NRTRLabelEncode"}},
                {"RecResizeImg": {"image_shape": _REC_IMAGE_SHAPE}},
                {
                    "KeepKeys": {
                        "keep_keys": [
                            "image",
                            "label_ctc",
                            "label_gtc",
                            "length",
                            "valid_ratio",
                        ]
                    }
                },
            ]
        },
        "PostProcess": {
            "name": "CTCLabelDecode",
        },
    }


_SLANEXT_CHARACTER_DICT = [
    "<thead>",
    "</thead>",
    "<tbody>",
    "</tbody>",
    "<tr>",
    "</tr>",
    "<td>",
    "<td",
    ">",
    "</td>",
    ' colspan="2"',
    ' colspan="3"',
    ' colspan="4"',
    ' colspan="5"',
    ' colspan="6"',
    ' colspan="7"',
    ' colspan="8"',
    ' colspan="9"',
    ' colspan="10"',
    ' colspan="11"',
    ' colspan="12"',
    ' colspan="13"',
    ' colspan="14"',
    ' colspan="15"',
    ' colspan="16"',
    ' colspan="17"',
    ' colspan="18"',
    ' colspan="19"',
    ' colspan="20"',
    ' rowspan="2"',
    ' rowspan="3"',
    ' rowspan="4"',
    ' rowspan="5"',
    ' rowspan="6"',
    ' rowspan="7"',
    ' rowspan="8"',
    ' rowspan="9"',
    ' rowspan="10"',
    ' rowspan="11"',
    ' rowspan="12"',
    ' rowspan="13"',
    ' rowspan="14"',
    ' rowspan="15"',
    ' rowspan="16"',
    ' rowspan="17"',
    ' rowspan="18"',
    ' rowspan="19"',
    ' rowspan="20"',
]


def _meta_slanext(model_name):
    return {
        "Hpi": _hpi_simple("x", [[1, 3, 512, 512], [1, 3, 512, 512], [1, 3, 512, 512]]),
        "PreProcess": {
            "transform_ops": [
                {"DecodeImage": {"channel_first": False, "img_mode": "BGR"}},
                {
                    "TableLabelEncode": {
                        "learn_empty_box": False,
                        "loc_reg_num": 8,
                        "max_text_length": 500,
                        "merge_no_span_structure": True,
                        "replace_empty_cell_token": False,
                    }
                },
                {
                    "TableBoxEncode": {
                        "in_box_format": "xyxyxyxy",
                        "out_box_format": "xyxyxyxy",
                    }
                },
                {"ResizeTableImage": {"max_len": 512, "resize_bboxes": True}},
                {
                    "NormalizeImage": {
                        "mean": [0.485, 0.456, 0.406],
                        "order": "hwc",
                        "scale": "1./255.",
                        "std": [0.229, 0.224, 0.225],
                    }
                },
                {"PaddingTableImage": {"size": [512, 512]}},
                {"ToCHWImage": None},
                {
                    "KeepKeys": {
                        "keep_keys": [
                            "image",
                            "structure",
                            "bboxes",
                            "bbox_masks",
                            "length",
                            "shape",
                        ]
                    }
                },
            ]
        },
        "PostProcess": {
            "name": "TableLabelDecode",
            "merge_no_span_structure": True,
            "character_dict": _SLANEXT_CHARACTER_DICT,
        },
    }


def _meta_slanet(model_name):
    # Max-batch dim in the HPI hint mirrors the uploaded inference.yml:
    # SLANet uses batch=8; SLANet_plus uses batch=1.
    max_batch = 8 if model_name == "SLANet" else 1
    return {
        "Hpi": _hpi_simple(
            "x",
            [
                [1, 3, 32, 32],
                [1, 3, 64, 448],
                [max_batch, 3, 488, 488],
            ],
        ),
        "PreProcess": {
            "transform_ops": [
                {"DecodeImage": {"channel_first": False, "img_mode": "BGR"}},
                {
                    "TableLabelEncode": {
                        "learn_empty_box": False,
                        "loc_reg_num": 8,
                        "max_text_length": 500,
                        "merge_no_span_structure": True,
                        "replace_empty_cell_token": False,
                    }
                },
                {
                    "TableBoxEncode": {
                        "in_box_format": "xyxyxyxy",
                        "out_box_format": "xyxyxyxy",
                    }
                },
                {"ResizeTableImage": {"max_len": 488}},
                {
                    "NormalizeImage": {
                        "mean": [0.485, 0.456, 0.406],
                        "order": "hwc",
                        "scale": "1./255.",
                        "std": [0.229, 0.224, 0.225],
                    }
                },
                {"PaddingTableImage": {"size": [488, 488]}},
                {"ToCHWImage": None},
                {
                    "KeepKeys": {
                        "keep_keys": [
                            "image",
                            "structure",
                            "bboxes",
                            "bbox_masks",
                            "length",
                            "shape",
                        ]
                    }
                },
            ]
        },
        "PostProcess": {
            "name": "TableLabelDecode",
            "merge_no_span_structure": True,
            "character_dict": _SLANEXT_CHARACTER_DICT,
        },
    }


def _meta_uvdoc():
    return {
        "Hpi": _hpi_simple(
            "img", [[1, 3, 128, 64], [1, 3, 256, 128], [8, 3, 512, 256]]
        ),
    }


def _meta_chart2table():
    return {
        "mode": "paddle",
    }


def _meta_pp_formulanet(model_name):
    """Build inference.yml metadata for PP-FormulaNet-L / PP-FormulaNet_plus-L.

    The ``character_dict`` (UniMERNet tokenizer) is filled in by the converter
    in :meth:`WeightConverter._save_inference_yml` — ``tokenizer.json`` is
    resolved from the input dir or the official_models cache, and
    ``tokenizer_config.json`` is hardcoded as ``UNIMERNET_TOKENIZER_CONFIG``.

    Mirrors the inference.yml published with the official safetensors repos
    (PaddlePaddle/PP-FormulaNet-L_safetensors etc.). Note the static-graph
    input is 1-channel: ``LatexImageFormat`` selects a single channel from
    the BGR/RGB image; the model expands 1->3 internally.
    """
    return {
        "Hpi": _hpi_simple("x", [[1, 1, 768, 768], [1, 1, 768, 768], [8, 1, 768, 768]]),
        "PreProcess": {
            "transform_ops": [
                {"UniMERNetImgDecode": {"input_size": [768, 768]}},
                {"UniMERNetTestTransform": None},
                {"LatexImageFormat": None},
                {
                    "UniMERNetLabelEncode": {
                        # Matches the published artifact for both variants;
                        # this op is registered as a no-op in the predictor,
                        # so the exact value isn't load-bearing at runtime.
                        "max_seq_len": 2560,
                        "rec_char_dict_path": "ppocr/utils/dict/unimernet_tokenizer",
                    }
                },
                {
                    "KeepKeys": {
                        "keep_keys": ["image", "label", "attention_mask", "filename"]
                    }
                },
            ]
        },
        "PostProcess": {
            "name": "UniMERNetDecode",
            # ``character_dict`` is injected by WeightConverter._save_inference_yml.
        },
    }


_INFERENCE_META_REGISTRY = {
    "PP-LCNet_x1_0_doc_ori": _meta_cls_doc_ori,
    "PP-LCNet_x1_0_table_cls": _meta_cls_table,
    "PP-LCNet_x0_25_textline_ori": _meta_cls_textline,
    "PP-LCNet_x1_0_textline_ori": _meta_cls_textline,
    "PP-OCRv5_mobile_det": _meta_det,
    "PP-OCRv5_server_det": _meta_det,
    # PP-OCRv6 det shares the v5 DB postprocess + Hpi shapes; only
    # preprocessor_config.json differs (see _PPOCRV6_DET_PREPROC).
    "PP-OCRv6_small_det": _meta_det,
    "PP-OCRv6_tiny_det": _meta_det,
    "PP-OCRv6_medium_det": _meta_det,
    "PP-OCRv5_mobile_rec": _meta_rec,
    "PP-OCRv5_server_rec": _meta_rec,
    # v6 rec inference.yml matches v5 rec exactly (same Hpi shapes, same
    # CTC transform_ops); only the per-model character_dict differs and is
    # injected by WeightConverter._save_inference_yml.
    "PP-OCRv6_small_rec": _meta_rec,
    "PP-OCRv6_medium_rec": _meta_rec,
    "PP-OCRv6_tiny_rec": _meta_rec,
    "SLANet": lambda: _meta_slanet("SLANet"),
    "SLANet_plus": lambda: _meta_slanet("SLANet_plus"),
    "SLANeXt_wired": lambda: _meta_slanext("SLANeXt_wired"),
    "SLANeXt_wireless": lambda: _meta_slanext("SLANeXt_wireless"),
    "PP-DocLayoutV2": _meta_doclayoutv3,
    "PP-DocLayoutV3": _meta_doclayoutv3,
    "RT-DETR-L_wired_table_cell_det": lambda: _meta_det_rtdetr(
        _LABEL_TABLE_CELL_DET, _RTDETR_PREPROCESS_640, 640
    ),
    "RT-DETR-L_wireless_table_cell_det": lambda: _meta_det_rtdetr(
        _LABEL_TABLE_CELL_DET, _RTDETR_PREPROCESS_640, 640
    ),
    "PP-DocLayout_plus-L": lambda: _meta_det_rtdetr(
        _LABEL_DOC_LAYOUT_PLUS, _RTDETR_PREPROCESS_800, 800
    ),
    "PP-DocBlockLayout": lambda: _meta_det_rtdetr(
        _LABEL_DOC_BLOCK_LAYOUT, _RTDETR_PREPROCESS_640, 640
    ),
    "UVDoc": _meta_uvdoc,
    "PP-FormulaNet-L": lambda: _meta_pp_formulanet("PP-FormulaNet-L"),
    "PP-FormulaNet_plus-L": lambda: _meta_pp_formulanet("PP-FormulaNet_plus-L"),
    "PP-Chart2Table": _meta_chart2table,
}


# Preprocessor configs (preprocessor_config.json)
_VALID_PROCESSOR_KEYS = [
    "images",
    "do_resize",
    "size",
    "resample",
    "do_rescale",
    "rescale_factor",
    "do_normalize",
    "image_mean",
    "image_std",
    "return_tensors",
    "data_format",
    "input_data_format",
]

_PPLCNET_PREPROC_BASE = {
    "_valid_processor_keys": _VALID_PROCESSOR_KEYS,
    "do_normalize": True,
    "do_rescale": True,
    "do_resize": True,
    "image_processor_type": "PPLCNetImageProcessor",
    "model_input_names": ["pixel_values", "original_image_size"],
    "image_mode": "BGR",
    "channel_first": False,
    "image_mean": [0.406, 0.456, 0.485],
    "image_std": [0.225, 0.224, 0.229],
    "rescale_factor": 1.0 / 255,
    "resample": 2,
    "keep_keys": ["image", "shape", "polys", "ignore_tags"],
}

_PPLCNET_CROP_PREPROC = {
    **_PPLCNET_PREPROC_BASE,
    "do_center_crop": True,
    "crop_size": 224,
    "resize_short": 256,
}

_PPLCNET_TEXTLINE_PREPROC = {
    **_PPLCNET_PREPROC_BASE,
    "do_center_crop": False,
    "size": {"width": 160, "height": 80},
    "resize_short": None,
}

_DET_PREPROC_BASE = {
    "_valid_processor_keys": _VALID_PROCESSOR_KEYS,
    "do_normalize": True,
    "do_rescale": True,
    "do_resize": True,
    "model_input_names": ["pixel_values", "original_image_size"],
    "image_mode": "BGR",
    "channel_first": False,
    "limit_side_len": 960,
    "max_side_limit": 4000,
    "image_mean": [0.406, 0.456, 0.485],
    "image_std": [0.225, 0.224, 0.229],
    "rescale_factor": 1.0 / 255,
    "resample": 2,
    "keep_keys": ["image", "shape", "polys", "ignore_tags"],
}

_MOBILE_DET_PREPROC = {
    **_DET_PREPROC_BASE,
    "model_type": "pp_ocrv5_mobile_det",
}

_SERVER_DET_PREPROC = {
    **_DET_PREPROC_BASE,
    "image_processor_type": "PPOCRV5ServerDetImageProcessor",
    "limit_type": "max",
    "normalize_order": "hwc",
    "do_to_chw": True,
}

# PP-OCRv6 det: same image processor class as v5 server, but the released
# checkpoints differ in side-length policy (``limit_side_len=736`` /
# ``limit_type=min``) and add ``do_convert_rgb``.
_PPOCRV6_DET_PREPROC = {
    **_DET_PREPROC_BASE,
    "do_convert_rgb": True,
    "image_processor_type": "PPOCRV5ServerDetImageProcessor",
    "limit_side_len": 736,
    "limit_type": "min",
    "normalize_order": "hwc",
    "do_to_chw": True,
}

_REC_PREPROC_BASE = {
    "size": {"height": 48, "width": 320},
    "pad_size": {"height": 48, "width": 320},
    "do_resize": True,
    "do_rescale": True,
    "do_convert_rgb": True,
    "do_normalize": True,
    "do_pad": True,
    "max_image_width": 3200,
}

_MOBILE_REC_PREPROC = {
    "model_type": "pp_ocrv5_mobile_rec",
    **_REC_PREPROC_BASE,
}

_SERVER_REC_PREPROC = {
    **_REC_PREPROC_BASE,
}


def _rtdetr_preproc(image_processor_type, height, width):
    return {
        "_valid_processor_keys": _VALID_PROCESSOR_KEYS,
        "do_normalize": True,
        "do_rescale": True,
        "do_resize": True,
        "image_processor_type": image_processor_type,
        "image_mean": [0, 0, 0],
        "image_std": [1, 1, 1],
        "rescale_factor": 1.0 / 255,
        "resample": 3,
        "size": {"height": height, "width": width},
    }


_UVDOC_PREPROC = {
    "data_format": "channels_first",
    "do_rescale": True,
    "do_resize": True,
    "image_processor_type": "UVDocImageProcessor",
    "resample": 2,
    "rescale_factor": 1.0 / 255,
    "size": {"height": 712, "width": 488},
}

PREPROCESSOR_CONFIGS = {
    "PP-LCNet_x1_0_doc_ori": _PPLCNET_CROP_PREPROC,
    "PP-LCNet_x1_0_table_cls": _PPLCNET_CROP_PREPROC,
    "PP-LCNet_x0_25_textline_ori": _PPLCNET_TEXTLINE_PREPROC,
    "PP-LCNet_x1_0_textline_ori": _PPLCNET_TEXTLINE_PREPROC,
    "PP-OCRv5_mobile_det": _MOBILE_DET_PREPROC,
    "PP-OCRv5_server_det": _SERVER_DET_PREPROC,
    "PP-OCRv6_small_det": _PPOCRV6_DET_PREPROC,
    "PP-OCRv6_tiny_det": _PPOCRV6_DET_PREPROC,
    "PP-OCRv6_medium_det": _PPOCRV6_DET_PREPROC,
    "PP-OCRv5_mobile_rec": _MOBILE_REC_PREPROC,
    "PP-OCRv5_server_rec": _SERVER_REC_PREPROC,
    # v6 rec preprocessor_config.json is exactly _REC_PREPROC_BASE + per-model
    # character_list (injected separately). No model_type field, no extra
    # tweaks like the v5 mobile/server split.
    "PP-OCRv6_small_rec": _REC_PREPROC_BASE,
    "PP-OCRv6_medium_rec": _REC_PREPROC_BASE,
    "PP-OCRv6_tiny_rec": _REC_PREPROC_BASE,
    "SLANet": {
        "image_processor_type": "SLANeXtImageProcessor",
        "do_resize": True,
        "size": {"height": 488, "width": 488},
        "pad_size": {"height": 488, "width": 488},
        "do_normalize": True,
        "image_mean": [0.485, 0.456, 0.406],
        "image_std": [0.229, 0.224, 0.225],
        "do_pad": True,
    },
    "SLANet_plus": {
        "image_processor_type": "SLANeXtImageProcessor",
        "do_resize": True,
        "size": {"height": 488, "width": 488},
        "pad_size": {"height": 488, "width": 488},
        "do_normalize": True,
        "image_mean": [0.485, 0.456, 0.406],
        "image_std": [0.229, 0.224, 0.225],
        "do_pad": True,
    },
    "SLANeXt_wired": {
        "do_resize": True,
        "size": {"height": 512, "width": 512},
        "do_normalize": True,
        "image_mean": [0.485, 0.456, 0.406],
        "image_std": [0.229, 0.224, 0.225],
        "do_pad": True,
    },
    "SLANeXt_wireless": {
        "do_resize": True,
        "size": {"height": 512, "width": 512},
        "do_normalize": True,
        "image_mean": [0.485, 0.456, 0.406],
        "image_std": [0.229, 0.224, 0.225],
        "do_pad": True,
    },
    "PP-DocLayoutV3": _rtdetr_preproc("PPDocLayoutV3ImageProcessor", 800, 800),
    "RT-DETR-L_wired_table_cell_det": _rtdetr_preproc("RTDetrImageProcessor", 640, 640),
    "RT-DETR-L_wireless_table_cell_det": _rtdetr_preproc(
        "RTDetrImageProcessor", 640, 640
    ),
    "PP-DocLayout_plus-L": _rtdetr_preproc("RTDetrImageProcessor", 800, 800),
    "PP-DocBlockLayout": _rtdetr_preproc("RTDetrImageProcessor", 640, 640),
    "UVDoc": _UVDOC_PREPROC,
    # PP-FormulaNet uses processor_config.json instead of preprocessor_config.json,
    # so it has no entry here — see UNIMERNET_PROCESSOR_CONFIG and
    # WeightConverter._save_pp_formulanet_assets.
    "PP-Chart2Table": {
        "_valid_processor_keys": _VALID_PROCESSOR_KEYS,
        "do_normalize": True,
        "do_rescale": True,
        "do_resize": True,
        "image_processor_type": "PPChart2TableImageProcessor",
        "model_input_names": ["pixel_values", "original_image_size"],
        "image_mode": "RGB",
        "channel_first": False,
        "image_mean": [0.48145466, 0.4578275, 0.40821073],
        "image_std": [0.26862954, 0.26130258, 0.27577711],
        "rescale_factor": 0.00392156862745098,
        "resample": 3,
        "normalize_order": "hwc",
        "do_to_chw": True,
        "size": {"height": 1024, "width": 1024},
        "keep_keys": ["image", "shape", "polys", "ignore_tags"],
    },
}


def build_inference_meta(model_name):
    """Build inference.yml metadata for a model."""
    if model_name not in _INFERENCE_META_REGISTRY:
        raise ValueError(f"No inference metadata defined for {model_name}")
    return _INFERENCE_META_REGISTRY[model_name]()


# Character dict loading
_RES_DIR = Path(__file__).resolve().parent.parent.parent / "res"
_BUNDLED_DICT_PATH = _RES_DIR / "ppocrv5_dict.txt"

# Per-model bundled character dict. v6 small/medium share one dict, tiny
# has its own; everything not in here defaults to the v5 dict.
_REC_DICT_PATHS = {
    "PP-OCRv5_mobile_rec": _BUNDLED_DICT_PATH,
    "PP-OCRv5_server_rec": _BUNDLED_DICT_PATH,
    "PP-OCRv6_small_rec": _RES_DIR / "ppocrv6_rec_dict.txt",
    "PP-OCRv6_medium_rec": _RES_DIR / "ppocrv6_rec_dict.txt",
    "PP-OCRv6_tiny_rec": _RES_DIR / "ppocrv6_tiny_rec_dict.txt",
}


def load_character_dict(model_name=None):
    """Load the character dict bundled for ``model_name``.

    Returns a list of character strings (no leading ``"blank"`` and no
    trailing space). ``model_name=None`` keeps the legacy v5 behavior.
    """
    path = _REC_DICT_PATHS.get(model_name, _BUNDLED_DICT_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"Bundled character dict not found at {path}. "
            "This file is required for rec model conversion."
        )
    chars = path.read_text("utf-8").rstrip("\n").split("\n")
    logging.info(f"Loaded character dict ({len(chars)} chars) for {model_name!r}")
    return chars


# PP-Chart2Table tokenizer assets
def _build_chart2table_added_tokens():
    """Build the expanded added_tokens.json for PP-Chart2Table."""
    tokens = {}
    tokens["<|endoftext|>"] = 151643
    tokens["<|im_start|>"] = 151644
    tokens["<|im_end|>"] = 151645
    for i in range(205):
        tokens[f"<|extra_{i}|>"] = 151646 + i
    tokens["<ref>"] = 151851
    tokens["</ref>"] = 151852
    tokens["<box>"] = 151853
    tokens["</box>"] = 151854
    tokens["<quad>"] = 151855
    tokens["</quad>"] = 151856
    tokens["<img>"] = 151857
    tokens["</img>"] = 151858
    tokens["<imgpad>"] = 151859
    return tokens


CHART2TABLE_ADDED_TOKENS = _build_chart2table_added_tokens()

CHART2TABLE_GENERATION_CONFIG = {
    "eos_token_id": 151645,
    "pad_token_id": 151643,
    "max_new_tokens": 2048,
}

CHART2TABLE_SPECIAL_TOKENS_MAP = {
    "additional_special_tokens": [
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<ref>",
        "</ref>",
        "<box>",
        "</box>",
        "<quad>",
        "</quad>",
        "<img>",
        "</img>",
        "<imgpad>",
    ],
    "eos_token": {
        "content": "<|im_end|>",
        "lstrip": False,
        "normalized": False,
        "rstrip": False,
        "single_word": False,
    },
    "pad_token": {
        "content": "<|endoftext|>",
        "lstrip": False,
        "normalized": False,
        "rstrip": False,
        "single_word": False,
    },
}

CHART2TABLE_TOKENIZER_CONFIG = {
    "add_prefix_space": None,
    "backend": "tokenizers",
    "bos_token": None,
    "eos_token": "<|endoftext|>",
    "model_max_length": 1000000000000000019884624838656,
    "pad_token": "<|endoftext|>",
    "tokenizer_class": "Qwen2Tokenizer",
    "unk_token": "<|endoftext|>",
}


# PP-FormulaNet tokenizer assets
def _unimernet_special_token(content):
    return {
        "content": content,
        "lstrip": False,
        "normalized": False,
        "rstrip": False,
        "single_word": False,
        "special": True,
    }


# Mirrors tokenizer_config.json published with
# PaddlePaddle/PP-FormulaNet-L_safetensors and PP-FormulaNet_plus-L_safetensors.
# tokenizer.json itself is too large to hardcode (~2.1 MB) — it is resolved at
# conversion time via the input dir or the official_models cache.
UNIMERNET_TOKENIZER_CONFIG = {
    "added_tokens_decoder": {
        str(i): _unimernet_special_token(content)
        for i, content in enumerate(
            [
                "<s>",
                "<pad>",
                "</s>",
                "<unk>",
                "[START_REF]",
                "[END_REF]",
                "[IMAGE]",
                "<fragments>",
                "</fragments>",
                "<work>",
                "</work>",
                "[START_SUP]",
                "[END_SUP]",
                "[START_SUB]",
                "[END_SUB]",
                "[START_DNA]",
                "[END_DNA]",
                "[START_AMINO]",
                "[END_AMINO]",
                "[START_SMILES]",
                "[END_SMILES]",
                "[START_I_SMILES]",
                "[END_I_SMILES]",
            ]
        )
    },
    "additional_special_tokens": [],
    "bos_token": "<s>",
    "clean_up_tokenization_spaces": False,
    "eos_token": "</s>",
    "max_length": 4096,
    "model_max_length": 768,
    "pad_to_multiple_of": None,
    "pad_token": "<pad>",
    "pad_token_type_id": 0,
    "padding_side": "right",
    # Standalone tokenizer_config.json on the published HF repo uses
    # "NougatProcessor"; the embedded copy in inference.yml uses
    # "VariableDonutProcessor". WeightConverter._save_inference_yml swaps it
    # at injection time. Either value is functionally equivalent at runtime
    # since UniMERNetDecode only reads added_tokens_decoder.
    "processor_class": "NougatProcessor",
    "stride": 0,
    "tokenizer_class": "NougatTokenizer",
    "truncation_side": "right",
    "truncation_strategy": "longest_first",
    "unk_token": "<unk>",
    "vocab_file": None,
}


# Mirrors processor_config.json published with PP-FormulaNet HF safetensors repos.
UNIMERNET_PROCESSOR_CONFIG = {
    "image_processor": {
        "do_align_long_axis": False,
        "do_crop_margin": True,
        "do_normalize": True,
        "do_pad": True,
        "do_rescale": True,
        "do_resize": True,
        "do_thumbnail": True,
        "size": {"height": 768, "width": 768},
        "image_mean": [0.7931, 0.7931, 0.7931],
        "image_std": [0.1738, 0.1738, 0.1738],
        "image_processor_type": "PPFormulaNetImageProcessor",
        "resample": 2,
        "return_tensors": "pt",
    },
    "processor_class": "PPFormulaNetProcessor",
}


# Mirrors generation_config.json published with PP-FormulaNet HF safetensors repos.
UNIMERNET_GENERATION_CONFIG = {
    "bos_token_id": 0,
    "eos_token_id": 2,
    "forced_eos_token_id": 2,
    "decoder_start_token_id": 2,
    "max_length": 1537,
    "pad_token_id": 1,
    "use_cache": True,
}
