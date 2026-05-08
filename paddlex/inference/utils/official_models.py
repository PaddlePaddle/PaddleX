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

import hashlib
import os
import shutil
import tempfile
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Sequence, Set, Tuple

import filelock
import huggingface_hub as hf_hub
import huggingface_hub.utils as hf_hub_utils

hf_hub.logging.set_verbosity_error()

import requests

os.environ["AISTUDIO_LOG"] = "critical"
import modelscope
from aistudio_sdk.errors import NotExistError
from aistudio_sdk.snapshot_download import snapshot_download as aistudio_download

from ...utils import logging

ms_hub_errors = None
try:
    import modelscope.hub.errors as _ms_hub_errors

    ms_hub_errors = _ms_hub_errors
except Exception as e:
    logging.debug(
        "Failed to import `modelscope.hub.errors` (%r). ModelScope downloads can still "
        "be used; not-found detection will use generic fallbacks.",
        e,
    )

from ...utils.cache import CACHE_DIR, FILE_LOCK_DIR
from ...utils.download import download_and_extract
from ...utils.flags import (
    DISABLE_MODEL_SOURCE_CHECK,
    HUGGING_FACE_ENDPOINT,
    MODEL_SOURCE,
)
from ..models.utils.model_paths import LocalModelFormat

ALL_MODELS = [
    "ResNet18",
    "ResNet18_vd",
    "ResNet34",
    "ResNet34_vd",
    "ResNet50",
    "ResNet50_vd",
    "ResNet101",
    "ResNet101_vd",
    "ResNet152",
    "ResNet152_vd",
    "ResNet200_vd",
    "PaddleOCR-VL",
    "PaddleOCR-VL-1.5",
    "PP-LCNet_x0_25",
    "PP-LCNet_x0_25_textline_ori",
    "PP-LCNet_x0_35",
    "PP-LCNet_x0_5",
    "PP-LCNet_x0_75",
    "PP-LCNet_x1_0",
    "PP-LCNet_x1_0_doc_ori",
    "PP-LCNet_x1_0_textline_ori",
    "PP-LCNet_x1_5",
    "PP-LCNet_x2_5",
    "PP-LCNet_x2_0",
    "PP-LCNetV2_small",
    "PP-LCNetV2_base",
    "PP-LCNetV2_large",
    "MobileNetV3_large_x0_35",
    "MobileNetV3_large_x0_5",
    "MobileNetV3_large_x0_75",
    "MobileNetV3_large_x1_0",
    "MobileNetV3_large_x1_25",
    "MobileNetV3_small_x0_35",
    "MobileNetV3_small_x0_5",
    "MobileNetV3_small_x0_75",
    "MobileNetV3_small_x1_0",
    "MobileNetV3_small_x1_25",
    "ConvNeXt_tiny",
    "ConvNeXt_small",
    "ConvNeXt_base_224",
    "ConvNeXt_base_384",
    "ConvNeXt_large_224",
    "ConvNeXt_large_384",
    "MobileNetV2_x0_25",
    "MobileNetV2_x0_5",
    "MobileNetV2_x1_0",
    "MobileNetV2_x1_5",
    "MobileNetV2_x2_0",
    "MobileNetV1_x0_25",
    "MobileNetV1_x0_5",
    "MobileNetV1_x0_75",
    "MobileNetV1_x1_0",
    "SwinTransformer_tiny_patch4_window7_224",
    "SwinTransformer_small_patch4_window7_224",
    "SwinTransformer_base_patch4_window7_224",
    "SwinTransformer_base_patch4_window12_384",
    "SwinTransformer_large_patch4_window7_224",
    "SwinTransformer_large_patch4_window12_384",
    "PP-HGNet_tiny",
    "PP-HGNet_small",
    "PP-HGNet_base",
    "PP-HGNetV2-B0",
    "PP-HGNetV2-B1",
    "PP-HGNetV2-B2",
    "PP-HGNetV2-B3",
    "PP-HGNetV2-B4",
    "PP-HGNetV2-B5",
    "PP-HGNetV2-B6",
    "FasterNet-L",
    "FasterNet-M",
    "FasterNet-S",
    "FasterNet-T0",
    "FasterNet-T1",
    "FasterNet-T2",
    "StarNet-S1",
    "StarNet-S2",
    "StarNet-S3",
    "StarNet-S4",
    "MobileNetV4_conv_small",
    "MobileNetV4_conv_medium",
    "MobileNetV4_conv_large",
    "MobileNetV4_hybrid_medium",
    "MobileNetV4_hybrid_large",
    "CLIP_vit_base_patch16_224",
    "CLIP_vit_large_patch14_224",
    "PP-LCNet_x1_0_ML",
    "PP-HGNetV2-B0_ML",
    "PP-HGNetV2-B4_ML",
    "PP-HGNetV2-B6_ML",
    "ResNet50_ML",
    "CLIP_vit_base_patch16_448_ML",
    "PP-YOLOE_plus-X",
    "PP-YOLOE_plus-L",
    "PP-YOLOE_plus-M",
    "PP-YOLOE_plus-S",
    "RT-DETR-L",
    "RT-DETR-H",
    "RT-DETR-X",
    "YOLOv3-DarkNet53",
    "YOLOv3-MobileNetV3",
    "YOLOv3-ResNet50_vd_DCN",
    "YOLOX-L",
    "YOLOX-M",
    "YOLOX-N",
    "YOLOX-S",
    "YOLOX-T",
    "YOLOX-X",
    "RT-DETR-R18",
    "RT-DETR-R50",
    "PicoDet-S",
    "PicoDet-L",
    "Deeplabv3-R50",
    "Deeplabv3-R101",
    "Deeplabv3_Plus-R50",
    "Deeplabv3_Plus-R101",
    "PP-ShiTuV2_rec",
    "PP-ShiTuV2_rec_CLIP_vit_base",
    "PP-ShiTuV2_rec_CLIP_vit_large",
    "PP-LiteSeg-T",
    "PP-LiteSeg-B",
    "OCRNet_HRNet-W48",
    "OCRNet_HRNet-W18",
    "SegFormer-B0",
    "SegFormer-B1",
    "SegFormer-B2",
    "SegFormer-B3",
    "SegFormer-B4",
    "SegFormer-B5",
    "SeaFormer_tiny",
    "SeaFormer_small",
    "SeaFormer_base",
    "SeaFormer_large",
    "Mask-RT-DETR-H",
    "Mask-RT-DETR-L",
    "PP-OCRv4_server_rec",
    "Mask-RT-DETR-S",
    "Mask-RT-DETR-M",
    "Mask-RT-DETR-X",
    "SOLOv2",
    "MaskRCNN-ResNet50",
    "MaskRCNN-ResNet50-FPN",
    "MaskRCNN-ResNet50-vd-FPN",
    "MaskRCNN-ResNet101-FPN",
    "MaskRCNN-ResNet101-vd-FPN",
    "MaskRCNN-ResNeXt101-vd-FPN",
    "Cascade-MaskRCNN-ResNet50-FPN",
    "Cascade-MaskRCNN-ResNet50-vd-SSLDv2-FPN",
    "PP-YOLOE_seg-S",
    "PP-OCRv3_mobile_rec",
    "en_PP-OCRv3_mobile_rec",
    "korean_PP-OCRv3_mobile_rec",
    "japan_PP-OCRv3_mobile_rec",
    "chinese_cht_PP-OCRv3_mobile_rec",
    "te_PP-OCRv3_mobile_rec",
    "ka_PP-OCRv3_mobile_rec",
    "ta_PP-OCRv3_mobile_rec",
    "latin_PP-OCRv3_mobile_rec",
    "arabic_PP-OCRv3_mobile_rec",
    "cyrillic_PP-OCRv3_mobile_rec",
    "devanagari_PP-OCRv3_mobile_rec",
    "en_PP-OCRv4_mobile_rec",
    "PP-OCRv4_server_rec_doc",
    "PP-OCRv4_mobile_rec",
    "PP-OCRv4_server_det",
    "PP-OCRv4_mobile_det",
    "PP-OCRv3_server_det",
    "PP-OCRv3_mobile_det",
    "PP-OCRv4_server_seal_det",
    "PP-OCRv4_mobile_seal_det",
    "ch_RepSVTR_rec",
    "ch_SVTRv2_rec",
    "PP-LCNet_x1_0_pedestrian_attribute",
    "PP-LCNet_x1_0_vehicle_attribute",
    "PicoDet_layout_1x",
    "PicoDet_layout_1x_table",
    "SLANet",
    "SLANet_plus",
    "LaTeX_OCR_rec",
    "UniMERNet",
    "PP-FormulaNet-S",
    "PP-FormulaNet-L",
    "PP-FormulaNet_plus-S",
    "PP-FormulaNet_plus-M",
    "PP-FormulaNet_plus-L",
    "FasterRCNN-ResNet34-FPN",
    "FasterRCNN-ResNet50",
    "FasterRCNN-ResNet50-FPN",
    "FasterRCNN-ResNet50-vd-FPN",
    "FasterRCNN-ResNet50-vd-SSLDv2-FPN",
    "FasterRCNN-ResNet101",
    "FasterRCNN-ResNet101-FPN",
    "FasterRCNN-ResNeXt101-vd-FPN",
    "FasterRCNN-Swin-Tiny-FPN",
    "Cascade-FasterRCNN-ResNet50-FPN",
    "Cascade-FasterRCNN-ResNet50-vd-SSLDv2-FPN",
    "UVDoc",
    "DLinear",
    "NLinear",
    "RLinear",
    "Nonstationary",
    "TimesNet",
    "TiDE",
    "PatchTST",
    "DLinear_ad",
    "AutoEncoder_ad",
    "Nonstationary_ad",
    "PatchTST_ad",
    "TimesNet_ad",
    "TimesNet_cls",
    "STFPM",
    "FCOS-ResNet50",
    "DETR-R50",
    "PP-YOLOE-L_vehicle",
    "PP-YOLOE-S_vehicle",
    "PP-ShiTuV2_det",
    "PP-YOLOE-S_human",
    "PP-YOLOE-L_human",
    "PicoDet-M",
    "PicoDet-XS",
    "PP-YOLOE_plus_SOD-L",
    "PP-YOLOE_plus_SOD-S",
    "PP-YOLOE_plus_SOD-largesize-L",
    "CenterNet-DLA-34",
    "CenterNet-ResNet50",
    "PicoDet-S_layout_3cls",
    "PicoDet-S_layout_17cls",
    "PicoDet-L_layout_3cls",
    "PicoDet-L_layout_17cls",
    "RT-DETR-H_layout_3cls",
    "RT-DETR-H_layout_17cls",
    "PicoDet_LCNet_x2_5_face",
    "BlazeFace",
    "BlazeFace-FPN-SSH",
    "PP-YOLOE_plus-S_face",
    "MobileFaceNet",
    "ResNet50_face",
    "PP-YOLOE-R-L",
    "Co-Deformable-DETR-R50",
    "Co-Deformable-DETR-Swin-T",
    "Co-DINO-R50",
    "Co-DINO-Swin-L",
    "whisper_large",
    "whisper_base",
    "whisper_medium",
    "whisper_small",
    "whisper_tiny",
    "PP-TSM-R50_8frames_uniform",
    "PP-TSMv2-LCNetV2_8frames_uniform",
    "PP-TSMv2-LCNetV2_16frames_uniform",
    "MaskFormer_tiny",
    "MaskFormer_small",
    "PP-LCNet_x1_0_table_cls",
    "SLANeXt_wired",
    "SLANeXt_wireless",
    "RT-DETR-L_wired_table_cell_det",
    "RT-DETR-L_wireless_table_cell_det",
    "YOWO",
    "PP-TinyPose_128x96",
    "PP-TinyPose_256x192",
    "GroundingDINO-T",
    "SAM-H_box",
    "SAM-H_point",
    "PP-DocLayoutV2",
    "PP-DocLayoutV3",
    "PP-DocLayout-L",
    "PP-DocLayout-M",
    "PP-DocLayout-S",
    "PP-DocLayout_plus-L",
    "PP-DocBlockLayout",
    "BEVFusion",
    "YOLO-Worldv2-L",
    "PP-DocBee-2B",
    "PP-DocBee-7B",
    "PP-Chart2Table",
    "PP-OCRv5_server_det",
    "PP-OCRv5_mobile_det",
    "PP-OCRv5_server_rec",
    "PP-OCRv5_mobile_rec",
    "eslav_PP-OCRv5_mobile_rec",
    "PP-DocBee2-3B",
    "latin_PP-OCRv5_mobile_rec",
    "korean_PP-OCRv5_mobile_rec",
    "th_PP-OCRv5_mobile_rec",
    "el_PP-OCRv5_mobile_rec",
    "en_PP-OCRv5_mobile_rec",
    "arabic_PP-OCRv5_mobile_rec",
    "te_PP-OCRv5_mobile_rec",
    "ta_PP-OCRv5_mobile_rec",
    "devanagari_PP-OCRv5_mobile_rec",
    "cyrillic_PP-OCRv5_mobile_rec",
    "G2PWModel",
    "fastspeech2_csmsc",
    "pwgan_csmsc",
]


OCR_MODELS = [
    "arabic_PP-OCRv3_mobile_rec",
    "chinese_cht_PP-OCRv3_mobile_rec",
    "ch_RepSVTR_rec",
    "ch_SVTRv2_rec",
    "cyrillic_PP-OCRv3_mobile_rec",
    "devanagari_PP-OCRv3_mobile_rec",
    "en_PP-OCRv3_mobile_rec",
    "en_PP-OCRv4_mobile_rec",
    "eslav_PP-OCRv5_mobile_rec",
    "japan_PP-OCRv3_mobile_rec",
    "ka_PP-OCRv3_mobile_rec",
    "korean_PP-OCRv3_mobile_rec",
    "korean_PP-OCRv5_mobile_rec",
    "LaTeX_OCR_rec",
    "latin_PP-OCRv3_mobile_rec",
    "latin_PP-OCRv5_mobile_rec",
    "en_PP-OCRv5_mobile_rec",
    "th_PP-OCRv5_mobile_rec",
    "el_PP-OCRv5_mobile_rec",
    "PaddleOCR-VL",
    "PaddleOCR-VL-1.5",
    "PicoDet_layout_1x",
    "PicoDet_layout_1x_table",
    "PicoDet-L_layout_17cls",
    "PicoDet-L_layout_3cls",
    "PicoDet-S_layout_17cls",
    "PicoDet-S_layout_3cls",
    "PP-DocBee2-3B",
    "PP-Chart2Table",
    "PP-DocBee-2B",
    "PP-DocBee-7B",
    "PP-DocBlockLayout",
    "PP-DocLayoutV2",
    "PP-DocLayoutV3",
    "PP-DocLayout-L",
    "PP-DocLayout-M",
    "PP-DocLayout_plus-L",
    "PP-DocLayout-S",
    "PP-FormulaNet-L",
    "PP-FormulaNet_plus-L",
    "PP-FormulaNet_plus-M",
    "PP-FormulaNet_plus-S",
    "PP-FormulaNet-S",
    "PP-LCNet_x0_25_textline_ori",
    "PP-LCNet_x1_0_doc_ori",
    "PP-LCNet_x1_0_table_cls",
    "PP-LCNet_x1_0_textline_ori",
    "PP-OCRv3_mobile_det",
    "PP-OCRv3_mobile_rec",
    "PP-OCRv3_server_det",
    "PP-OCRv4_mobile_det",
    "PP-OCRv4_mobile_rec",
    "PP-OCRv4_mobile_seal_det",
    "PP-OCRv4_server_det",
    "PP-OCRv4_server_rec_doc",
    "PP-OCRv4_server_rec",
    "PP-OCRv4_server_seal_det",
    "PP-OCRv5_mobile_det",
    "PP-OCRv5_mobile_rec",
    "PP-OCRv5_server_det",
    "PP-OCRv5_server_rec",
    "RT-DETR-H_layout_17cls",
    "RT-DETR-H_layout_3cls",
    "RT-DETR-L_wired_table_cell_det",
    "RT-DETR-L_wireless_table_cell_det",
    "SLANet",
    "SLANet_plus",
    "SLANeXt_wired",
    "SLANeXt_wireless",
    "ta_PP-OCRv3_mobile_rec",
    "te_PP-OCRv3_mobile_rec",
    "UniMERNet",
    "UVDoc",
    "arabic_PP-OCRv5_mobile_rec",
    "te_PP-OCRv5_mobile_rec",
    "ta_PP-OCRv5_mobile_rec",
    "devanagari_PP-OCRv5_mobile_rec",
    "cyrillic_PP-OCRv5_mobile_rec",
]

SAFETENSORS_SUPPORTED_MODELS_WITH_SUFFIX: Set[str] = {
    "PP-LCNet_x0_25_textline_ori",
    "PP-LCNet_x1_0_doc_ori",
    "PP-LCNet_x1_0_textline_ori",
    "PP-LCNet_x1_0_table_cls",
    "PP-DocLayoutV2",
    "PP-DocLayoutV3",
    "PP-DocLayout_plus-L",
    "PP-DocBlockLayout",
    "SLANeXt_wired",
    "SLANeXt_wireless",
    "RT-DETR-L_wired_table_cell_det",
    "RT-DETR-L_wireless_table_cell_det",
    "PP-OCRv5_server_det",
    "PP-OCRv5_mobile_det",
    "PP-OCRv5_server_rec",
    "PP-OCRv5_mobile_rec",
    "UVDoc",
    "PP-Chart2Table",
    "eslav_PP-OCRv5_mobile_rec",
    "korean_PP-OCRv5_mobile_rec",
    "latin_PP-OCRv5_mobile_rec",
    "en_PP-OCRv5_mobile_rec",
    "th_PP-OCRv5_mobile_rec",
    "el_PP-OCRv5_mobile_rec",
    "arabic_PP-OCRv5_mobile_rec",
    "te_PP-OCRv5_mobile_rec",
    "ta_PP-OCRv5_mobile_rec",
    "devanagari_PP-OCRv5_mobile_rec",
    "cyrillic_PP-OCRv5_mobile_rec",
}

SAFETENSORS_SUPPORTED_MODELS_WITHOUT_SUFFIX: Set[str] = {
    "PaddleOCR-VL-0.9B",
    "PaddleOCR-VL-1.5-0.9B",
}

SAFETENSORS_SUPPORTED_MODELS: Set[str] = (
    SAFETENSORS_SUPPORTED_MODELS_WITH_SUFFIX
    | SAFETENSORS_SUPPORTED_MODELS_WITHOUT_SUFFIX
)

PADDLE_DYN_SUPPORTED_MODELS: Set[str] = {
    "PP-DocBee-2B",
    "PP-DocBee-7B",
    "PP-DocBee2-3B",
    "whisper_large",
    "whisper_medium",
    "whisper_base",
    "whisper_small",
    "whisper_tiny",
}

ONNX_SUPPORTED_MODELS: Set[str] = {
    "PP-LCNet_x0_25_textline_ori",
    "PP-LCNet_x1_0_doc_ori",
    "PP-LCNet_x1_0_textline_ori",
    "PP-LCNet_x1_0_table_cls",
    "PP-DocLayout_plus-L",
    "PP-DocBlockLayout",
    "SLANeXt_wired",
    "SLANeXt_wireless",
    "RT-DETR-L_wired_table_cell_det",
    "RT-DETR-L_wireless_table_cell_det",
    "PP-OCRv5_server_det",
    "PP-OCRv5_mobile_det",
    "PP-OCRv5_server_rec",
    "PP-OCRv5_mobile_rec",
}


def _canonical_download_support_name(model_name: str) -> str:
    if model_name in {"PaddleOCR-VL", "PaddleOCR-VL-0.9B"}:
        return "PaddleOCR-VL-0.9B"
    return model_name


def _format_download_model_name(model_name: str, model_format: LocalModelFormat) -> str:
    if model_format in {"paddle", "paddle_dyn"}:
        return model_name
    if model_format == "safetensors":
        if model_name in SAFETENSORS_SUPPORTED_MODELS_WITH_SUFFIX:
            return f"{model_name}_safetensors"
        elif model_name in SAFETENSORS_SUPPORTED_MODELS_WITHOUT_SUFFIX:
            return model_name
        raise ValueError(f"Unknown safetensors model name: {model_name}")
    if model_format == "onnx":
        return f"{model_name}_onnx"
    raise ValueError(f"Unknown official model format: {model_format!r}.")


def _is_supported_official_model_format(
    model_name: str, model_format: LocalModelFormat
) -> bool:
    canonical_name = _canonical_download_support_name(model_name)
    if model_format == "paddle":
        return True
    if model_format == "paddle_dyn":
        return canonical_name in PADDLE_DYN_SUPPORTED_MODELS
    if model_format == "safetensors":
        return canonical_name in SAFETENSORS_SUPPORTED_MODELS
    if model_format == "onnx":
        return canonical_name in ONNX_SUPPORTED_MODELS
    if model_format == "om":
        return False
    raise ValueError(f"Unknown official model format: {model_format!r}.")


def _resolve_download_model_names(
    model_name: str,
    model_formats: Optional[Sequence[LocalModelFormat]] = None,
) -> Tuple[str, ...]:
    if model_formats is None:
        return (model_name,)
    formats = tuple(model_formats)
    model_names = []
    unsupported_formats = []
    for model_format in formats:
        if not _is_supported_official_model_format(model_name, model_format):
            unsupported_formats.append(model_format)
            continue
        download_model_name = _format_download_model_name(model_name, model_format)
        if download_model_name not in model_names:
            model_names.append(download_model_name)
    if not model_names:
        if len(formats) == 1:
            raise ValueError(
                f"Official model source does not provide a {formats[0]!r} package "
                f"for model {model_name!r}."
            )
        raise ValueError(
            f"Official model source does not provide any of the requested packages "
            f"{list(unsupported_formats)!r} for model {model_name!r}."
        )
    return tuple(model_names)


def _official_model_download_lock_path(model_names: Tuple[str, ...]) -> str:
    """Cross-process lock path for a resolved official model download key."""
    lock_dir = os.path.join(FILE_LOCK_DIR, "official_models")
    os.makedirs(lock_dir, exist_ok=True)
    key = hashlib.sha256("\0".join(model_names).encode("utf-8")).hexdigest()
    return os.path.join(lock_dir, f"{key}.lock")


def _iter_exception_chain(exc: Exception):
    current = exc
    visited = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        yield current
        current = getattr(current, "__cause__", None) or getattr(
            current, "__context__", None
        )


def _exception_http_status_code(exc_obj: BaseException) -> Optional[int]:
    # NOTE: Normally `requests.HTTPError` sets `.response`;
    # ModelScope `hub/api.py` sometimes does `raise HTTPError(r)` without `response=`,
    # so the `requests.Response` only appears in `args[0]`.
    response = getattr(exc_obj, "response", None)
    code = getattr(response, "status_code", None)
    if isinstance(code, int):
        return code
    for arg in getattr(exc_obj, "args", ()) or ():
        sc = getattr(arg, "status_code", None)
        if isinstance(sc, int):
            return sc
    return None


def _modelscope_is_model_package_not_found_error(exc: Exception) -> bool:
    """Detect ModelScope 'model not found' errors with or without `ms_hub_errors`."""
    if ms_hub_errors is not None:
        for current in _iter_exception_chain(exc):
            if isinstance(current, ms_hub_errors.NotExistError):
                return True
            if isinstance(current, ms_hub_errors.HTTPError):
                if _exception_http_status_code(current) == 404:
                    return True
        return False
    for current in _iter_exception_chain(exc):
        if isinstance(current, requests.HTTPError):
            if _exception_http_status_code(current) == 404:
                return True
        if current.__class__.__name__ == "NotExistError":
            return True
        # ModelScope hub HTTPError may not be a requests.HTTPError subclass.
        if (
            current.__class__.__name__ == "HTTPError"
            and _exception_http_status_code(current) == 404
        ):
            return True
    return False


class _BaseModelHoster(ABC):
    alias = ""
    model_list = []
    healthcheck_url = None
    _healthcheck_timeout = 1

    def __init__(self, save_dir):
        self._save_dir = save_dir

    @staticmethod
    def _strip_repo_suffix(model_name):
        for suffix in ("_safetensors", "_onnx"):
            if model_name.endswith(suffix):
                return model_name[: -len(suffix)]
        return model_name

    def supports_model(self, model_name):
        if model_name in self.model_list:
            return True
        return self._strip_repo_suffix(model_name) in self.model_list

    def get_model(self, model_name):
        assert self.supports_model(
            model_name
        ), f"The model {model_name} is not supported on hosting {self.__class__.__name__}!"

        model_dir = self._save_dir / f"{model_name}"
        logging.info(
            f"Using official model ({model_name}), the model files will be automatically downloaded and saved in `{model_dir}`."
        )
        self._download(model_name, model_dir)
        logging.debug(
            f"`{model_name}` model files has been download from model source: `{self.alias}`!"
        )

        return model_dir

    @abstractmethod
    def _download(self):
        raise NotImplementedError

    @abstractmethod
    def is_model_package_not_found_error(self, exc: Exception) -> bool:
        raise NotImplementedError

    @classmethod
    def is_available(cls):
        if cls.healthcheck_url is None:
            return True
        try:
            response = requests.head(
                cls.healthcheck_url, timeout=cls._healthcheck_timeout
            )
            return response.ok == True
        except Exception:
            logging.debug(f"The model hosting platform({cls.__name__}) is unreachable!")
            return False


class _BosModelHoster(_BaseModelHoster):
    model_list = ALL_MODELS
    alias = "bos"
    healthcheck_url = "https://paddle-model-ecology.bj.bcebos.com"

    version = "paddle3.0.0"
    base_url = (
        "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model"
    )
    special_model_fn = {
        "whisper_large": "whisper_large.tar",
        "whisper_base": "whisper_base.tar",
        "whisper_medium": "whisper_medium.tar",
        "whisper_small": "whisper_small.tar",
        "whisper_tiny": "whisper_tiny.tar",
    }

    def _download(self, model_name, save_dir):
        if model_name in self.special_model_fn:
            fn = self.special_model_fn[model_name]
        else:
            fn = f"{model_name}_infer.tar"
        url = f"{self.base_url}/{self.version}/{fn}"
        download_and_extract(url, save_dir.parent, model_name, overwrite=False)

    def is_model_package_not_found_error(self, exc: Exception) -> bool:
        for current in _iter_exception_chain(exc):
            if isinstance(current, requests.HTTPError):
                response = current.response
                if response is not None and response.status_code == 404:
                    return True
        return False


class _HuggingFaceModelHoster(_BaseModelHoster):
    model_list = OCR_MODELS
    alias = "huggingface"
    healthcheck_url = HUGGING_FACE_ENDPOINT

    def _download(self, model_name, save_dir):
        def _clone(local_dir):
            hf_hub.snapshot_download(
                repo_id=f"PaddlePaddle/{model_name}",
                local_dir=local_dir,
                endpoint=HUGGING_FACE_ENDPOINT,
            )

        if os.path.exists(save_dir):
            _clone(save_dir)
        else:
            with tempfile.TemporaryDirectory() as td:
                temp_dir = os.path.join(td, "temp_dir")
                _clone(temp_dir)
                shutil.move(temp_dir, save_dir)

    def is_model_package_not_found_error(self, exc: Exception) -> bool:
        for current in _iter_exception_chain(exc):
            if isinstance(
                current,
                (
                    hf_hub_utils.RepositoryNotFoundError,
                    hf_hub_utils.EntryNotFoundError,
                    hf_hub_utils.RevisionNotFoundError,
                ),
            ):
                return True
            if isinstance(current, hf_hub_utils.HfHubHTTPError):
                response = current.response
                if response is not None and response.status_code == 404:
                    return True
                if (
                    response is not None
                    and response.status_code == 401
                    and (
                        "Repository Not Found" in str(current)
                        or "Entry Not Found" in str(current)
                        or "Revision Not Found" in str(current)
                    )
                ):
                    return True
        return False


class _ModelScopeModelHoster(_BaseModelHoster):
    model_list = OCR_MODELS
    alias = "modelscope"
    healthcheck_url = "https://modelscope.cn"

    def _download(self, model_name, save_dir):
        def _clone(local_dir):
            modelscope.snapshot_download(
                repo_id=f"PaddlePaddle/{model_name}", local_dir=local_dir
            )

        if os.path.exists(save_dir):
            _clone(save_dir)
        else:
            with tempfile.TemporaryDirectory() as td:
                temp_dir = os.path.join(td, "temp_dir")
                _clone(temp_dir)
                shutil.move(temp_dir, save_dir)

    def is_model_package_not_found_error(self, exc: Exception) -> bool:
        return _modelscope_is_model_package_not_found_error(exc)


class _AIStudioModelHoster(_BaseModelHoster):
    model_list = OCR_MODELS
    alias = "aistudio"
    healthcheck_url = "https://aistudio.baidu.com"

    def _download(self, model_name, save_dir):
        def _clone(local_dir):
            if "PaddleOCR-VL" in model_name:
                aistudio_download(
                    repo_id=f"PaddlePaddle/{model_name}", local_dir=local_dir
                )
            else:
                aistudio_download(repo_id=f"PaddleX/{model_name}", local_dir=local_dir)

        if os.path.exists(save_dir):
            _clone(save_dir)
        else:
            with tempfile.TemporaryDirectory() as td:
                temp_dir = os.path.join(td, "temp_dir")
                _clone(temp_dir)
                shutil.move(temp_dir, save_dir)

    def is_model_package_not_found_error(self, exc: Exception) -> bool:
        for current in _iter_exception_chain(exc):
            if isinstance(current, NotExistError):
                return True
            if isinstance(current, requests.HTTPError):
                response = current.response
                if response is not None and response.status_code == 404:
                    return True
        return False


class _ModelManager:
    model_list = ALL_MODELS
    _save_dir = Path(CACHE_DIR) / "official_models"
    hoster_candidates = [
        _HuggingFaceModelHoster,
        _AIStudioModelHoster,
        _ModelScopeModelHoster,
        _BosModelHoster,
    ]

    def __init__(self) -> None:
        self._hosters = None
        self._hosters_lock = threading.Lock()

    def _build_hosters(self):

        if DISABLE_MODEL_SOURCE_CHECK:
            logging.warning(
                f"Connectivity check to the model hoster has been skipped because `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK` is enabled."
            )
            hosters = []
            for hoster_cls in self.hoster_candidates:
                if hoster_cls.alias == MODEL_SOURCE:
                    hosters.insert(0, hoster_cls(self._save_dir))
                else:
                    hosters.append(hoster_cls(self._save_dir))
            return hosters

        logging.warning(
            f"Checking connectivity to the model hosters, this may take a while. To bypass this check, set `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK` to `True`."
        )
        hosters = []
        for hoster_cls in self.hoster_candidates:
            if hoster_cls.alias == MODEL_SOURCE:
                if hoster_cls.is_available():
                    hosters.insert(0, hoster_cls(self._save_dir))
            else:
                if hoster_cls.is_available():
                    hosters.append(hoster_cls(self._save_dir))
        if len(hosters) == 0:
            logging.warning(
                f"No model hoster is available! Please check your network connection to one of the following model hoster: HuggingFace ({_HuggingFaceModelHoster.healthcheck_url}), ModelScope ({_ModelScopeModelHoster.healthcheck_url}), AIStudio ({_AIStudioModelHoster.healthcheck_url}), or BOS ({_BosModelHoster.healthcheck_url}). Otherwise, only local models can be used."
            )
        return hosters

    def _get_hosters(self):
        if self._hosters is None:
            with self._hosters_lock:
                if self._hosters is None:
                    self._hosters = self._build_hosters()
        return self._hosters

    def _get_model_local_path(self, model_name):
        model_names = (
            (model_name,) if isinstance(model_name, str) else tuple(model_name)
        )
        resolved_names = []
        for candidate_name in model_names:
            if "PaddleOCR-VL" in candidate_name:
                candidate_name = candidate_name.replace("-0.9B", "")
            resolved_names.append(candidate_name)

        model_dir = None
        for candidate_name in resolved_names:
            candidate_dir = self._save_dir / f"{candidate_name}"
            if os.path.exists(candidate_dir):
                logging.info(
                    f"Model files already exist. Using cached files. To redownload, please delete the directory manually: `{candidate_dir}`."
                )
                model_dir = candidate_dir
                break

        if model_dir is None:
            lock_path = _official_model_download_lock_path(tuple(resolved_names))
            with filelock.FileLock(lock_path):
                for candidate_name in resolved_names:
                    candidate_dir = self._save_dir / f"{candidate_name}"
                    if os.path.exists(candidate_dir):
                        logging.info(
                            f"Model files already exist. Using cached files. To redownload, please delete the directory manually: `{candidate_dir}`."
                        )
                        model_dir = candidate_dir
                        break

                if model_dir is None:
                    hosters = self._get_hosters()
                    if len(hosters) == 0:
                        msg = "No available model hosting platforms detected. Please check your network connection."
                        logging.error(msg)
                        raise Exception(msg)

                    model_dir = self._download_from_hoster(hosters, resolved_names)

        if resolved_names[0] == "PaddleOCR-VL":
            vl_model_dir = model_dir / "PaddleOCR-VL-0.9B"
            if vl_model_dir.exists() and vl_model_dir.is_dir():
                return vl_model_dir

        return model_dir

    def get_model_path(
        self,
        model_name: str,
        *,
        model_formats: Optional[Sequence[LocalModelFormat]] = None,
    ):
        download_model_names = _resolve_download_model_names(model_name, model_formats)
        return self._get_model_local_path(download_model_names)

    def _download_from_hoster(self, hosters, model_name):
        model_names = (
            (model_name,) if isinstance(model_name, str) else tuple(model_name)
        )
        last_exception = None
        for idx, hoster in enumerate(hosters):
            attempted_candidates = []
            all_attempted_candidates_not_found = True
            for candidate_idx, candidate_name in enumerate(model_names):
                if not hoster.supports_model(candidate_name):
                    continue
                attempted_candidates.append(candidate_name)
                try:
                    model_path = hoster.get_model(candidate_name)
                    return model_path
                except Exception as e:
                    last_exception = e
                    is_not_found = hoster.is_model_package_not_found_error(e)
                    if is_not_found:
                        has_fallback = candidate_idx + 1 < len(model_names)
                        if has_fallback:
                            logging.warning(
                                f"Model package `{candidate_name}` was not found on "
                                f"{hoster.alias}, trying fallback package "
                                f"`{model_names[candidate_idx + 1]}`."
                            )
                        continue
                    all_attempted_candidates_not_found = False
                    if idx + 1 >= len(hosters):
                        raise Exception(
                            f"Encounter exception when download model from {hoster.alias}. No model source is available! Please check network or use local model files!"
                        ) from e
                    logging.warning(
                        f"Encountering exception when download model `{candidate_name}` "
                        f"from {hoster.alias}: \n{e}, will try to download from other "
                        f"model sources: `{hosters[idx + 1].alias}`."
                    )
                    break

            if attempted_candidates and all_attempted_candidates_not_found:
                if idx + 1 >= len(hosters):
                    break
                logging.warning(
                    f"Model packages {attempted_candidates!r} were not found on "
                    f"{hoster.alias}, will try model source `{hosters[idx + 1].alias}`."
                )
                continue

            if attempted_candidates:
                continue

        raise Exception(
            f"No model source is available for model `{model_names[0]}`! Please check "
            f"model name and network, or use local model files!"
        ) from last_exception

    def __contains__(self, model_name):
        return model_name in self.model_list

    def __getitem__(self, model_name):
        return self.get_model_path(model_name)


official_models = _ModelManager()
