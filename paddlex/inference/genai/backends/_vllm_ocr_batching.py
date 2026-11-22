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

"""Select a batching adapter only for the per-image vLLM input contract."""

import json
from pathlib import Path
from typing import Annotated, get_args, get_origin, get_type_hints

from ....utils import logging

OCR_BATCHED_ARCH = "PaddleXVisionBatchedPaddleOCRVLForConditionalGeneration"
OCR_BATCHED_IMPL = (
    "paddlex.inference.genai.models.paddleocr_vl_09b._vllm_batching:" + OCR_BATCHED_ARCH
)
_NATIVE_ARCH = "PaddleOCRVLForConditionalGeneration"


def _uses_per_image_inputs(schema):
    """Recognize ragged [images, patches, C, H, W], not native packed [patches, C, H, W]."""
    try:
        pixel_annotation = get_type_hints(schema, include_extras=True)["pixel_values"]
    except (KeyError, NameError, TypeError):
        return False
    if get_origin(pixel_annotation) is not Annotated:
        return False
    return any(
        tuple(getattr(shape, "dims", ())) == ("bn", "p", 3, "patch_size", "patch_size")
        and "p" in getattr(shape, "dynamic_dims", ())
        for shape in get_args(pixel_annotation)[1:]
    )


def enable_default_vision_batching(config, model_dir):
    """Keep custom architectures and already-batched vLLM models untouched."""
    raw = config.get("hf-overrides", config.get("hf_overrides", "{}"))
    try:
        overrides = json.loads(raw)
    except (TypeError, ValueError):
        return
    if not isinstance(overrides, dict) or "architectures" in overrides:
        return
    try:
        architectures = json.loads(
            (Path(model_dir) / "config.json").read_text(encoding="utf-8")
        )["architectures"]
    except (OSError, ValueError, TypeError, KeyError):
        return
    if architectures != [_NATIVE_ARCH]:
        return

    try:
        from vllm.model_executor.models import paddleocr_vl
    except ImportError:
        logging.debug(
            "PaddleOCR-VL vision batching: could not load the vLLM image schema."
        )
        return
    schema = getattr(paddleocr_vl, "PaddleOCRImagePixelInputs", None)
    if not _uses_per_image_inputs(schema) or not all(
        hasattr(paddleocr_vl, symbol)
        for symbol in (
            "PaddleOCRVLForConditionalGeneration",
            "PaddleOCRVLMultiModalProcessor",
            "PaddleOCRVLProcessingInfo",
            "PaddleOCRVLDummyInputsBuilder",
        )
    ):
        return

    overrides["architectures"] = [OCR_BATCHED_ARCH]
    config.pop("hf_overrides", None)
    config["hf-overrides"] = json.dumps(overrides)
