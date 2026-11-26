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
VISION_ROPE_OPTION = "paddlex-vision-performance"
_ROPE_ARCH = "PaddleXVisionRoPEPaddleOCRVLForConditionalGeneration"
_BATCHED_ROPE_ARCH = "PaddleXVisionBatchedRoPEPaddleOCRVLForConditionalGeneration"
_ROPE_MODULE = "paddlex.inference.genai.models.paddleocr_vl_18_09b._vllm_vision_rope"
VISION_ROPE_MODELS = {
    arch: f"{_ROPE_MODULE}:{arch}" for arch in (_ROPE_ARCH, _BATCHED_ROPE_ARCH)
}


def _pixel_input_shapes(schema):
    try:
        pixel_annotation = get_type_hints(schema, include_extras=True)["pixel_values"]
    except (KeyError, NameError, TypeError):
        return ()
    if get_origin(pixel_annotation) is not Annotated:
        return ()
    return get_args(pixel_annotation)[1:]


def _uses_per_image_inputs(schema):
    """Recognize ragged [images, patches, C, H, W], not native packed [patches, C, H, W]."""
    return any(
        tuple(getattr(shape, "dims", ())) == ("bn", "p", 3, "patch_size", "patch_size")
        and "p" in getattr(shape, "dynamic_dims", ())
        for shape in _pixel_input_shapes(schema)
    )


def enable_default_vision_batching(config, model_dir, *, fused_rope=False):
    """Keep custom architectures and already-batched vLLM models untouched."""
    raw = config.get("hf-overrides", config.get("hf_overrides", "{}"))
    try:
        overrides = json.loads(raw)
    except (TypeError, ValueError):
        if fused_rope:
            raise ValueError("Vision RoPE requires a JSON hf-overrides object.")
        return
    if not isinstance(overrides, dict) or "architectures" in overrides:
        if fused_rope:
            raise ValueError("Vision RoPE cannot override an explicit architecture.")
        return
    try:
        architectures = json.loads(
            (Path(model_dir) / "config.json").read_text(encoding="utf-8")
        )["architectures"]
    except (OSError, ValueError, TypeError, KeyError):
        architectures = None
    if architectures != [_NATIVE_ARCH]:
        if fused_rope:
            raise ValueError("Vision RoPE requires the integrated PaddleOCR-VL model.")
        return

    try:
        from vllm.model_executor.models import paddleocr_vl
    except ImportError:
        if fused_rope:
            raise
        logging.debug(
            "PaddleOCR-VL vision batching: could not load the vLLM image schema."
        )
        return
    schema = getattr(paddleocr_vl, "PaddleOCRImagePixelInputs", None)
    if not all(
        hasattr(paddleocr_vl, symbol)
        for symbol in (
            "PaddleOCRVLForConditionalGeneration",
            "PaddleOCRVLMultiModalProcessor",
            "PaddleOCRVLProcessingInfo",
            "PaddleOCRVLDummyInputsBuilder",
        )
    ):
        if fused_rope:
            raise ValueError("Vision RoPE requires the integrated PaddleOCR-VL model.")
        return

    ragged = _uses_per_image_inputs(schema)
    if fused_rope:
        if not ragged and not any(
            tuple(getattr(shape, "dims", ()))[1:] == (3, "patch_size", "patch_size")
            for shape in _pixel_input_shapes(schema)
        ):
            raise ValueError("Vision RoPE requires a recognized vLLM image schema.")
        arch = _BATCHED_ROPE_ARCH if ragged else _ROPE_ARCH
    elif ragged:
        arch = OCR_BATCHED_ARCH
    else:
        return
    overrides["architectures"] = [arch]
    config.pop("hf_overrides", None)
    config["hf-overrides"] = json.dumps(overrides)


def configure_vllm_ocr_vision(config, model_name, model_dir, integrated):
    """Apply #7 batching independently of the opt-in RoPE adapter."""
    fused_rope = config.pop(VISION_ROPE_OPTION, False)
    if not isinstance(fused_rope, bool):
        raise ValueError(f"{VISION_ROPE_OPTION} must be a YAML boolean.")
    if fused_rope:
        from vllm import __version__

        if (
            not integrated
            or model_name != "PaddleOCR-VL-1.8-0.9B"
            or __version__.split("+", 1)[0] not in ("0.19.0", "0.33.0")
        ):
            raise ValueError(
                f"{VISION_ROPE_OPTION} requires PaddleOCR-VL 1.8 and "
                "vLLM 0.19.0 or 0.33.0."
            )
    if integrated:
        enable_default_vision_batching(config, model_dir, fused_rope=fused_rope)
