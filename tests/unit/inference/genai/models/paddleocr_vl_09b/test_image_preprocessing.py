#
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
#
"""Unit tests for PaddleOCR-VL image preprocessing helpers."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from PIL import Image


def _load_image_utils_module():
    current = Path(__file__).resolve()
    repo_root = next(parent for parent in current.parents if (parent / "pyproject.toml").exists())
    module_path = (
        repo_root
        / "paddlex"
        / "inference"
        / "genai"
        / "models"
        / "paddleocr_vl_09b"
        / "_image_utils.py"
    )
    spec = spec_from_file_location("paddleocr_vl_image_utils", module_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class DummyImageProcessor:
    min_pixels = 28 * 28 * 130
    max_pixels = 28 * 28 * 1280
    patch_size = 14
    merge_size = 2


def test_smart_resize_upscales_32_square_to_min_pixel_grid():
    image_utils = _load_image_utils_module()

    assert image_utils.smart_resize(
        height=32,
        width=32,
        factor=28,
        min_pixels=DummyImageProcessor.min_pixels,
        max_pixels=DummyImageProcessor.max_pixels,
    ) == (336, 336)


def test_prepare_hf_processor_mm_data_resizes_tiny_pil_images():
    image_utils = _load_image_utils_module()
    mm_data = {"image": [Image.new("RGB", (32, 32), "white")], "meta": "keep"}

    prepared = image_utils.prepare_hf_processor_mm_data(
        mm_data, DummyImageProcessor()
    )

    assert prepared["meta"] == "keep"
    assert prepared["image"][0].size == (336, 336)


def test_prepare_hf_processor_mm_data_does_not_mutate_original_payload():
    image_utils = _load_image_utils_module()
    original_image = Image.new("RGB", (32, 32), "white")
    mm_data = {"image": [original_image]}

    prepared = image_utils.prepare_hf_processor_mm_data(
        mm_data, DummyImageProcessor()
    )

    assert mm_data["image"][0].size == (32, 32)
    assert prepared["image"][0].size == (336, 336)
    assert prepared["image"][0] is not original_image


def test_prepare_hf_processor_mm_data_keeps_non_pil_payloads():
    image_utils = _load_image_utils_module()
    mm_data = {"image": ["already-encoded"], "meta": "keep"}

    prepared = image_utils.prepare_hf_processor_mm_data(
        mm_data, DummyImageProcessor()
    )

    assert prepared == mm_data
