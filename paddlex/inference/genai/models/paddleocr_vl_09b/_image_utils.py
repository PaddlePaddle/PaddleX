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
"""Shared image preprocessing helpers for PaddleOCR-VL GenAI backends."""

import math
from collections.abc import Mapping

from PIL import Image


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 28 * 28 * 130,
    max_pixels: int = 28 * 28 * 1280,
):
    """Resize dimensions to the nearest valid grid under the pixel budget."""
    if height < factor:
        width = round((width * factor) / height)
        height = factor

    if width < factor:
        height = round((height * factor) / width)
        width = factor

    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )

    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def resize_image(image, min_pixels, max_pixels, factor) -> Image.Image:
    width, height = image.size
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    return image.resize((resized_width, resized_height))


def _resize_image_payload(payload, *, min_pixels, max_pixels, factor):
    if isinstance(payload, Image.Image):
        return resize_image(payload, min_pixels, max_pixels, factor)
    if isinstance(payload, list):
        return [
            _resize_image_payload(
                item,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                factor=factor,
            )
            for item in payload
        ]
    if isinstance(payload, tuple):
        return tuple(
            _resize_image_payload(
                item,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                factor=factor,
            )
            for item in payload
        )
    return payload


def prepare_hf_processor_mm_data(mm_data: Mapping[str, object], image_processor):
    prepared = dict(mm_data)
    if "image" not in prepared:
        return prepared

    factor = image_processor.patch_size * image_processor.merge_size
    prepared["image"] = _resize_image_payload(
        prepared["image"],
        min_pixels=image_processor.min_pixels,
        max_pixels=image_processor.max_pixels,
        factor=factor,
    )
    return prepared
