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

import math
from typing import Any, Optional, Tuple

from ...utils.flags import PDF_MIN_RENDER_SCALE, PDF_RENDER_SCALE

DEFAULT_MAX_IMAGE_PIXELS: int = 178_956_970

__all__ = [
    "DEFAULT_MAX_IMAGE_PIXELS",
    "PDFRenderSizeError",
    "estimate_pdf_render_pixels",
    "get_pdf_render_scale_within_pixel_limit",
    "render_pdf_page_to_numpy",
]


class PDFRenderSizeError(Exception):
    """Raised when a PDF page cannot fit the pixel budget at minimum scale."""

    def __init__(
        self,
        message: str,
        *,
        width: Optional[int] = None,
        height: Optional[int] = None,
        pixel_count: Optional[int] = None,
        max_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
        page_index: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.width = width
        self.height = height
        self.pixel_count = pixel_count
        self.max_pixels = max_pixels
        self.page_index = page_index


def estimate_pdf_render_pixels(
    page_size: Tuple[float, float], scale: float
) -> Tuple[int, int, int]:
    w_pdf, h_pdf = float(page_size[0]), float(page_size[1])
    w_px = int(math.ceil(w_pdf * scale))
    h_px = int(math.ceil(h_pdf * scale))
    return w_px, h_px, w_px * h_px


def get_pdf_render_scale_within_pixel_limit(
    page_size: Tuple[float, float],
    *,
    page_index: int,
    requested_scale: float = PDF_RENDER_SCALE,
    min_scale: float = PDF_MIN_RENDER_SCALE,
    max_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
) -> float:
    w_pdf, h_pdf = float(page_size[0]), float(page_size[1])
    if w_pdf <= 0 or h_pdf <= 0:
        raise ValueError(
            f"Page {page_index}: Invalid PDF page size width={w_pdf}, height={h_pdf}."
        )
    if requested_scale <= 0:
        raise ValueError(f"PDF render scale must be positive, got {requested_scale}.")
    if min_scale <= 0:
        raise ValueError(f"Minimum PDF render scale must be positive, got {min_scale}.")
    if max_pixels <= 0:
        raise ValueError(f"Maximum image pixels must be positive, got {max_pixels}.")

    _, _, requested_pixels = estimate_pdf_render_pixels(page_size, requested_scale)
    if requested_pixels <= max_pixels:
        return requested_scale

    _, _, min_pixels = estimate_pdf_render_pixels(page_size, min_scale)
    if min_pixels > max_pixels:
        w_px, h_px, est = estimate_pdf_render_pixels(page_size, min_scale)
        msg = (
            f"Page {page_index}: Estimated render size width={w_px}, height={h_px} "
            f"(pixel count {est}) at minimum PDF render scale {min_scale} would exceed "
            f"maximum allowed {max_pixels}."
        )
        raise PDFRenderSizeError(
            msg,
            width=w_px,
            height=h_px,
            pixel_count=est,
            max_pixels=max_pixels,
            page_index=page_index,
        )

    upper = min(requested_scale, math.sqrt(max_pixels / (w_pdf * h_pdf)))
    lower = min_scale
    for _ in range(32):
        scale = (lower + upper) / 2
        _, _, pixels = estimate_pdf_render_pixels(page_size, scale)
        if pixels <= max_pixels:
            lower = scale
        else:
            upper = scale
    return lower


def render_pdf_page_to_numpy(
    page: Any,
    *,
    page_index: int,
    requested_scale: float = PDF_RENDER_SCALE,
    rotation: int = 0,
    min_scale: float = PDF_MIN_RENDER_SCALE,
    max_pixels: Optional[int] = DEFAULT_MAX_IMAGE_PIXELS,
) -> Any:
    if max_pixels is None:
        if requested_scale <= 0:
            raise ValueError(
                f"PDF render scale must be positive, got {requested_scale}."
            )
        scale = requested_scale
    else:
        scale = get_pdf_render_scale_within_pixel_limit(
            page.get_size(),
            page_index=page_index,
            requested_scale=requested_scale,
            min_scale=min_scale,
            max_pixels=max_pixels,
        )
    return page.render(scale=scale, rotation=rotation).to_numpy()
