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

"""Document exports for serving (DOCX and future formats). Shared by basic and HPS entrypoints."""

from __future__ import annotations

import base64
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL.Image import Image as PILImage

from paddlex.inference.pipelines.layout_parsing.utils import construct_img_path

from ....utils.deps import is_dep_available
from ..infra import utils as serving_utils
from ..infra.storage import Storage, SupportsGetURL
from ..schemas.shared.export import (
    DocumentExports,
    ExportContent,
    normalize_output_formats,
)

if is_dep_available("requests"):
    import requests


def _decode_image_value(value: str) -> PILImage:
    value = (value or "").strip()
    if not value:
        raise ValueError("Empty image payload")
    if value.startswith(("http://", "https://")):
        if not is_dep_available("requests"):
            raise RuntimeError(
                "The `requests` package is required to load export images from URLs."
            )
        resp = requests.get(value, timeout=120)
        resp.raise_for_status()
        return serving_utils.image_bytes_to_image(resp.content)
    if value.startswith("data:"):
        comma = value.find(",")
        if comma == -1:
            raise ValueError("Invalid data URL for image")
        b64_part = value[comma + 1 :]
        raw = base64.b64decode(b64_part)
        return serving_utils.image_bytes_to_image(raw)
    raw = base64.b64decode(value)
    return serving_utils.image_bytes_to_image(raw)


def refill_paddleocr_vl_images_from_markdown(
    result: Any, markdown_images: Optional[Dict[str, str]]
) -> None:
    """Restore PIL images on `PaddleOCRVLResult` blocks using path→payload from markdown.

    Payloads may be raw Base64, data URLs, or http(s) URLs (same as image outputs).
    """
    if not markdown_images:
        return

    ms = result.get("model_settings", {})
    use_chart_rec = ms.get("use_chart_recognition", False)

    for block in result["parsing_res_list"]:
        if isinstance(block.image, dict):
            p = block.image.get("path")
            if p and block.image.get("img") is None and p in markdown_images:
                block.image["img"] = _decode_image_value(markdown_images[p])
                continue

        if block.image is not None:
            continue

        label = block.label
        bbox = block.bbox
        if label in ("image", "header_image", "footer_image", "seal"):
            path = construct_img_path(label, bbox)
        elif label == "chart" and not use_chart_rec:
            path = construct_img_path(label, bbox)
        else:
            path = None

        if path is not None and path in markdown_images:
            img = _decode_image_value(markdown_images[path])
            block.image = {"path": path, "img": img}


def postprocess_docx_bytes(
    doc_bytes: bytes,
    log_id: str,
    filename: str,
    *,
    file_storage: Optional[Storage] = None,
    return_url: bool = False,
    url_expires_in: int = -1,
) -> str:
    if return_url:
        if not file_storage:
            raise ValueError(
                "`file_storage` must not be None when URLs need to be returned."
            )
        if not isinstance(file_storage, SupportsGetURL):
            raise TypeError("The provided storage does not support getting URLs.")

    key = f"{log_id}/{filename}"
    if file_storage is not None:
        file_storage.set(key, doc_bytes)
        if return_url:
            assert isinstance(file_storage, SupportsGetURL)
            return file_storage.get_url(key, expires_in=url_expires_in)
    return serving_utils.base64_encode(doc_bytes)


def docx_bytes_and_filename_from_word_mixin(result: Any) -> Tuple[bytes, str]:
    """Run `save_to_word` into a temporary directory and read produced bytes."""
    with tempfile.TemporaryDirectory() as tmp:
        result.save_to_word(tmp)
        stem = Path(result._get_input_fn()).stem
        docx_path = Path(tmp) / f"{stem}.docx"
        if not docx_path.is_file():
            raise FileNotFoundError(
                f"DOCX export did not produce expected file: {docx_path}"
            )
        return docx_path.read_bytes(), f"{stem}.docx"


def build_docx_export_content(
    result_obj: Any,
    *,
    log_id: str,
    file_storage: Optional[Storage] = None,
    return_urls: bool = False,
    url_expires_in: int = -1,
) -> ExportContent:
    doc_bytes, fname = docx_bytes_and_filename_from_word_mixin(result_obj)
    content = postprocess_docx_bytes(
        doc_bytes,
        log_id,
        fname,
        file_storage=file_storage,
        return_url=return_urls,
        url_expires_in=url_expires_in,
    )
    return ExportContent(content=content, fileName=fname)


def build_pipeline_exports(
    output_formats: Optional[List[str]],
    result_obj: Any,
    *,
    log_id: str,
    file_storage: Optional[Storage] = None,
    return_urls: bool = False,
    url_expires_in: int = -1,
) -> Optional[DocumentExports]:
    """Build `exports` for all requested formats. Returns None if nothing was requested."""
    fmts = normalize_output_formats(output_formats)
    if not fmts:
        return None

    kwargs = dict(
        log_id=log_id,
        file_storage=file_storage,
        return_urls=return_urls,
        url_expires_in=url_expires_in,
    )
    docx_content: Optional[ExportContent] = None
    for fmt in fmts:
        if fmt == "docx":
            docx_content = build_docx_export_content(result_obj, **kwargs)
        # Future non-docx formats: branch here or register handlers.

    return DocumentExports(docx=docx_content)
