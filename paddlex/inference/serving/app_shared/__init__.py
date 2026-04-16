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

"""Shared application logic for serving entrypoints (basic FastAPI and HPS), separate from infra."""

from .document_export import (
    build_docx_export_content,
    build_pipeline_exports,
    docx_bytes_and_filename_from_word_mixin,
    postprocess_docx_bytes,
    refill_paddleocr_vl_images_from_markdown,
)

__all__ = [
    "build_docx_export_content",
    "build_pipeline_exports",
    "docx_bytes_and_filename_from_word_mixin",
    "postprocess_docx_bytes",
    "refill_paddleocr_vl_images_from_markdown",
]
