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

from typing import FrozenSet, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "SUPPORTED_OUTPUT_FORMATS",
    "ExportContent",
    "DocumentExports",
    "normalize_output_formats",
]

SUPPORTED_OUTPUT_FORMATS: FrozenSet[str] = frozenset({"docx"})


class ExportContent(BaseModel):
    """Single exported artifact. `content` holds Base64 or a URL, mirroring image fields."""

    content: str
    fileName: str


class DocumentExports(BaseModel):
    """Structured export payload; extend with optional fields as new formats are added."""

    model_config = ConfigDict(extra="forbid")

    docx: Optional[ExportContent] = None


def normalize_output_formats(formats: Optional[List[str]]) -> List[str]:
    if not formats:
        return []
    return list(formats)


def validate_output_formats_list(formats: Optional[List[str]]) -> None:
    """Raise ValueError if any format string is unsupported."""
    if not formats:
        return
    for fmt in formats:
        if fmt not in SUPPORTED_OUTPUT_FORMATS:
            raise ValueError(
                f"Unsupported `outputFormats` entry {fmt!r}. "
                f"Supported values: {sorted(SUPPORTED_OUTPUT_FORMATS)}"
            )


class OutputFormatsMixin(BaseModel):
    outputFormats: Optional[List[str]] = Field(
        default=None,
        description='Optional list of extra formats to return, e.g. ["docx"].',
    )

    @field_validator("outputFormats")
    @classmethod
    def _check_output_formats(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is None:
            return v
        validate_output_formats_list(v)
        return v
