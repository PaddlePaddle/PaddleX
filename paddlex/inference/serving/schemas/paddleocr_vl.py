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

from typing import Dict, Final, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator, model_validator
from typing_extensions import Annotated, Literal

from ..infra.models import DataInfo, PrimaryOperations
from .shared import ocr

__all__ = [
    "INFER_ENDPOINT",
    "InferRequest",
    "LayoutParsingResult",
    "InferResult",
    "RESTRUCTURE_PAGES_ENDPOINT",
    "RestructurePagesRequest",
    "RestructurePagesResult",
    "PRIMARY_OPERATIONS",
    "MarkdownData",
    "Page",
]

INFER_ENDPOINT: Final[str] = "/layout-parsing"
RESTRUCTURE_PAGES_ENDPOINT: Final[str] = "/restructure-pages"


class InferRequest(ocr.BaseInferRequest):
    useDocOrientationClassify: Optional[bool] = None
    useDocUnwarping: Optional[bool] = None
    useLayoutDetection: Optional[bool] = None
    useChartRecognition: Optional[bool] = None
    useSealRecognition: Optional[bool] = None
    useOcrForImageBlock: Optional[bool] = None
    layoutThreshold: Optional[Union[float, dict]] = None
    layoutNms: Optional[bool] = None
    layoutUnclipRatio: Optional[Union[float, Tuple[float, float], dict]] = None
    layoutMergeBboxesMode: Optional[Union[str, dict]] = None
    layoutShapeMode: Literal["rect", "quad", "poly", "auto"] = "auto"
    promptLabel: Optional[str] = None
    formatBlockContent: Optional[bool] = None
    repetitionPenalty: Optional[float] = None
    temperature: Optional[float] = None
    topP: Optional[float] = None
    minPixels: Optional[Annotated[int, Field(gt=0)]] = None
    maxPixels: Optional[Annotated[int, Field(gt=0)]] = None
    maxNewTokens: Optional[Annotated[int, Field(gt=0)]] = None
    mergeLayoutBlocks: Optional[bool] = None
    markdownIgnoreLabels: Optional[List[str]] = None
    vlmExtraArgs: Optional[dict] = None
    prettifyMarkdown: bool = True
    showFormulaNumber: bool = False
    restructurePages: bool = False
    mergeTables: bool = True
    relevelTitles: bool = True
    visualize: Optional[bool] = None
    logId: Optional[str] = None

    @field_validator("topP")
    @classmethod
    def validate_top_p(cls, v):
        if v is not None and not (0 < v <= 1):
            raise ValueError(f"`topP` must be > 0 and ≤ 1; got: {v}")
        return v

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v):
        if v is not None and v < 0:
            raise ValueError(f"`temperature` must be ≥ 0; got: {v}")
        return v

    @field_validator("repetitionPenalty")
    @classmethod
    def validate_repetition_penalty(cls, v):
        if v is not None and v <= 0:
            raise ValueError(f"`repetitionPenalty` must be > 0; got: {v}")
        return v

    @field_validator("promptLabel")
    @classmethod
    def validate_prompt_label(cls, v):
        _ALLOWED_PROMPT_LABELS = (
            "ocr",
            "formula",
            "table",
            "chart",
            "seal",
            "spotting",
        )
        if v is not None and v not in _ALLOWED_PROMPT_LABELS:
            valid_values = ", ".join(_ALLOWED_PROMPT_LABELS)
            raise ValueError(f"`promptLabel` must be one of: {valid_values}; got: {v}")
        return v

    @field_validator("layoutMergeBboxesMode")
    @classmethod
    def validate_merge_bboxes_mode(cls, v):
        _ALLOWED_MERGE_BBOXES_MODES = ("large", "small", "union")

        if v is None:
            return v

        if isinstance(v, str):
            if v not in _ALLOWED_MERGE_BBOXES_MODES:
                raise ValueError(
                    f"`layoutMergeBboxesMode` must be one of: {', '.join(_ALLOWED_MERGE_BBOXES_MODES)}; got: {v}"
                )
        elif isinstance(v, dict):
            for key, value in v.items():
                if not isinstance(value, str):
                    raise ValueError(
                        f"`layoutMergeBboxesMode` dictionary values must be strings; got: {type(value).__name__}"
                    )
                if value not in _ALLOWED_MERGE_BBOXES_MODES:
                    raise ValueError(
                        f"`layoutMergeBboxesMode` dictionary value must be one of: {', '.join(_ALLOWED_MERGE_BBOXES_MODES)}; got: {value}"
                    )
        else:
            raise ValueError("`layoutMergeBboxesMode` must be a string or dictionary")

        return v

    @field_validator("layoutUnclipRatio")
    @classmethod
    def validate_unclip_ratio(cls, v):
        if v is None:
            return v

        def _validate_ratio_value(value, context=""):
            if isinstance(value, (int, float)):
                if value <= 0:
                    raise ValueError(
                        f"`layoutUnclipRatio`{context} must be > 0; got: {value}"
                    )
            elif isinstance(value, list):
                if len(value) != 2:
                    raise ValueError(
                        f"`layoutUnclipRatio`{context} must be two numbers; got: {len(value)} values"
                    )
                for i, item in enumerate(value):
                    if not isinstance(item, (int, float)):
                        raise ValueError(
                            f"`layoutUnclipRatio`{context} values must be numbers; got: {type(item).__name__} at position {i}"
                        )
                    if item <= 0:
                        raise ValueError(
                            f"`layoutUnclipRatio`{context} values must be > 0; got: {item} at position {i}"
                        )
            else:
                raise ValueError(
                    f"`layoutUnclipRatio`{context} must be a number or two numbers; got: {type(value).__name__}"
                )

        if isinstance(v, dict):
            for key, value in v.items():
                _validate_ratio_value(value, f" value for key '{key}'")
        else:
            _validate_ratio_value(v)

        return v

    @field_validator("layoutThreshold")
    @classmethod
    def validate_threshold(cls, v):
        if v is None:
            return v

        def _validate_threshold_value(value, context=""):
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"`layoutThreshold`{context} must be a number; got: {type(value).__name__}"
                )
            if value < 0 or value > 1:
                raise ValueError(
                    f"`layoutThreshold`{context} must be between 0 and 1 inclusive; got: {value}"
                )

        if isinstance(v, dict):
            for key, value in v.items():
                _validate_threshold_value(value, f" value for key '{key}'")
        else:
            _validate_threshold_value(v)

        return v

    @model_validator(mode="after")
    def validate_pixel_range(self):
        if self.minPixels is not None and self.maxPixels is not None:
            if self.minPixels > self.maxPixels:
                raise ValueError(
                    f"`minPixels` ({self.minPixels}) cannot be greater than `maxPixels` ({self.maxPixels})"
                )
        return self


class MarkdownData(BaseModel):
    text: str
    images: Optional[Dict[str, str]] = None


class LayoutParsingResult(BaseModel):
    prunedResult: dict
    markdown: MarkdownData
    outputImages: Optional[Dict[str, str]] = None
    inputImage: Optional[str] = None


class InferResult(BaseModel):
    layoutParsingResults: List[LayoutParsingResult]
    dataInfo: DataInfo


class Page(BaseModel):
    prunedResult: dict
    markdownImages: Optional[Dict[str, str]] = None


class RestructurePagesRequest(BaseModel):
    pages: List[Page]
    mergeTables: bool = True
    relevelTitles: bool = True
    concatenatePages: bool = False
    prettifyMarkdown: bool = True
    showFormulaNumber: bool = False
    logId: Optional[str] = None


class RestructurePagesResult(BaseModel):
    layoutParsingResults: List[LayoutParsingResult]


PRIMARY_OPERATIONS: Final[PrimaryOperations] = {
    "infer": (INFER_ENDPOINT, InferRequest, InferResult),
    "restructure-pages": (
        RESTRUCTURE_PAGES_ENDPOINT,
        RestructurePagesRequest,
        RestructurePagesResult,
    ),
}
