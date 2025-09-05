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

from pydantic import BaseModel

from ..infra.models import DataInfo, PrimaryOperations
from .shared import ocr

__all__ = [
    "ANALYZE_IMAGES_ENDPOINT",
    "AnalyzeImagesRequest",
    "LayoutParsingResult",
    "AnalyzeImagesResult",
    "TRANSLATE_ENDPOINT",
    "TranslateRequest",
    "TranslationResult",
    "TranslateResult",
    "PRIMARY_OPERATIONS",
]

ANALYZE_IMAGES_ENDPOINT: Final[str] = "/doctrans-visual"


class AnalyzeImagesRequest(ocr.BaseInferRequest):
    useDocOrientationClassify: Optional[bool] = None
    useDocUnwarping: Optional[bool] = None
    useTextlineOrientation: Optional[bool] = None
    useSealRecognition: Optional[bool] = None
    useTableRecognition: Optional[bool] = None
    useFormulaRecognition: Optional[bool] = None
    useChartRecognition: Optional[bool] = None
    useRegionDetection: Optional[bool] = None
    layoutThreshold: Optional[Union[float, dict]] = None
    layoutNms: Optional[bool] = None
    layoutUnclipRatio: Optional[Union[float, Tuple[float, float], dict]] = None
    layoutMergeBboxesMode: Optional[Union[str, dict]] = None
    textDetLimitSideLen: Optional[int] = None
    textDetLimitType: Optional[str] = None
    textDetThresh: Optional[float] = None
    textDetBoxThresh: Optional[float] = None
    textDetUnclipRatio: Optional[float] = None
    textRecScoreThresh: Optional[float] = None
    sealDetLimitSideLen: Optional[int] = None
    sealDetLimitType: Optional[str] = None
    sealDetThresh: Optional[float] = None
    sealDetBoxThresh: Optional[float] = None
    sealDetUnclipRatio: Optional[float] = None
    sealRecScoreThresh: Optional[float] = None
    useWiredTableCellsTransToHtml: bool = False
    useWirelessTableCellsTransToHtml: bool = False
    useTableOrientationClassify: bool = True
    useOcrResultsWithTableCells: bool = True
    useE2eWiredTableRecModel: bool = False
    useE2eWirelessTableRecModel: bool = True
    visualize: Optional[bool] = None


class LayoutParsingResult(BaseModel):
    prunedResult: dict
    markdown: ocr.MarkdownData
    outputImages: Optional[Dict[str, str]] = None
    inputImage: Optional[str] = None


class AnalyzeImagesResult(BaseModel):
    layoutParsingResults: List[LayoutParsingResult]
    dataInfo: DataInfo


TRANSLATE_ENDPOINT: Final[str] = "/doctrans-translate"


class TranslateRequest(BaseModel):
    markdownList: List[ocr.MarkdownData]
    targetLanguage: str = "zh"
    chunkSize: int = 5000
    taskDescription: Optional[str] = None
    outputFormat: Optional[str] = None
    rulesStr: Optional[str] = None
    fewShotDemoTextContent: Optional[str] = None
    fewShotDemoKeyValueList: Optional[str] = None
    glossary: Optional[dict] = None
    llmRequestInterval: float = 0.0
    chatBotConfig: Optional[dict] = None


class TranslationResult(BaseModel):
    language: str
    markdown: ocr.MarkdownData


class TranslateResult(BaseModel):
    translationResults: List[TranslationResult]


PRIMARY_OPERATIONS: Final[PrimaryOperations] = {
    "analyzeImages": (
        ANALYZE_IMAGES_ENDPOINT,
        AnalyzeImagesRequest,
        AnalyzeImagesResult,
    ),
    "translate": (TRANSLATE_ENDPOINT, TranslateRequest, TranslateResult),
}
