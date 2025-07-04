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
    "INFER_ENDPOINT",
    "InferRequest",
    "MarkdownData",
    "LayoutParsingResult",
    "InferResult",
    "PRIMARY_OPERATIONS",
]

INFER_ENDPOINT: Final[str] = "/layout-parsing"


class InferRequest(ocr.BaseInferRequest):
    useDocOrientationClassify: Optional[bool] = False
    useDocUnwarping: Optional[bool] = False
    useTextlineOrientation: Optional[bool] = None
    useSealRecognition: Optional[bool] = None
    useTableRecognition: Optional[bool] = None
    useFormulaRecognition: Optional[bool] = None
    useChartRecognition: Optional[bool] = False
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


class MarkdownData(BaseModel):
    text: str
    images: Dict[str, str]
    isStart: bool
    isEnd: bool


class LayoutParsingResult(BaseModel):
    prunedResult: dict
    markdown: MarkdownData
    outputImages: Optional[Dict[str, str]] = None
    inputImage: Optional[str] = None


class InferResult(BaseModel):
    layoutParsingResults: List[LayoutParsingResult]
    dataInfo: DataInfo


PRIMARY_OPERATIONS: Final[PrimaryOperations] = {
    "infer": (INFER_ENDPOINT, InferRequest, InferResult),
}
