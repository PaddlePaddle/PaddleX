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
    "BUILD_VECTOR_STORE_ENDPOINT",
    "BuildVectorStoreRequest",
    "BuildVectorStoreResult",
    "INVOKE_MLLM_ENDPOINT",
    "InvokeMLLMRequest",
    "InvokeMLLMResult",
    "CHAT_ENDPOINT",
    "ChatRequest",
    "ChatResult",
    "PRIMARY_OPERATIONS",
]

ANALYZE_IMAGES_ENDPOINT: Final[str] = "/chatocr-visual"


class AnalyzeImagesRequest(ocr.BaseInferRequest):
    useDocOrientationClassify: Optional[bool] = None
    useDocUnwarping: Optional[bool] = None
    useTextlineOrientation: Optional[bool] = None
    useSealRecognition: Optional[bool] = None
    useTableRecognition: Optional[bool] = None
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


class LayoutParsingResult(BaseModel):
    prunedResult: dict
    outputImages: Optional[Dict[str, str]] = None
    inputImage: Optional[str] = None


class AnalyzeImagesResult(BaseModel):
    layoutParsingResults: List[LayoutParsingResult]
    visualInfo: List[dict]
    dataInfo: DataInfo


BUILD_VECTOR_STORE_ENDPOINT: Final[str] = "/chatocr-vector"


class BuildVectorStoreRequest(BaseModel):
    visualInfo: List[dict]
    minCharacters: Optional[int] = None
    blockSize: Optional[int] = None
    retrieverConfig: Optional[dict] = None


class BuildVectorStoreResult(BaseModel):
    vectorInfo: dict


INVOKE_MLLM_ENDPOINT: Final[str] = "/chatocr-mllm"


class InvokeMLLMRequest(BaseModel):
    image: str
    keyList: List[str]
    mllmChatBotConfig: Optional[dict] = None


class InvokeMLLMResult(BaseModel):
    mllmPredictInfo: dict


CHAT_ENDPOINT: Final[str] = "/chatocr-chat"


class ChatRequest(BaseModel):
    keyList: List[str]
    visualInfo: List[dict]
    useVectorRetrieval: Optional[bool] = None
    vectorInfo: Optional[dict] = None
    minCharacters: Optional[int] = None
    textTaskDescription: Optional[str] = None
    textOutputFormat: Optional[str] = None
    textRulesStr: Optional[str] = None
    textFewShotDemoTextContent: Optional[str] = None
    textFewShotDemoKeyValueList: Optional[str] = None
    tableTaskDescription: Optional[str] = None
    tableOutputFormat: Optional[str] = None
    tableRulesStr: Optional[str] = None
    tableFewShotDemoTextContent: Optional[str] = None
    tableFewShotDemoKeyValueList: Optional[str] = None
    mllmPredictInfo: Optional[dict] = None
    mllmIntegrationStrategy: Optional[str] = None
    chatBotConfig: Optional[dict] = None
    retrieverConfig: Optional[dict] = None


class ChatResult(BaseModel):
    chatResult: dict


PRIMARY_OPERATIONS: Final[PrimaryOperations] = {
    "analyzeImages": (
        ANALYZE_IMAGES_ENDPOINT,
        AnalyzeImagesRequest,
        AnalyzeImagesResult,
    ),
    "buildVectorStore": (
        BUILD_VECTOR_STORE_ENDPOINT,
        BuildVectorStoreRequest,
        BuildVectorStoreResult,
    ),
    "invokeMllm": (
        INVOKE_MLLM_ENDPOINT,
        InvokeMLLMRequest,
        InvokeMLLMResult,
    ),
    "chat": (CHAT_ENDPOINT, ChatRequest, ChatResult),
}
