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
    "SealRecResult",
    "InferResult",
    "PRIMARY_OPERATIONS",
]

INFER_ENDPOINT: Final[str] = "/seal-recognition"


class InferRequest(ocr.BaseInferRequest):
    useDocOrientationClassify: Optional[bool] = None
    useDocUnwarping: Optional[bool] = None
    useLayoutDetection: Optional[bool] = None
    layoutThreshold: Optional[Union[float, dict]] = None
    layoutNms: Optional[bool] = None
    layoutUnclipRatio: Optional[Union[float, Tuple[float, float]]] = None
    layoutMergeBboxesMode: Optional[str] = None
    sealDetLimitSideLen: Optional[int] = None
    sealDetLimitType: Optional[str] = None
    sealDetThresh: Optional[float] = None
    sealDetBoxThresh: Optional[float] = None
    sealDetUnclipRatio: Optional[float] = None
    sealRecScoreThresh: Optional[float] = None


class SealRecResult(BaseModel):
    prunedResult: dict
    outputImages: Optional[Dict[str, str]] = None
    inputImage: Optional[str] = None


class InferResult(BaseModel):
    sealRecResults: List[SealRecResult]
    dataInfo: DataInfo


PRIMARY_OPERATIONS: Final[PrimaryOperations] = {
    "infer": (INFER_ENDPOINT, InferRequest, InferResult),
}
