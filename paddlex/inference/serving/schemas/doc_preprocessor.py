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

from typing import Final, List, Optional

from pydantic import BaseModel

from ..infra.models import DataInfo, PrimaryOperations
from .shared import ocr

__all__ = [
    "INFER_ENDPOINT",
    "InferRequest",
    "DocPreprocessingResult",
    "InferResult",
    "PRIMARY_OPERATIONS",
]

INFER_ENDPOINT: Final[str] = "/document-preprocessing"


class InferRequest(ocr.BaseInferRequest):
    # Should it be "Classification" instead of "Classify"? Keep the names
    # consistent with the parameters of the wrapped function though.
    useDocOrientationClassify: Optional[bool] = None
    useDocUnwarping: Optional[bool] = None


class DocPreprocessingResult(BaseModel):
    outputImage: str
    prunedResult: dict
    docPreprocessingImage: Optional[str] = None
    inputImage: Optional[str] = None


class InferResult(BaseModel):
    docPreprocessingResults: List[DocPreprocessingResult]
    dataInfo: DataInfo


PRIMARY_OPERATIONS: Final[PrimaryOperations] = {
    "infer": (INFER_ENDPOINT, InferRequest, InferResult),
}
