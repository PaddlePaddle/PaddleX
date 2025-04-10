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

from typing import Final, List, Tuple

from pydantic import BaseModel

from ..infra.models import PrimaryOperations

__all__ = [
    "INFER_ENDPOINT",
    "InferRequest",
    "DetectedObject",
    "InferResult",
    "PRIMARY_OPERATIONS",
]

INFER_ENDPOINT: Final[str] = "/bev-3d-object-detection"


class InferRequest(BaseModel):
    tar: str


class DetectedObject(BaseModel):
    bbox: Tuple[float, float, float, float, float, float, float, float, float]
    categoryId: int
    score: float


class InferResult(BaseModel):
    detectedObjects: List[DetectedObject]


PRIMARY_OPERATIONS: Final[PrimaryOperations] = {
    "infer": (INFER_ENDPOINT, InferRequest, InferResult),
}
