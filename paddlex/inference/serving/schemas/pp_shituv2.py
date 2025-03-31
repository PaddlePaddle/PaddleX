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

from ..infra.models import PrimaryOperations
from .shared import object_detection

__all__ = [
    "ImageLabelPair",
    "BUILD_INDEX_ENDPOINT",
    "BuildIndexRequest",
    "BuildIndexResult",
    "ADD_IMAGES_TO_INDEX_ENDPOINT",
    "AddImagesToIndexRequest",
    "AddImagesToIndexResult",
    "REMOVE_IMAGES_FROM_INDEX_ENDPOINT",
    "RemoveImagesFromIndexRequest",
    "RemoveImagesFromIndexResult",
    "INFER_ENDPOINT",
    "InferRequest",
    "RecResult",
    "DetectedObject",
    "InferResult",
    "PRIMARY_OPERATIONS",
]


class ImageLabelPair(BaseModel):
    image: str
    label: str


BUILD_INDEX_ENDPOINT: Final[str] = "/shitu-index-build"


class BuildIndexRequest(BaseModel):
    imageLabelPairs: List[ImageLabelPair]


class BuildIndexResult(BaseModel):
    indexKey: str
    imageCount: int


ADD_IMAGES_TO_INDEX_ENDPOINT: Final[str] = "/shitu-index-add"


class AddImagesToIndexRequest(BaseModel):
    imageLabelPairs: List[ImageLabelPair]
    indexKey: str


class AddImagesToIndexResult(BaseModel):
    imageCount: int


REMOVE_IMAGES_FROM_INDEX_ENDPOINT: Final[str] = "/shitu-index-remove"


class RemoveImagesFromIndexRequest(BaseModel):
    ids: List[int]
    indexKey: str


class RemoveImagesFromIndexResult(BaseModel):
    imageCount: int


INFER_ENDPOINT: Final[str] = "/shitu-infer"


class InferRequest(BaseModel):
    image: str
    indexKey: Optional[str] = None
    detThreshold: Optional[float] = None
    recThreshold: Optional[float] = None
    hammingRadius: Optional[float] = None
    topk: Optional[int] = None


class RecResult(BaseModel):
    label: str
    score: float


class DetectedObject(BaseModel):
    bbox: object_detection.BoundingBox
    recResults: List[RecResult]
    score: float


class InferResult(BaseModel):
    detectedObjects: List[DetectedObject]
    image: Optional[str] = None


PRIMARY_OPERATIONS: Final[PrimaryOperations] = {
    "buildIndex": (BUILD_INDEX_ENDPOINT, BuildIndexRequest, BuildIndexResult),
    "addImagesToIndex": (
        ADD_IMAGES_TO_INDEX_ENDPOINT,
        AddImagesToIndexRequest,
        AddImagesToIndexResult,
    ),
    "removeImagesFromIndex": (
        REMOVE_IMAGES_FROM_INDEX_ENDPOINT,
        RemoveImagesFromIndexRequest,
        RemoveImagesFromIndexResult,
    ),
    "infer": (INFER_ENDPOINT, InferRequest, InferResult),
}
