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

from typing import Dict, Generic, List, Tuple, TypeVar, Union

from pydantic import BaseModel, Discriminator
from typing_extensions import Annotated, Literal, TypeAlias

from ....utils.deps import is_dep_available

if is_dep_available("openai"):
    from openai.types.chat import ChatCompletion

__all__ = [
    "AIStudioNoResultResponse",
    "ResultT",
    "AIStudioResultResponse",
    "Response",
    "ImageInfo",
    "PDFPageInfo",
    "PDFInfo",
    "DataInfo",
    "PrimaryOperations",
]


class AIStudioNoResultResponse(BaseModel):
    logId: str
    errorCode: int
    errorMsg: str


ResultT = TypeVar("ResultT", bound=BaseModel)


class AIStudioResultResponse(BaseModel, Generic[ResultT]):
    logId: str
    result: ResultT
    errorCode: Literal[0] = 0
    errorMsg: Literal["Success"] = "Success"


Response: TypeAlias = Union[
    AIStudioResultResponse, AIStudioNoResultResponse, "ChatCompletion"
]


class ImageInfo(BaseModel):
    width: int
    height: int
    type: Literal["image"] = "image"


class PDFPageInfo(BaseModel):
    width: int
    height: int


class PDFInfo(BaseModel):
    numPages: int
    pages: List[PDFPageInfo]
    type: Literal["pdf"] = "pdf"


DataInfo: TypeAlias = Annotated[Union[ImageInfo, PDFInfo], Discriminator("type")]

# Should we use generics?
PrimaryOperations: TypeAlias = Dict[str, Tuple[str, BaseModel, BaseModel]]
