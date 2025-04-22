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

from enum import Enum
from typing import Final, List, Literal, Optional, Union

from pydantic import BaseModel, HttpUrl

from ....utils.deps import is_dep_available
from ..infra.models import PrimaryOperations

if is_dep_available("openai"):
    from openai.types.chat import ChatCompletion

__all__ = [
    "INFER_ENDPOINT",
    "InferRequest",
    "PRIMARY_OPERATIONS",
]

INFER_ENDPOINT: Final[str] = "/document-understanding"


class ContentType(str, Enum):
    TEXT = "text"
    IMAGE_URL = "image_url"


class RoleType(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ImageUrl(BaseModel):
    url: Union[HttpUrl, str]
    detail: Optional[Literal["low", "high", "auto"]] = "auto"


class TextContent(BaseModel):
    type: Literal[ContentType.TEXT] = ContentType.TEXT
    text: str


class ImageContent(BaseModel):
    type: Literal[ContentType.IMAGE_URL] = ContentType.IMAGE_URL
    image_url: Union[HttpUrl, ImageUrl]


class Message(BaseModel):
    role: str
    content: Union[str, List[Union[TextContent, ImageContent]]]


class InferRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.1
    top_p: Optional[float] = 0.95
    stream: Optional[bool] = False
    max_image_tokens: Optional[int] = None


PRIMARY_OPERATIONS: Final[PrimaryOperations] = {
    "infer": (INFER_ENDPOINT, InferRequest, ChatCompletion),
}
