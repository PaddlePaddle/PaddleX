# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from typing import TYPE_CHECKING, Optional

import numpy as np
from paddlex.inference.serving.infra.models import (
    AIStudioNoResultResponse,
    AIStudioResultResponse,
    DataInfo,
    ImageInfo,
    PDFInfo,
    PDFPageInfo,
    Response,
    ResultT,
)
from paddlex.inference.serving.infra.utils import generate_log_id

__all__ = [
    "DataInfo",
    "ImageInfo",
    "PDFInfo",
    "PDFPageInfo",
    "generate_log_id",
    "AIStudioOutputWithoutResult",
    "AIStudioOutputWithResult",
    "Output",
    "create_aistudio_output_without_result",
    "create_aistudio_output_with_result",
    "parse_triton_input",
    "create_triton_output",
]

AIStudioOutputWithoutResult = AIStudioNoResultResponse
AIStudioOutputWithResult = AIStudioResultResponse
Output = Response


def create_aistudio_output_without_result(
    error_code: int, error_msg: str, *, log_id: Optional[str] = None
) -> AIStudioOutputWithoutResult:
    if log_id is None:
        log_id = generate_log_id()
    output = AIStudioOutputWithoutResult(
        logId=log_id, errorCode=error_code, errorMsg=error_msg
    )
    return output


def create_aistudio_output_with_result(
    result: ResultT, log_id: Optional[str] = None
) -> AIStudioOutputWithResult[ResultT]:
    if log_id is None:
        log_id = generate_log_id()
    if TYPE_CHECKING:
        output_type = AIStudioOutputWithResult[ResultT]
    else:
        # HACK: We should have used `AIStudioOutputWithResult[type(result)]`
        # instead, as it matches the type hint more accurately. However,
        # Pydantic's implementation relies on stack frame info, which is missing
        # in the Cython-compiled code. Therefore, we have to use the bare
        # `AIStudioOutputWithResult` here.
        output_type = AIStudioOutputWithResult
    output = output_type(logId=log_id, errorCode=0, errorMsg="Success", result=result)
    return output


def parse_triton_input(data, model_type):
    if data.shape != (1, 1):
        raise ValueError(f"Invalid shape: {data.shape}")
    data = data[0, 0]
    data = data.decode("utf-8")
    model = model_type.model_validate_json(data)
    return model


def create_triton_output(model):
    data = model.model_dump_json(exclude_none=True)
    data = data.encode("utf-8")
    data = [[data]]
    return np.array(data, dtype=np.object_)
