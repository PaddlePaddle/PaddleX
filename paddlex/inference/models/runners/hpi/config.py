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

import os
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypeAlias

from ...hpi.hpi_info import HPIInfo

InferenceBackend: TypeAlias = Literal[
    "paddle", "openvino", "onnxruntime", "tensorrt", "om"
]


class HPIConfig(BaseModel):
    pdx_model_name: Annotated[str, Field(alias="model_name")]
    device_type: str
    device_id: Optional[int] = None
    auto_config: bool = True
    backend: Optional[InferenceBackend] = None
    backend_config: Optional[Dict[str, Any]] = None
    hpi_info: Optional[HPIInfo] = None
    auto_paddle2onnx: bool = True


class OpenVINOConfig(BaseModel):
    cpu_num_threads: int = Field(
        default_factory=lambda: int(os.getenv("PADDLE_PDX_CPU_NUM_THREADS", 10))
    )


class ONNXRuntimeConfig(BaseModel):
    cpu_num_threads: int = Field(
        default_factory=lambda: int(os.getenv("PADDLE_PDX_CPU_NUM_THREADS", 10))
    )


class TensorRTConfig(BaseModel):
    precision: Literal["fp32", "fp16"] = "fp32"
    use_dynamic_shapes: bool = True
    dynamic_shapes: Optional[Dict[str, List[List[int]]]] = None


class OMConfig(BaseModel):
    pass
