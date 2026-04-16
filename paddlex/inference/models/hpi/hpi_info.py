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

"""HPI info and model metadata schema (shared by engines and registry)."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ValidationError
from typing_extensions import Literal, TypeAlias


class PaddleInferenceInfo(BaseModel):
    trt_dynamic_shapes: Optional[Dict[str, List[List[int]]]] = None
    trt_dynamic_shape_input_data: Optional[Dict[str, List[List[float]]]] = None


class TensorRTInfo(BaseModel):
    dynamic_shapes: Optional[Dict[str, List[List[int]]]] = None


class InferenceBackendInfoCollection(BaseModel):
    paddle_infer: Optional[PaddleInferenceInfo] = None
    tensorrt: Optional[TensorRTInfo] = None


class HPIInfo(BaseModel):
    backend_configs: Optional[InferenceBackendInfoCollection] = None


class ModelInfo(BaseModel):
    name: str
    hpi_info: Optional[HPIInfo] = None


ModelFormat: TypeAlias = Literal["paddle", "onnx", "om"]


def get_hpi_info(model_config: Optional[Dict[str, Any]]) -> Optional[HPIInfo]:
    """Extract and validate HPIInfo from a model config dict."""
    if not model_config or "Hpi" not in model_config:
        return None
    try:
        return HPIInfo.model_validate(model_config["Hpi"])
    except ValidationError as e:
        raise RuntimeError(f"Invalid HPI info: {str(e)}") from e
