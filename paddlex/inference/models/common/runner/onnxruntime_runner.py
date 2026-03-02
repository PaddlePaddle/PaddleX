# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""ONNXRuntime runner."""

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from .....utils.deps import class_requires_deps

__all__ = ["ONNXRuntimeRunnerConfig", "ONNXRuntimeRunner"]


class ONNXRuntimeRunnerConfig(BaseModel):
    """Engine config for onnxruntime inference."""

    model_config = ConfigDict(extra="forbid")

    device_type: Optional[str] = None
    device_id: Optional[int] = None
    providers: Optional[List[str]] = None
    provider_options: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None
    graph_optimization_level: Optional[int] = None
    intra_op_num_threads: Optional[int] = None
    inter_op_num_threads: Optional[int] = None
    execution_mode: Optional[str] = None
    log_severity_level: Optional[int] = None
    enable_mem_pattern: Optional[bool] = None
    enable_cpu_mem_arena: Optional[bool] = None
    session_options: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def check_provider_options(self):
        if (
            self.providers is not None
            and isinstance(self.provider_options, list)
            and len(self.providers) != len(self.provider_options)
        ):
            raise ValueError(
                "Length mismatch between `providers` and `provider_options`."
            )
        return self


@class_requires_deps("onnxruntime")
class ONNXRuntimeRunner:
    """Placeholder for ONNXRuntime inference - not yet implemented."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def __call__(self, x: Sequence[np.ndarray]) -> List[np.ndarray]:
        raise NotImplementedError(
            "ONNXRuntime engine is not yet implemented for direct inference. "
            "Consider using engine='paddle' or HPIP with onnxruntime backend."
        )
