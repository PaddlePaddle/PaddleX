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

"""ONNX Runtime runner."""

from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from ....utils.deps import class_requires_deps
from ..utils.model_paths import get_model_paths
from .inference_runner import InferenceRunner
from .utils import sort_inputs

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
        if (
            self.providers is not None
            and isinstance(self.provider_options, dict)
            and len(self.providers) != 1
        ):
            raise ValueError(
                "When `provider_options` is a dict, `providers` must contain exactly one provider."
            )
        return self


@class_requires_deps("onnxruntime")
class ONNXRuntimeRunner(InferenceRunner):
    """ONNX Runtime inference runner."""

    def __init__(
        self,
        model_dir: Union[str, PathLike],
        model_file_prefix: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        import onnxruntime as ort

        super().__init__()
        self.model_dir = Path(model_dir)
        self.model_file_prefix = model_file_prefix
        self._config = ONNXRuntimeRunnerConfig.model_validate(config or {}).model_dump(
            exclude_none=True
        )
        self._ort = ort
        self.session = self._create_session()
        self._input_names = [meta.name for meta in self.session.get_inputs()]
        self._output_names = [meta.name for meta in self.session.get_outputs()]

    def __call__(
        self,
        x: Union[Sequence[np.ndarray], np.ndarray, None] = None,
        **kwargs: Any,
    ) -> List[np.ndarray]:
        if x is None and "x" in kwargs:
            x = kwargs["x"]
        if x is None:
            raise TypeError("`ONNXRuntimeRunner.__call__` requires `x`")
        if isinstance(x, np.ndarray):
            x = [x]

        if len(self._input_names) != len(x):
            raise ValueError(
                f"The number of inputs does not match the model: "
                f"{len(self._input_names)} vs {len(x)}"
            )

        x = sort_inputs(x, self._input_names)
        feeds = {
            name: np.ascontiguousarray(input_)
            for name, input_ in zip(self._input_names, x)
        }
        return self.session.run(self._output_names, feeds)

    def close(self) -> None:
        pass

    def _create_session(self):
        model_path = self._get_model_path()
        session_options = self._build_session_options()
        providers, provider_options = self._resolve_providers()
        return self._ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=providers,
            provider_options=provider_options,
        )

    def _get_model_path(self) -> Path:
        model_paths = get_model_paths(self.model_dir, self.model_file_prefix)
        if "onnx" not in model_paths:
            raise RuntimeError("No valid ONNX model found")
        return model_paths["onnx"]

    def _build_session_options(self):
        sess_options = self._ort.SessionOptions()

        graph_level = self._config.get("graph_optimization_level")
        if graph_level is not None:
            sess_options.graph_optimization_level = self._ort.GraphOptimizationLevel(
                graph_level
            )

        execution_mode = self._config.get("execution_mode")
        if execution_mode is not None:
            sess_options.execution_mode = self._resolve_execution_mode(execution_mode)

        for key in (
            "intra_op_num_threads",
            "inter_op_num_threads",
            "log_severity_level",
            "enable_mem_pattern",
            "enable_cpu_mem_arena",
        ):
            value = self._config.get(key)
            if value is not None:
                setattr(sess_options, key, value)

        for key, value in self._config.get("session_options", {}).items():
            if not hasattr(sess_options, key):
                raise ValueError(f"Invalid ONNX Runtime session option: {key!r}")
            setattr(sess_options, key, value)

        return sess_options

    def _resolve_execution_mode(self, execution_mode):
        if isinstance(execution_mode, int):
            return self._ort.ExecutionMode(execution_mode)

        mode = execution_mode.lower()
        if mode in {"sequential", "ort_sequential"}:
            return self._ort.ExecutionMode.ORT_SEQUENTIAL
        if mode in {"parallel", "ort_parallel"}:
            return self._ort.ExecutionMode.ORT_PARALLEL
        raise ValueError(
            "Invalid `execution_mode`, expected one of "
            "'sequential', 'parallel', 'ORT_SEQUENTIAL', 'ORT_PARALLEL'."
        )

    def _resolve_providers(self):
        providers = self._config.get("providers")
        provider_options = self._config.get("provider_options")

        if providers is None:
            providers, default_provider_options = self._default_providers()
            if provider_options is None:
                provider_options = default_provider_options
            elif isinstance(provider_options, dict):
                provider_options = [provider_options, *default_provider_options[1:]]
        elif isinstance(provider_options, dict):
            provider_options = [provider_options]

        provider_options = self._inject_default_provider_options(
            providers, provider_options
        )
        self._validate_providers(providers)
        return providers, provider_options

    def _default_providers(self):
        device_type = (self._config.get("device_type") or "cpu").lower()
        device_id = self._config.get("device_id")

        if device_type == "cpu":
            return ["CPUExecutionProvider"], None

        if device_type == "gpu":
            if device_id is None:
                device_id = 0
            return [
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ], [{"device_id": device_id}, {}]

        raise ValueError(
            "`engine='onnxruntime'` only supports `device_type` of "
            "'cpu' or 'gpu' unless `providers` is specified explicitly."
        )

    def _validate_providers(self, providers):
        available = set(self._ort.get_available_providers())
        missing = [provider for provider in providers if provider not in available]
        if missing:
            raise RuntimeError(
                f"Requested ONNX Runtime providers are unavailable: {missing!r}. "
                f"Available providers: {sorted(available)!r}."
            )

    def _inject_default_provider_options(self, providers, provider_options):
        if provider_options is None:
            provider_options = [{} for _ in providers]
        else:
            provider_options = list(provider_options)

        device_id = self._config.get("device_id")
        if device_id is None:
            return provider_options

        for idx, provider in enumerate(providers):
            if provider == "CUDAExecutionProvider":
                provider_options[idx] = {
                    "device_id": device_id,
                    **provider_options[idx],
                }

        return provider_options
