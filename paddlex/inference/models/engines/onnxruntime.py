#!/usr/bin/env python3
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

"""ONNX Runtime engine."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type

from ....constants import MODEL_FILE_PREFIX
from ....utils.deps import is_dep_available
from ..runners import ONNXRuntimeRunner
from ..runners.inference_runner import InferenceRunner
from ..runners.onnxruntime_runner import ONNXRuntimeRunnerConfig
from ..utils.model_paths import LocalModelFormat
from ._base import RunnerBuilder, RunnerEngine


class ONNXRuntimeEngine(RunnerEngine):
    """Engine for ONNX Runtime inference."""

    entities = "onnxruntime"

    @property
    def name(self) -> str:
        return "onnxruntime"

    @property
    def engine_config_model(self) -> Type[ONNXRuntimeRunnerConfig]:
        return ONNXRuntimeRunnerConfig

    def get_supported_model_formats(
        self,
    ) -> Optional[Tuple[LocalModelFormat, ...]]:
        return ("onnx",)

    def prepare_config_dict(
        self,
        raw: Dict[str, Any],
        *,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ) -> Dict[str, Any]:
        del model_name
        self._apply_device(raw, device)
        return raw

    def ensure_environment(self) -> None:
        if not is_dep_available("onnxruntime"):
            raise RuntimeError(
                "Engine 'onnxruntime' is unavailable because dependency "
                "'onnxruntime' is not installed."
            )

    def _check_device_support(self, engine_config: Dict[str, Any]) -> None:
        device_type = engine_config.get("device_type")
        if device_type is None or device_type == "cpu":
            return
        if device_type != "gpu":
            raise ValueError(
                "`engine='onnxruntime'` currently only supports `cpu` and `gpu`."
            )
        import onnxruntime as ort

        available_providers = set(ort.get_available_providers())
        if "CUDAExecutionProvider" not in available_providers:
            raise RuntimeError(
                "ONNX Runtime GPU inference is unavailable because "
                "`CUDAExecutionProvider` is not available. "
                f"Available providers: {sorted(available_providers)!r}."
            )

    def get_default_runner_builder(self) -> RunnerBuilder:
        def runner_builder(
            *,
            model_name: str,
            model_dir: Optional[Path],
            model_config: Optional[Dict[str, Any]],
            engine_config: Dict[str, Any],
            default_builder: Optional[RunnerBuilder] = None,
        ) -> InferenceRunner:
            del model_name, model_config, default_builder
            if model_dir is None:
                raise ValueError("`model_dir` is required for engine='onnxruntime'.")
            self._check_device_support(engine_config)
            return ONNXRuntimeRunner(
                model_dir=model_dir,
                model_file_prefix=MODEL_FILE_PREFIX,
                config=engine_config,
            )

        return runner_builder

    def validate_runner(self, runner: InferenceRunner) -> None:
        if not isinstance(runner, ONNXRuntimeRunner):
            raise TypeError(
                "Engine 'onnxruntime' must build an ONNXRuntimeRunner, "
                f"but got {type(runner).__name__}."
            )
