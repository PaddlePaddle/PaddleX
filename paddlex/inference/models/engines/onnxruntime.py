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
from ....utils.device import parse_device
from ..runners import ONNXRuntimeRunner
from ..runners.inference_runner import InferenceRunner
from ..runners.onnxruntime_runner import ONNXRuntimeRunnerConfig
from ..utils.model_paths import LocalModelFormat
from ._base import RunnerEngine


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
        if device:
            device_type, device_ids = parse_device(device)
            raw["device_type"] = device_type
            raw["device_id"] = device_ids[0] if device_ids is not None else None
        return raw

    def ensure_environment(
        self,
        *,
        device: Optional[str] = None,
        engine_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not is_dep_available("onnxruntime"):
            raise RuntimeError(
                "Engine 'onnxruntime' is unavailable because dependency "
                "'onnxruntime' is not installed."
            )
        device_type = (engine_config or {}).get("device_type")
        if device_type is None and device is not None:
            device_type, _ = parse_device(device)
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

    def build_runner(
        self,
        *,
        model_name: str,
        model_dir: Optional[Path],
        model_config: Optional[Dict[str, Any]],
        engine_config: Dict[str, Any],
        binding: Any = None,
    ) -> InferenceRunner:
        del model_name, model_config, binding
        if model_dir is None:
            raise ValueError("`model_dir` is required for engine='onnxruntime'.")
        return ONNXRuntimeRunner(
            model_dir=model_dir,
            model_file_prefix=MODEL_FILE_PREFIX,
            config=engine_config,
        )
