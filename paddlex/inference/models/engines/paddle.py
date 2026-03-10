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

"""Engine specs for Paddle engines."""

from pathlib import Path
from typing import Any, Dict, Optional, Type

from ....constants import MODEL_FILE_PREFIX
from ....utils.deps import is_dep_available
from ....utils.device import parse_device
from ...utils.model_paths import get_model_paths, resolve_paddle_engine_from_model_files
from ..predictors import BasePredictor, RunnerPredictor
from ..runners.paddle_dynamic_runner import PaddleDynamicRunnerConfig
from ..runners.paddle_static_runner import PaddleStaticRunnerConfig
from ._base import EngineSpec


class PaddleEngineSpec(EngineSpec):
    entities = "paddle"

    @property
    def name(self) -> str:
        return "paddle"

    def get_base_predictor_cls(self) -> Type[BasePredictor]:
        return RunnerPredictor

    def ensure_predictor_support(self, model_name: str) -> None:
        self.get_predictor_cls(model_name)

    def resolve_engine_from_model_dir(self, model_dir: Path) -> str:
        resolved = resolve_paddle_engine_from_model_files(model_dir, MODEL_FILE_PREFIX)
        if resolved is None:
            raise ValueError("No Paddle model files were found.")
        return resolved


class PaddleStaticEngineSpec(EngineSpec):
    entities = "paddle_static"

    @property
    def name(self) -> str:
        return "paddle_static"

    @property
    def engine_config_model(self) -> Type[PaddleStaticRunnerConfig]:
        return PaddleStaticRunnerConfig

    def get_base_predictor_cls(self) -> Type[BasePredictor]:
        return RunnerPredictor

    def prepare_config_dict(
        self,
        raw: Dict[str, Any],
        *,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ) -> Dict[str, Any]:
        del model_name
        valid_fields = set(PaddleStaticRunnerConfig.model_fields)
        raw = {key: value for key, value in raw.items() if key in valid_fields}
        if device:
            device_type, device_ids = parse_device(device)
            raw["device_type"] = device_type
            raw["device_id"] = device_ids[0] if device_ids is not None else None
        return raw

    def ensure_model_files(self, model_dir: Path) -> None:
        if "paddle" not in get_model_paths(model_dir, MODEL_FILE_PREFIX):
            raise ValueError("No valid Paddle static model files were found.")

    def ensure_environment(
        self,
        *,
        device: Optional[str] = None,
        engine_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        del device, engine_config
        if not is_dep_available("paddlepaddle"):
            raise RuntimeError(
                "Engine 'paddle_static' is unavailable because dependency "
                "'paddlepaddle' is not installed."
            )


class PaddleDynamicEngineSpec(EngineSpec):
    entities = "paddle_dynamic"

    @property
    def name(self) -> str:
        return "paddle_dynamic"

    @property
    def engine_config_model(self) -> Type[PaddleDynamicRunnerConfig]:
        return PaddleDynamicRunnerConfig

    def get_base_predictor_cls(self) -> Type[BasePredictor]:
        return RunnerPredictor

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

    def ensure_model_files(self, model_dir: Path) -> None:
        model_paths = get_model_paths(model_dir, MODEL_FILE_PREFIX)
        if "safetensors" not in model_paths and "paddle_dyn" not in model_paths:
            raise ValueError("No valid Paddle dynamic model files were found.")

    def ensure_environment(
        self,
        *,
        device: Optional[str] = None,
        engine_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        del device, engine_config
        if not is_dep_available("paddlepaddle"):
            raise RuntimeError(
                "Engine 'paddle_dynamic' is unavailable because dependency "
                "'paddlepaddle' is not installed."
            )
