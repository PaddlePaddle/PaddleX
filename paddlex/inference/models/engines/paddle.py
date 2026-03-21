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

"""Paddle engines."""
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type

from pydantic import ValidationError

from ....constants import MODEL_FILE_PREFIX
from ....utils import logging
from ....utils.deps import is_dep_available
from ....utils.device import parse_device
from ..hpi import HPIInfo
from ..runners import PaddleStaticRunner
from ..runners.inference_runner import InferenceRunner
from ..runners.paddle_dynamic_runner import PaddleDynamicRunnerConfig
from ..runners.paddle_static import PaddleStaticRunnerConfig
from ..utils.model_paths import LocalModelFormat
from ._base import InferenceEngine


def _get_hpi_info(model_config: Optional[Dict[str, Any]]) -> Optional[HPIInfo]:
    if not model_config or "Hpi" not in model_config:
        return None
    try:
        return HPIInfo.model_validate(model_config["Hpi"])
    except ValidationError as e:
        raise RuntimeError(f"Invalid HPI info: {str(e)}") from e


def _inject_trt_info(
    model_config: Optional[Dict[str, Any]],
    engine_config: Dict[str, Any],
) -> Dict[str, Any]:
    hpi_info = _get_hpi_info(model_config)
    if hpi_info is None:
        return engine_config
    paddle_info = None
    if hpi_info.backend_configs:
        paddle_info = hpi_info.backend_configs.paddle_infer
    if paddle_info is None:
        return engine_config
    if (
        engine_config.get("trt_dynamic_shapes") is None
        and paddle_info.trt_dynamic_shapes is not None
    ):
        logging.debug(
            "TensorRT dynamic shapes set to %s", paddle_info.trt_dynamic_shapes
        )
        engine_config = {
            **engine_config,
            "trt_dynamic_shapes": paddle_info.trt_dynamic_shapes,
        }
    if (
        engine_config.get("trt_dynamic_shape_input_data") is None
        and paddle_info.trt_dynamic_shape_input_data is not None
    ):
        logging.debug(
            "TensorRT dynamic shape input data set to %s",
            paddle_info.trt_dynamic_shape_input_data,
        )
        engine_config = {
            **engine_config,
            "trt_dynamic_shape_input_data": paddle_info.trt_dynamic_shape_input_data,
        }
    return engine_config


class PaddleStaticEngineSpec(InferenceEngine):
    """Engine for Paddle static-graph inference."""

    entities = "paddle_static"

    @property
    def name(self) -> str:
        return "paddle_static"

    @property
    def engine_config_model(self) -> Type[PaddleStaticRunnerConfig]:
        return PaddleStaticRunnerConfig

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

    def build_runner(
        self,
        *,
        model_name: str,
        model_dir: Optional[Path],
        model_config: Optional[Dict[str, Any]],
        engine_config: Dict[str, Any],
        binding: Any = None,
    ) -> InferenceRunner:
        del binding
        if model_dir is None:
            raise ValueError("`model_dir` is required for engine='paddle_static'.")
        runner_config = _inject_trt_info(model_config, dict(engine_config))
        return PaddleStaticRunner(
            model_name=model_name,
            model_dir=model_dir,
            model_file_prefix=MODEL_FILE_PREFIX,
            config=runner_config,
        )


class PaddleDynamicEngineSpec(InferenceEngine):
    """Engine for Paddle dynamic-graph inference."""

    entities = "paddle_dynamic"

    BINDING_EXTRA_RUNNER_BUILDER_KEY = "runner_builder"

    @property
    def name(self) -> str:
        return "paddle_dynamic"

    @property
    def engine_config_model(self) -> Type[PaddleDynamicRunnerConfig]:
        return PaddleDynamicRunnerConfig

    def get_supported_model_formats(
        self,
    ) -> Optional[Tuple[LocalModelFormat, ...]]:
        return ("safetensors", "paddle_dyn")

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
        del device, engine_config
        if not is_dep_available("paddlepaddle"):
            raise RuntimeError(
                "Engine 'paddle_dynamic' is unavailable because dependency "
                "'paddlepaddle' is not installed."
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
        runner_builder = None
        if binding is not None:
            runner_builder = binding.extra_info.get(
                self.BINDING_EXTRA_RUNNER_BUILDER_KEY
            )
        if not callable(runner_builder):
            raise RuntimeError(
                f"Model {model_name!r} does not provide paddle_dynamic runner metadata."
            )
        return runner_builder(
            model_name,
            model_dir,
            model_config,
            engine_config,
        )
