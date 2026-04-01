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

from ....constants import MODEL_FILE_PREFIX
from ....utils import logging
from ....utils.deps import is_dep_available
from ..hpi import get_hpi_info
from ..runners import PaddleDynamicRunner, PaddleStaticRunner
from ..runners.inference_runner import InferenceRunner
from ..runners.paddle_dynamic_runner import PaddleDynamicRunnerConfig
from ..runners.paddle_static import PaddleStaticRunnerConfig
from ..utils.model_paths import LocalModelFormat
from ._base import RunnerBuilder, RunnerEngine


def _inject_trt_info(
    model_config: Optional[Dict[str, Any]],
    engine_config: Dict[str, Any],
) -> Dict[str, Any]:
    hpi_info = get_hpi_info(model_config)
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


class PaddleStaticEngine(RunnerEngine):
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
        self._apply_device(raw, device)
        return raw

    def ensure_environment(self) -> None:
        if not is_dep_available("paddlepaddle"):
            raise RuntimeError(
                "Engine 'paddle_static' is unavailable because dependency "
                "'paddlepaddle' is not installed."
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
            del default_builder
            if model_dir is None:
                raise ValueError("`model_dir` is required for engine='paddle_static'.")
            runner_config = _inject_trt_info(model_config, dict(engine_config))
            return PaddleStaticRunner(
                model_name=model_name,
                model_dir=model_dir,
                model_file_prefix=MODEL_FILE_PREFIX,
                config=runner_config,
            )

        return runner_builder

    def validate_runner(self, runner: InferenceRunner) -> None:
        if not isinstance(runner, PaddleStaticRunner):
            raise TypeError(
                "Engine 'paddle_static' must build a PaddleStaticRunner, "
                f"but got {type(runner).__name__}."
            )


class PaddleDynamicEngine(RunnerEngine):
    """Engine for Paddle dynamic-graph inference."""

    entities = "paddle_dynamic"

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
        self._apply_device(raw, device)
        return raw

    def ensure_environment(self) -> None:
        if not is_dep_available("paddlepaddle"):
            raise RuntimeError(
                "Engine 'paddle_dynamic' is unavailable because dependency "
                "'paddlepaddle' is not installed."
            )

    def get_default_runner_builder(self) -> Optional[RunnerBuilder]:
        return None

    def validate_runner(self, runner: InferenceRunner) -> None:
        if not isinstance(runner, PaddleDynamicRunner):
            raise TypeError(
                "Engine 'paddle_dynamic' must build a PaddleDynamicRunner, "
                f"but got {type(runner).__name__}."
            )
