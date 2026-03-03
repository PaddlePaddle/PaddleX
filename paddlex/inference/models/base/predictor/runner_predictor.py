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

from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ..... import constants
from .....utils import logging
from .....utils.device import constr_device
from ....utils.hpi import HPIConfig, HPIInfo
from ...common.runner import PaddleDynamicRunner, PaddleStaticRunner
from .base_predictor import BasePredictor
from .utils import resolve_model_args


class PredictionWrap:
    """Wrapper for prediction results with batch size info."""

    def __init__(self, prediction: Any, batch_size: int):
        self.prediction = prediction
        self.batch_size = batch_size


class RunnerPredictor(BasePredictor):
    """Base class for predictors that use inference runners (Paddle/HPI)."""

    __is_base = True

    @classmethod
    @abstractmethod
    def get_supported_engines(cls) -> Tuple[str, ...]:
        """Return the engines this predictor supports. Must be overridden by subclasses."""
        raise NotImplementedError

    def __init__(
        self,
        model_dir: Optional[str] = None,
        model_config: Optional[Dict] = None,
        model_name: Optional[str] = None,
        engine: str = "paddle_static",
        engine_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        self._check_engine_support(engine)
        self._model_dir, self.config, resolved_name = resolve_model_args(
            model_dir, model_config, model_name
        )
        super().__init__(
            model_name=resolved_name,
            engine=engine,
            engine_config=engine_config,
            **kwargs,
        )
        self._engine_config = engine_config or {}

    @property
    def model_dir(self) -> Optional[Path]:
        return self._model_dir

    @property
    def device(self) -> Optional[str]:
        """Device string from engine_config."""
        device_type = self._engine_config.get("device_type")
        if device_type:
            device_id = self._engine_config.get("device_id")
            device_ids = [device_id] if device_id is not None else None
            return constr_device(device_type, device_ids)
        return None

    def _check_engine_support(self, engine: str) -> None:
        """Validate that the model supports the requested engine."""
        supported = tuple(e.lower() for e in self.__class__.get_supported_engines())
        if engine.lower() not in supported:
            raise ValueError(
                f"Model {self.__class__.__name__!r} does not support engine {engine!r}. "
                f"Supported engines: {list(supported)!r}."
            )

    def _get_hpi_info(self):
        """Read HPI info from model config if available."""
        if not self.config or "Hpi" not in self.config:
            return None
        from pydantic import ValidationError

        try:
            return HPIInfo.model_validate(self.config["Hpi"])
        except ValidationError as e:
            raise RuntimeError(f"Invalid HPI info: {str(e)}") from e

    def _inject_trt_info(self, engine_config: Dict[str, Any]) -> Dict[str, Any]:
        """Inject TRT dynamic shape info from HPI config into engine_config if missing."""
        hpi_info = self._get_hpi_info()
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

    def build_paddle_static_runner(self):
        """Build PaddleStaticRunner for engine=paddle_static."""
        model_file_prefix = getattr(
            self.__class__, "MODEL_FILE_PREFIX", constants.MODEL_FILE_PREFIX
        )
        config = self._inject_trt_info(self._engine_config)
        return PaddleStaticRunner(
            model_name=self.model_name,
            model_dir=self._model_dir,
            model_file_prefix=model_file_prefix,
            config=config,
        )

    def build_paddle_dynamic_runner(self) -> PaddleDynamicRunner:
        """Build PaddleDynamicRunner for PaddlePaddle dynamic graph inference. Override in subclasses."""
        raise NotImplementedError(
            f"{self.__class__.__name__!r} does not support engine='paddle_dynamic'. "
            "Override build_paddle_dynamic_runner() to support dynamic graph inference."
        )

    def build_hpi_runner(self):
        """Build HPIRunner for engine=hpi."""
        from ...common.runner import HPIRunner

        model_file_prefix = getattr(
            self.__class__, "MODEL_FILE_PREFIX", constants.MODEL_FILE_PREFIX
        )
        hpi_cfg = dict(self._engine_config)
        hpi_cfg.setdefault("model_name", self.model_name)
        if "hpi_info" not in hpi_cfg:
            hpi_info = self._get_hpi_info()
            if hpi_info is not None:
                hpi_cfg["hpi_info"] = hpi_info
        hpi_config = HPIConfig.model_validate(hpi_cfg)
        return HPIRunner(
            model_dir=self._model_dir,
            model_file_prefix=model_file_prefix,
            config=hpi_config,
        )

    def create_runner(self):
        """Create the appropriate runner based on engine."""
        if self._engine == "paddle_static":
            return self.build_paddle_static_runner()
        if self._engine == "paddle_dynamic":
            return self.build_paddle_dynamic_runner()
        if self._engine == "hpi":
            return self.build_hpi_runner()
        raise RuntimeError(
            f"create_runner: no suitable runner for engine={self._engine!r}."
        )
