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
from .....utils.device import constr_device
from ....utils.hpi import HPIConfig
from ...common.runner import PaddleDynamicRunner, PaddleStaticRunner
from .base_predictor import BasePredictor


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
        super().__init__(
            model_name=model_name or "",
            engine=engine,
            engine_config=engine_config,
            **kwargs,
        )
        self._model_dir = Path(model_dir) if model_dir else None
        self.config = model_config or {}
        self.model_name = model_name or self.config.get("Global", {}).get(
            "model_name", ""
        )
        self._engine_config = engine_config or {}

    @property
    def device(self) -> Optional[str]:
        """Device string from engine_config."""
        device_type = self._engine_config.get("device_type")
        if device_type:
            device_id = self._engine_config.get("device_id")
            device_ids = [device_id] if device_id is not None else None
            return constr_device(device_type, device_ids)
        return None

    @property
    def model_dir(self) -> Optional[Path]:
        """Model directory path."""
        return self._model_dir

    def _check_engine_support(self, engine: str) -> None:
        """Validate that the model supports the requested engine."""
        supported = tuple(e.lower() for e in self.__class__.get_supported_engines())
        if engine.lower() not in supported:
            raise ValueError(
                f"Model {self.__class__.__name__!r} does not support engine {engine!r}. "
                f"Supported engines: {list(supported)!r}."
            )

    def build_paddle_static_runner(self):
        """Build PaddleStaticRunner for engine=paddle_static."""
        model_file_prefix = getattr(
            self.__class__, "MODEL_FILE_PREFIX", constants.MODEL_FILE_PREFIX
        )
        return PaddleStaticRunner(
            model_name=self.model_name,
            model_dir=self._model_dir,
            model_file_prefix=model_file_prefix,
            config=self._engine_config,
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
