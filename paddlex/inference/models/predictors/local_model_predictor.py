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

from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .base_predictor import BasePredictor


class LocalModelPredictor(BasePredictor):
    """Base class for predictors that use local model files."""

    __is_base = True

    def __init__(
        self,
        model_dir: Optional[str] = None,
        model_config: Optional[Dict] = None,
        model_name: Optional[str] = None,
        engine: str = "paddle_static",
        engine_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        self._model_dir, self._model_config, resolved_name = self.resolve_model_args(
            model_dir=model_dir,
            model_config=model_config,
            model_name=model_name,
        )
        # Keep `self.config` for backward compatibility with existing predictors.
        self.config = self._model_config
        super().__init__(
            model_name=resolved_name,
            engine=engine,
            engine_config=engine_config,
            **kwargs,
        )

    @property
    def model_dir(self) -> Optional[Path]:
        return self._model_dir

    @property
    def model_config(self) -> Dict[str, Any]:
        return self._model_config

    @staticmethod
    def resolve_model_args(
        model_dir: Optional[str],
        model_config: Optional[Dict],
        model_name: Optional[str],
    ) -> Tuple[Optional[Path], Dict[str, Any], str]:
        resolved_dir = Path(model_dir) if model_dir else None
        config = model_config or {}
        resolved_name = model_name or config.get("Global", {}).get("model_name", "")
        return resolved_dir, config, resolved_name

    @classmethod
    @abstractmethod
    def get_supported_engines(cls):
        raise NotImplementedError
