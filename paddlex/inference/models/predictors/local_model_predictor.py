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

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ....utils.device import constr_device
from ..utils.model_resolver import resolve_model_name
from .base_predictor import BasePredictor


class LocalModelPredictor(BasePredictor):
    """Base class for predictors that use local model files."""

    __is_base = True

    def __init__(
        self,
        model_dir: Optional[str] = None,
        model_config: Optional[Dict] = None,
        model_name: Optional[str] = None,
        engine_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        self._model_dir, self._model_config, resolved_name = self.resolve_model_args(
            model_dir=model_dir,
            model_config=model_config,
            model_name=model_name,
        )
        super().__init__(
            model_name=resolved_name,
            engine_config=engine_config,
            **kwargs,
        )

    @property
    def config(self) -> Dict[str, Any]:
        """Alias for model_config; kept for backward compatibility."""
        return self._model_config

    @property
    def model_dir(self) -> Optional[Path]:
        return self._model_dir

    @property
    def model_config(self) -> Dict[str, Any]:
        return self._model_config

    @property
    def device(self) -> Optional[str]:
        """Device string resolved from engine_config."""
        return self.resolve_device(self.engine_config)

    @staticmethod
    def resolve_model_args(
        model_dir: Optional[str],
        model_config: Optional[Dict],
        model_name: Optional[str],
    ) -> Tuple[Optional[Path], Dict[str, Any], str]:
        resolved_name, resolved_dir, resolved_config = resolve_model_name(
            model_name=model_name,
            model_dir=model_dir,
            model_config=model_config,
        )
        return resolved_dir, resolved_config, resolved_name

    @staticmethod
    def resolve_device(engine_config: Dict[str, Any]) -> Optional[str]:
        device_type = engine_config.get("device_type")
        if not device_type:
            return None
        device_id = engine_config.get("device_id")
        device_ids = [device_id] if device_id is not None else None
        return constr_device(device_type, device_ids)
