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

"""Engine spec for HPI."""

from pathlib import Path
from typing import Any, Dict, Optional, Type, Union

from pydantic import ValidationError

from ...constants import MODEL_FILE_PREFIX
from ...utils.deps import is_dep_available
from ...utils.device import get_default_device, parse_device
from ..base.predictor import BasePredictor, RunnerPredictor
from ..utils.hpi import HPIConfig
from ..utils.model_paths import get_model_paths
from ..utils.pp_option import PaddlePredictorOption
from ._base import EngineSpec


class HPIEngineSpec(EngineSpec):
    entities = "hpi"

    @property
    def name(self) -> str:
        return "hpi"

    def get_base_predictor_cls(self) -> Type[BasePredictor]:
        return RunnerPredictor

    def normalize_config(
        self,
        cfg: Optional[Union[Dict[str, Any], PaddlePredictorOption, Any]],
        *,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ) -> Dict[str, Any]:
        raw = self._engine_config_to_dict(cfg)
        try:
            raw.setdefault("model_name", model_name or "")
            if device:
                device_type, device_ids = parse_device(device)
                raw["device_type"] = device_type
                raw["device_id"] = device_ids[0] if device_ids is not None else None
            elif "device_type" not in raw:
                raw["device_type"], _ = parse_device(get_default_device())
            return HPIConfig.model_validate(raw).model_dump(
                exclude_none=True,
                by_alias=True,
            )
        except ValidationError as e:
            raise ValueError(f"Invalid hpi engine_config: {e}") from e

    def ensure_model_files(self, model_dir: Path) -> None:
        model_paths = get_model_paths(model_dir, MODEL_FILE_PREFIX)
        if not any(name in model_paths for name in ("paddle", "onnx", "om")):
            raise ValueError("No valid model files were found for engine 'hpi'.")

    def ensure_environment(
        self,
        *,
        device: Optional[str] = None,
        engine_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        del device, engine_config
        if not is_dep_available("ultra-infer"):
            raise RuntimeError(
                "Engine 'hpi' is unavailable because dependency "
                "'ultra-infer' is not installed."
            )
