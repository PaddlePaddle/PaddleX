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

"""Engine spec for remote genai client inference."""

from typing import Any, Dict, Optional, Type, Union

from pydantic import ValidationError

from ...utils.deps import is_genai_client_plugin_available
from ..base.predictor import BasePredictor, GenAIClientPredictor
from ..common.genai import SERVER_BACKENDS, GenAIConfig
from ..utils.pp_option import PaddlePredictorOption
from ._base import EngineSpec


class GenAIClientEngineSpec(EngineSpec):
    entities = "genai_client"

    @property
    def name(self) -> str:
        return "genai_client"

    @property
    def needs_local_model(self) -> bool:
        return False

    def get_base_predictor_cls(self) -> Type[BasePredictor]:
        return GenAIClientPredictor

    def normalize_config(
        self,
        cfg: Optional[Union[Dict[str, Any], PaddlePredictorOption, Any]],
        *,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ) -> Dict[str, Any]:
        del model_name, device
        raw = self._engine_config_to_dict(cfg)
        try:
            validated = GenAIConfig.model_validate(raw).model_dump(exclude_none=True)
            if validated.get("backend") not in SERVER_BACKENDS:
                raise ValueError(
                    f"engine='genai_client' requires backend in {SERVER_BACKENDS!r}, "
                    f"got {validated.get('backend')!r}."
                )
            return validated
        except ValidationError as e:
            raise ValueError(f"Invalid genai_client engine_config: {e}") from e

    def ensure_environment(
        self,
        *,
        device: Optional[str] = None,
        engine_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        del device, engine_config
        if not is_genai_client_plugin_available():
            raise RuntimeError("The genai client plugin is not available.")
