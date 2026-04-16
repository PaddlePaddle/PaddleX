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

"""Remote GenAI client engine."""

from typing import Any, Dict, Type

from ....utils.deps import is_genai_client_plugin_available
from ..common.genai import SERVER_BACKENDS, GenAIConfig
from ._base import InferenceEngine


class GenAIClientEngine(InferenceEngine):
    """Engine for remote GenAI client inference."""

    entities = "genai_client"

    @property
    def name(self) -> str:
        return "genai_client"

    @property
    def engine_config_model(self) -> Type[GenAIConfig]:
        return GenAIConfig

    @property
    def needs_local_model(self) -> bool:
        return False

    def post_normalize_config(self, validated: Dict[str, Any]) -> Dict[str, Any]:
        if validated.get("backend") not in SERVER_BACKENDS:
            raise ValueError(
                f"engine='genai_client' requires backend in {SERVER_BACKENDS!r}, "
                f"got {validated.get('backend')!r}."
            )
        return validated

    def ensure_environment(self) -> None:
        if not is_genai_client_plugin_available():
            raise RuntimeError("The genai client plugin is not available.")
