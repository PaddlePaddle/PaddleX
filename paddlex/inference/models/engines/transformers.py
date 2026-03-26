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

"""Transformers engine."""

from typing import Any, Dict, Optional, Tuple, Type

from pydantic import BaseModel, ConfigDict

from ....utils.deps import is_dep_available
from ..utils.model_paths import LocalModelFormat
from ._base import InferenceEngine


class TransformersEngineConfig(BaseModel):
    """Engine config for transformers inference."""

    model_config = ConfigDict(extra="forbid")

    dtype: Optional[str] = None
    device_type: Optional[str] = None
    device_id: Optional[int] = None
    trust_remote_code: Optional[bool] = None
    attn_implementation: Optional[str] = None
    generation_config: Optional[Dict[str, Any]] = None
    model_kwargs: Optional[Dict[str, Any]] = None
    processor_kwargs: Optional[Dict[str, Any]] = None
    tokenizer_kwargs: Optional[Dict[str, Any]] = None


class TransformersEngine(InferenceEngine):
    """Engine for Hugging Face Transformers inference."""

    entities = "transformers"

    @property
    def name(self) -> str:
        return "transformers"

    @property
    def engine_config_model(self) -> Type[TransformersEngineConfig]:
        return TransformersEngineConfig

    def get_supported_model_formats(
        self,
    ) -> Optional[Tuple[LocalModelFormat, ...]]:
        return ("safetensors",)

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
        if not is_dep_available("transformers"):
            raise RuntimeError(
                "Engine 'transformers' is unavailable because dependency "
                "'transformers' is not installed."
            )
