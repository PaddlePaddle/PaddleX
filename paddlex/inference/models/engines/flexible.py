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

"""Flexible engine."""

from typing import Any, Dict, Optional

from ....utils.device import parse_device
from ._base import InferenceEngine


class FlexibleEngineSpec(InferenceEngine):
    """Engine for predictors with custom local execution logic."""

    entities = "flexible"

    @property
    def name(self) -> str:
        return "flexible"

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
