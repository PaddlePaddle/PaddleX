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

"""Engine spec for transformers predictors."""

from typing import Any, Dict, Optional, Type, Union

from pydantic import ValidationError

from ...utils.device import parse_device
from ..base.predictor import BasePredictor, TransformersPredictor
from ..base.predictor.transformers_predictor import TransformersEngineConfig
from ..utils.pp_option import PaddlePredictorOption
from ._base import EngineSpec


class TransformersEngineSpec(EngineSpec):
    entities = "transformers"

    @property
    def name(self) -> str:
        return "transformers"

    def get_base_predictor_cls(self) -> Type[BasePredictor]:
        return TransformersPredictor

    def normalize_config(
        self,
        cfg: Optional[Union[Dict[str, Any], PaddlePredictorOption, Any]],
        *,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ) -> Dict[str, Any]:
        del model_name
        raw = self._engine_config_to_dict(cfg)
        if device:
            device_type, device_ids = parse_device(device)
            if device_type == "gpu":
                raw["device_map"] = f"cuda:{device_ids[0]}" if device_ids else "cuda"
            elif device_type == "cpu":
                raw["device_map"] = "cpu"
            else:
                raw["device_map"] = (
                    f"{device_type}:{device_ids[0]}" if device_ids else device_type
                )
        try:
            return TransformersEngineConfig.model_validate(raw).model_dump(
                exclude_none=True
            )
        except ValidationError as e:
            raise ValueError(f"Invalid transformers engine_config: {e}") from e
