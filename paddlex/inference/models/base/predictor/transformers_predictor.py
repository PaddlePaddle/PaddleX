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
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict

from .base_predictor import BasePredictor


class TransformersEngineConfig(BaseModel):
    """Engine config for transformers inference."""

    model_config = ConfigDict(extra="forbid")

    dtype: Optional[str] = None
    device_map: Optional[Union[str, Dict[str, Any]]] = None
    trust_remote_code: Optional[bool] = None
    attn_implementation: Optional[str] = None
    generation_config: Optional[Dict[str, Any]] = None
    model_kwargs: Optional[Dict[str, Any]] = None
    tokenizer_kwargs: Optional[Dict[str, Any]] = None


class TransformersPredictor(BasePredictor):
    """Base class for transformers-engine predictors."""

    __is_base = True

    def __init__(
        self,
        model_name: str = "",
        engine_config: Optional[Dict[str, Any]] = None,
        batch_size: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(
            model_name=model_name,
            engine="transformers",
            engine_config=engine_config,
            batch_size=batch_size,
            **kwargs,
        )

    @abstractmethod
    def process(self, batch_data: List[Any]) -> Dict[str, List[Any]]:
        raise NotImplementedError

    @abstractmethod
    def _build_batch_sampler(self):
        raise NotImplementedError

    @abstractmethod
    def _get_result_class(self) -> type:
        raise NotImplementedError
