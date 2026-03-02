# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

"""FlexiblePredictor for models with flexible/customizable implementations (no Paddle/HPI runners)."""

from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from ....common.batch_sampler import BaseBatchSampler
from .base_predictor import BasePredictor


class FlexiblePredictor(BasePredictor):
    """Base class for predictors with flexible/customizable implementations."""

    __is_base = True

    def __init__(
        self,
        model_dir: Optional[str] = None,
        model_config: Optional[Dict] = None,
        model_name: Optional[str] = None,
        engine_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            model_name=model_name or "",
            engine="flexible",
            engine_config=engine_config,
            **kwargs,
        )
        self._model_dir = Path(model_dir) if model_dir else None
        self.config = model_config or {}
        self.model_name = model_name or self.config.get("Global", {}).get(
            "model_name", ""
        )

    @property
    def model_dir(self) -> Optional[Path]:
        """Model directory path."""
        return self._model_dir

    @abstractmethod
    def _build(self) -> Any:
        """Build the model. Subclasses implement their custom logic."""
        raise NotImplementedError

    @abstractmethod
    def process(self, batch_data: List[Any]) -> Dict[str, List[Any]]:
        raise NotImplementedError

    @abstractmethod
    def _build_batch_sampler(self) -> BaseBatchSampler:
        raise NotImplementedError

    @abstractmethod
    def _get_result_class(self) -> type:
        raise NotImplementedError
