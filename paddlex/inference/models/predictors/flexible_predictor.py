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
from typing import Any, Dict, List, Optional

from ...common.batch_sampler import BaseBatchSampler
from .local_model_predictor import LocalModelPredictor


class FlexiblePredictor(LocalModelPredictor):
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
            model_dir=model_dir,
            model_config=model_config,
            model_name=model_name,
            engine_config=engine_config,
            **kwargs,
        )

    @property
    def supports_benchmark(self) -> bool:
        return False

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
