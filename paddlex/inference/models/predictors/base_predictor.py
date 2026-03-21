# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from ....utils.subclass_register import AutoRegisterABCMetaClass
from ...common.batch_sampler import BaseBatchSampler


class BasePredictor(ABC, metaclass=AutoRegisterABCMetaClass):
    """Abstract predictor interface."""

    def __init__(
        self,
        *,
        model_name: str = "",
        engine_config: Optional[Dict[str, Any]] = None,
        batch_size: int = 1,
        **kwargs: Any,
    ) -> None:
        del kwargs
        self.model_name = model_name
        self._engine_config = dict(engine_config or {})

        self.batch_sampler = self._build_batch_sampler()
        self.result_class = self._get_result_class()

        # alias predict() to the __call__()
        self.predict = self.__call__

        self.batch_sampler.batch_size = batch_size

    @property
    def engine_config(self) -> Dict[str, Any]:
        return dict(self._engine_config)

    def __call__(
        self,
        input: Any,
        batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        """Default: delegate to apply."""
        yield from self.apply(input, **kwargs)

    def apply(self, input: Any, **kwargs: Any) -> Iterator[Any]:
        """Default implementation: batch_sampler -> process -> wrap with result_class.

        Handles two process return formats:
        1. pred["result"] is a list of per-item results
        2. pred is a dict of lists (e.g. input_path, class_ids, scores) - split by index
        """
        for batch_data in self.batch_sampler(input):
            if hasattr(batch_data, "instances"):
                input_paths = getattr(batch_data, "input_paths", None)
            else:
                input_paths = None
            pred = self.process(batch_data, **kwargs)
            results = pred.get("result", pred)
            if isinstance(results, list):
                for idx, single in enumerate(results):
                    item = {"result": single}
                    if input_paths and idx < len(input_paths):
                        item["input_path"] = input_paths[idx]
                    yield self.result_class(item)
            else:
                first_val = next(iter(pred.values()), None)
                n = len(first_val) if isinstance(first_val, list) else 1
                for idx in range(n):
                    item = {}
                    for k, v in pred.items():
                        if isinstance(v, list) and idx < len(v):
                            item[k] = v[idx]
                        else:
                            item[k] = v
                    if input_paths and idx < len(input_paths):
                        item["input_path"] = input_paths[idx]
                    yield self.result_class(item)

    @abstractmethod
    def process(self, batch_data: List[Any]) -> Dict[str, List[Any]]:
        raise NotImplementedError

    @abstractmethod
    def _build_batch_sampler(self) -> BaseBatchSampler:
        raise NotImplementedError

    @abstractmethod
    def _get_result_class(self) -> type:
        raise NotImplementedError

    def close(self) -> None:
        pass

    @classmethod
    def get_config_path(cls, model_dir: Path) -> Path:
        from ..utils.model_config import get_model_config_path

        return get_model_config_path(model_dir)

    @classmethod
    def load_config(cls, model_dir: Path) -> Dict:
        from ..utils.model_config import load_model_config

        return load_model_config(model_dir)
