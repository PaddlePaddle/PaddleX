# copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.
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
from typing import Any, Dict, Optional

from ...utils.subclass_register import AutoRegisterABCMetaClass
from ..models import create_predictor


class BasePipeline(ABC, metaclass=AutoRegisterABCMetaClass):
    """Base Pipeline"""

    __is_base = True

    def __init__(self, device, predictor_kwargs={}) -> None:
        super().__init__()
        self._predictor_kwargs = predictor_kwargs
        self._device = device

    @abstractmethod
    def set_predictor():
        raise NotImplementedError(
            "The method `set_predictor` has not been implemented yet."
        )

    # alias the __call__() to predict()
    def __call__(self, *args, **kwargs):
        yield from self.predict(*args, **kwargs)

    def _create(self, model=None, pipeline=None, *args, **kwargs):
        if model:
            return create_predictor(
                *args,
                model=model,
                device=self._device,
                **kwargs,
                **self._predictor_kwargs
            )
        elif pipeline:
            return pipeline(
                *args,
                device=self._device,
                predictor_kwargs=self._predictor_kwargs,
                **kwargs
            )
        else:
            raise Exception()
