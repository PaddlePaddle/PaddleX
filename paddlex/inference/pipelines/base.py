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
from contextvars import ContextVar, copy_context
from typing import TypedDict, Type

from ...utils.subclass_register import AutoRegisterABCMetaClass
from ..models import create_predictor

pipeline_info_list_var = ContextVar("pipeline_info_list", default=None)


class _PipelineInfo(TypedDict):
    cls: Type["BasePipeline"]


class _PipelineMetaClass(AutoRegisterABCMetaClass):
    def __new__(mcs, name, bases, attrs):
        def _patch_init_func(init_func):
            def _patched___init__(self, *args, **kwargs):
                ctx = copy_context()
                pipeline_info_list = [
                    *ctx.get(pipeline_info_list_var, []),
                    _PipelineInfo(cls=type(self)),
                ]
                ctx.run(pipeline_info_list_var.set, pipeline_info_list)
                ret = ctx.run(init_func, self, *args, **kwargs)
                return ret

            return _patched___init__

        cls = super().__new__(mcs, name, bases, attrs)
        cls.__init__ = _patch_init_func(cls.__init__)
        return cls


class BasePipeline(ABC, metaclass=_PipelineMetaClass):
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
