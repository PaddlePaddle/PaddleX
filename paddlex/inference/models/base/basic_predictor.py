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

from abc import abstractmethod

from ....utils.subclass_register import AutoRegisterABCMetaClass
from ....utils.func_register import FuncRegister
from ....utils import logging
from ...components.base import BaseComponent, ComponentsEngine
from ...utils.pp_option import PaddlePredictorOption
from ...utils.process_hook import generatorable_method
from ..utils.predict_set import DeviceSetMixin, PPOptionSetMixin, BatchSizeSetMixin
from .base_predictor import BasePredictor


class BasicPredictor(
    BasePredictor,
    DeviceSetMixin,
    PPOptionSetMixin,
    BatchSizeSetMixin,
    metaclass=AutoRegisterABCMetaClass,
):

    __is_base = True

    def __init__(
        self, model_dir, config=None, device=None, pp_option=None, **option_kwargs
    ):
        super().__init__(model_dir=model_dir, config=config)
        self._pred_set_func_map = {}
        self._pred_set_register = FuncRegister(self._pred_set_func_map)
        self._pred_set_register("device")(self.set_device)
        self._pred_set_register("pp_option")(self.set_pp_option)
        self._pred_set_register("batch_size")(self.set_batch_size)

        self.pp_option = (
            pp_option
            if pp_option
            else PaddlePredictorOption(model_name=self.model_name, **option_kwargs)
        )
        self.pp_option.set_device(device)
        self.components = {}
        self._build_components()
        self.engine = ComponentsEngine(self.components)
        logging.debug(f"{self.__class__.__name__}: {self.model_dir}")

    def apply(self, x):
        """predict"""
        yield from self._generate_res(self.engine(x))

    @generatorable_method
    def _generate_res(self, batch_data):
        return [{"result": self._pack_res(data)} for data in batch_data]

    def _add_component(self, cmps):
        if not isinstance(cmps, list):
            cmps = [cmps]

        for cmp in cmps:
            if not isinstance(cmp, (list, tuple)):
                key = cmp.name
            else:
                assert len(cmp) == 2
                key = cmp[0]
                cmp = cmp[1]
            assert isinstance(key, str)
            assert isinstance(cmp, BaseComponent)
            assert (
                key not in self.components
            ), f"The key ({key}) has been used: {self.components}!"
            self.components[key] = cmp

    def set_predictor(self, **kwargs):
        for k in kwargs:
            assert (
                k in self._pred_set_func_map
            ), f"The arg({k}) is not supported to specify in predict() func! Only supports: {self._pred_set_func_map.keys()}"
            self._pred_set_func_map[k](kwargs[k])

    @abstractmethod
    def _build_components(self):
        raise NotImplementedError

    @abstractmethod
    def _pack_res(self, data):
        raise NotImplementedError
