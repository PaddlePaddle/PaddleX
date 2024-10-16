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
import inspect

from ....utils.subclass_register import AutoRegisterABCMetaClass
from ....utils import logging
from ...components.base import BaseComponent, ComponentsEngine
from ...utils.pp_option import PaddlePredictorOption
from ...utils.process_hook import generatorable_method
from .base_predictor import BasePredictor


class BasicPredictor(
    BasePredictor,
    metaclass=AutoRegisterABCMetaClass,
):

    __is_base = True

    def __init__(self, model_dir, config=None, device=None, pp_option=None):
        super().__init__(model_dir=model_dir, config=config)
        if not pp_option:
            pp_option = PaddlePredictorOption(model_name=self.model_name)
        if device:
            pp_option.device = device
        self.pp_option = pp_option

        self.components = {}
        self._build_components()
        self.engine = ComponentsEngine(self.components)
        logging.debug(f"{self.__class__.__name__}: {self.model_dir}")

    def apply(self, input):
        """predict"""
        yield from self._generate_res(self.engine(input))

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

    def set_predictor(self, batch_size=None, device=None, pp_option=None):
        if batch_size:
            self.components["ReadCmp"].batch_size = batch_size
        if device and device != self.pp_option.device:
            self.pp_option.device = device
            self.components["PPEngineCmp"].reset()
        if pp_option and pp_option != self.pp_option:
            self.pp_option = pp_option
            self.components["PPEngineCmp"].reset()

    def _has_setter(self, attr):
        prop = getattr(self.__class__, attr, None)
        return isinstance(prop, property) and prop.fset is not None

    @abstractmethod
    def _build_components(self):
        raise NotImplementedError

    @abstractmethod
    def _pack_res(self, data):
        raise NotImplementedError
