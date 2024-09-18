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

import yaml
import codecs
from pathlib import Path
from abc import abstractmethod

from ...utils.subclass_register import AutoRegisterABCMetaClass
from ...utils import logging
from ..components.base import BaseComponent, ComponentsEngine
from ..components.paddle_predictor.option import PaddlePredictorOption
from ..utils.process_hook import generatorable_method


class BasePredictor(BaseComponent, metaclass=AutoRegisterABCMetaClass):
    __is_base = True

    INPUT_KEYS = "x"
    DEAULT_INPUTS = {"x": "x"}
    OUTPUT_KEYS = "result"
    DEAULT_OUTPUTS = {"result": "result"}

    KEEP_INPUT = False

    MODEL_FILE_PREFIX = "inference"

    def __init__(self, model_dir, config=None, device=None, pp_option=None, **kwargs):
        super().__init__()
        self.model_dir = Path(model_dir)
        self.config = config if config else self.load_config(self.model_dir)
        self.kwargs = self._check_args(kwargs)

        self.pp_option = PaddlePredictorOption() if pp_option is None else pp_option
        if device is not None:
            self.pp_option.set_device(device)

        self.components = self._build_components()
        self.engine = ComponentsEngine(self.components)

        # alias predict() to the __call__()
        self.predict = self.__call__

        logging.debug(
            f"-------------------- {self.__class__.__name__} --------------------\nModel: {self.model_dir}\nEnv: {self.pp_option}"
        )

    @classmethod
    def load_config(cls, model_dir):
        config_path = model_dir / f"{cls.MODEL_FILE_PREFIX}.yml"
        with codecs.open(config_path, "r", "utf-8") as file:
            dic = yaml.load(file, Loader=yaml.FullLoader)
        return dic

    def apply(self, x):
        """predict"""
        yield from self._generate_res(self.engine(x))

    @generatorable_method
    def _generate_res(self, data):
        return self._pack_res(data)

    def _check_args(self, kwargs):
        return kwargs

    @abstractmethod
    def _build_components(self):
        raise NotImplementedError

    @abstractmethod
    def _pack_res(self, data):
        raise NotImplementedError
