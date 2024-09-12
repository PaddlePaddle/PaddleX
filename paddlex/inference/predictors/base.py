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

import GPUtil

from ...utils.subclass_register import AutoRegisterABCMetaClass
from ..utils.device import constr_device
from ..components.base import BaseComponent, ComponentsEngine
from .official_models import official_models


def _get_default_device():
    avail_gpus = GPUtil.getAvailable()
    if not avail_gpus:
        return "cpu"
    else:
        return constr_device("gpu", avail_gpus[0])


class BasePredictor(BaseComponent):
    INPUT_KEYS = "x"

    MODEL_FILE_PREFIX = "inference"

    def __init__(self, model, device=None, **kwargs):
        super().__init__()
        self.model_dir = self._check_model(model)
        self.device = device if device else _get_default_device()
        self.kwargs = kwargs
        self.config = self._load_config()
        # alias predict() to the __call__()
        self.predict = self.__call__

    @property
    def config_path(self):
        return self.model_dir / f"{self.MODEL_FILE_PREFIX}.yml"

    @abstractmethod
    def apply(self, x):
        raise NotImplementedError

    def _check_model(self, model):
        if Path(model).exists():
            return Path(model)
        elif model in official_models:
            return official_models[model]
        else:
            raise ValueError(
                f"The model ({model}) does not exist! Please use a local model directory or a model name supported by PaddleX!"
            )

    def _load_config(self):
        with codecs.open(self.config_path, "r", "utf-8") as file:
            dic = yaml.load(file, Loader=yaml.FullLoader)
        return dic


class BasicPredictor(BasePredictor, metaclass=AutoRegisterABCMetaClass):
    __is_base = True

    OUTPUT_KEYS = None

    KEEP_INPUT = False

    def __init__(self, model, device=None, **kwargs):
        super().__init__(model=model, device=device, **kwargs)
        self.components = self._build_components()
        self.engine = ComponentsEngine(self.components)

    def apply(self, x):
        """predict"""
        yield from self.engine(x)

    @abstractmethod
    def _build_components(self):
        raise NotImplementedError
