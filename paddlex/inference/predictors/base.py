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

from ...utils.misc import AutoRegisterABCMetaClass
from ..components.base import BaseComponent, ComponentsEngine
from .official_models import official_models


class BasePredictor(BaseComponent, metaclass=AutoRegisterABCMetaClass):
    __is_base = True

    INPUT_KEYS = "x"
    OUTPUT_KEYS = None

    KEEP_INPUT = False

    MODEL_FILE_PREFIX = "inference"

    def __init__(self, model, **kwargs):
        super().__init__()
        self.model_dir = self._check_model(model)
        self.kwargs = kwargs
        self.config = self._load_config()
        self.components = self._build_components()
        self.engine = ComponentsEngine(self.components)
        # alias predict() to the __call__()
        self.predict = self.__call__

    def _check_model(self, model):
        if Path(model).exists():
            return Path(model)
        elif model in official_models:
            return official_models[model]
        else:
            raise Exception(
                f"The model ({model}) is no exists! Please using directory of local model files or model name supported by PaddleX!"
            )

    def _load_config(self):
        config_path = self.model_dir / f"{self.MODEL_FILE_PREFIX}.yml"
        with codecs.open(config_path, "r", "utf-8") as file:
            dic = yaml.load(file, Loader=yaml.FullLoader)
        return dic

    def apply(self, x):
        """predict"""
        yield from self.engine(x)

    @abstractmethod
    def _build_components(self):
        raise NotImplementedError
