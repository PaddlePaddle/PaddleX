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

from ...components.base import BaseComponent
from ...utils.pp_option import PaddlePredictorOption
from ...utils.process_hook import generatorable_method
from ..utils.predict_set import DeviceSetMixin, PPOptionSetMixin, BatchSizeSetMixin


class BasePredictor(BaseComponent):

    KEEP_INPUT = False
    YIELD_BATCH = False

    INPUT_KEYS = "x"
    DEAULT_INPUTS = {"x": "x"}
    OUTPUT_KEYS = "result"
    DEAULT_OUTPUTS = {"result": "result"}

    MODEL_FILE_PREFIX = "inference"

    def __init__(self, model_dir, config=None):
        super().__init__()
        self.model_dir = Path(model_dir)
        self.config = config if config else self.load_config(self.model_dir)

        # alias predict() to the __call__()
        self.predict = self.__call__

    def __call__(self, input, **kwargs):
        self.set_predictor(**kwargs)
        for res in super().__call__(input):
            yield res["result"]

    @property
    def config_path(self):
        return self.get_config_path(self.model_dir)

    @property
    def model_name(self) -> str:
        return self.config["Global"]["model_name"]

    @abstractmethod
    def apply(self, x):
        raise NotImplementedError

    @abstractmethod
    def set_predictor(self):
        raise NotImplementedError

    @classmethod
    def get_config_path(cls, model_dir):
        return model_dir / f"{cls.MODEL_FILE_PREFIX}.yml"

    @classmethod
    def load_config(cls, model_dir):
        config_path = cls.get_config_path(model_dir)
        with codecs.open(config_path, "r", "utf-8") as file:
            dic = yaml.load(file, Loader=yaml.FullLoader)
        return dic
