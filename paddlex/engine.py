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

import os


from .utils.result_saver import try_except_decorator
from .utils.config import parse_args, get_config
from .utils.errors import raise_unsupported_api_error
from .model import _ModelBasedConfig


class Engine(object):
    """Engine"""

    def __init__(self):
        args = parse_args()
        config = get_config(args.config, overrides=args.override, show=False)
        self._mode = config.Global.mode
        self._output = config.Global.output
        self._model = _ModelBasedConfig(config)

    @try_except_decorator
    def run(self):
        """the main function"""
        if self._mode == "check_dataset":
            return self._model.check_dataset()
        elif self._mode == "train":
            self._model.train()
        elif self._mode == "evaluate":
            return self._model.evaluate()
        elif self._mode == "export":
            return self._model.export()
        elif self._mode == "predict":
            for res in self._model.predict():
                res.print(json_format=False)
        else:
            raise_unsupported_api_error(f"{self._mode}", self.__class__)
