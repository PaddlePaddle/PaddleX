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

from .modules.base import build_dataset_checker, build_trainer, build_evaluater, build_predictor
from .utils.result_saver import try_except_decorator
from .utils import config
from .utils.errors import raise_unsupported_api_error


class Engine(object):
    """ Engine """

    def __init__(self):
        args = config.parse_args()
        self.config = config.get_config(
            args.config, overrides=args.override, show=False)
        self.mode = self.config.Global.mode
        self.output_dir = self.config.Global.output

    @try_except_decorator
    def run(self):
        """ the main function """
        if self.config.Global.mode == "check_dataset":
            dataset_checker = build_dataset_checker(self.config)
            return dataset_checker.check_dataset()
        elif self.config.Global.mode == "train":
            trainer = build_trainer(self.config)
            trainer.train()
        elif self.config.Global.mode == "evaluate":
            evaluator = build_evaluater(self.config)
            return evaluator.evaluate()
        elif self.config.Global.mode == "export":
            raise_unsupported_api_error("export", self.__class__)
        elif self.config.Global.mode == "predict":
            predictor = build_predictor(self.config)
            return predictor.predict()
        else:
            raise_unsupported_api_error(f"{self.config.Global.mode}",
                                        self.__class__)