# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import abc
import logging

from ..model import BaseUltraInferModel
from ..runtime import Runtime, RuntimeOption

_logger = logging.getLogger(__name__)


class PyOnlyUltraInferModel(BaseUltraInferModel):
    def __init__(self, option):
        super().__init__()
        if option is None:
            self._option = RuntimeOption()
        else:
            self._option = option
        self._update_option()
        self._runtime = Runtime(self._option)
        _logger.debug("Python-only model initialized")

    def num_inputs_of_runtime(self):
        return self._runtime.num_inputs()

    def num_outputs_of_runtime(self):
        return self._runtime.num_outputs()

    def get_profile_time(self):
        return self._runtime.get_profile_time()

    def _update_option(self):
        pass


class PyOnlyProcessor(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, data):
        raise NotImplementedError


class PyOnlyProcessorChain(object):
    def __init__(self, processors):
        super().__init__()
        self._processors = processors

    def __call__(self, data):
        for processor in self._processors:
            data = processor(data)
        return data
