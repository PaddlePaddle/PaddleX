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

import inspect
from abc import ABC, abstractmethod
from .....utils.subclass_register import AutoRegisterABCMetaClass


class BaseGeneratePrompt(ABC, metaclass=AutoRegisterABCMetaClass):
    """Base Generate Prompt class."""

    __is_base = True

    def __init__(self):
        """Initializes an instance of base generate prompt."""
        super().__init__()

    @abstractmethod
    def generate_prompt(self):
        """Declaration of an abstract method. Subclasses are expected to
        provide a concrete implementation of generate prompt method."""
        raise NotImplementedError(
            "The method `generate_prompt` has not been implemented yet."
        )
