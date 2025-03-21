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

from abc import ABC, abstractmethod

from .....utils.subclass_register import AutoRegisterABCMetaClass


class BaseOperator(ABC, metaclass=AutoRegisterABCMetaClass):
    """Base Operator"""

    __is_base = True

    def __init__(self):
        """Initializes an instance of base operator."""
        super().__init__()

    @abstractmethod
    def __call__(self):
        """
        Declaration of an abstract method. Subclasses are expected to
        provide a concrete implementation of call method.
        """
        raise NotImplementedError(
            "The component method `__call__` has not been implemented yet."
        )
