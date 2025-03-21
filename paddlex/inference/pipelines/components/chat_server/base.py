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


class BaseChat(ABC, metaclass=AutoRegisterABCMetaClass):
    """Base class for all chat bots. This class serves as a foundation
    for creating various chat bots.
    """

    __is_base = True

    def __init__(self) -> None:
        """Initializes an instance of base chat."""
        super().__init__()

    @abstractmethod
    def generate_chat_results(self):
        """
        Declaration of an abstract method. Subclasses are expected to
        provide a concrete implementation of generate_chat_results.
        """
        raise NotImplementedError(
            "The method `generate_chat_results` has not been implemented yet."
        )
