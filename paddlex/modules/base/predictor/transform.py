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


import abc

from .utils.mixin import FromDictMixin
from .utils.batch import batchable_method
from .utils.node import Node


class BaseTransform(FromDictMixin, Node):
    """BaseTransform"""

    @batchable_method
    def __call__(self, data):
        self.check_input_keys(data)
        data = self.apply(data)
        self.check_output_keys(data)
        return data

    @abc.abstractmethod
    def apply(self, data):
        """apply"""
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__
