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


class BatchSizeMixin:
    NAME = "ReadCmp"

    def __init__(self, batch_size=1):
        self._batch_size = batch_size

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        if value <= 0:
            raise ValueError("Batch size must be positive.")
        self._batch_size = value


class PPEngineMixin:
    NAME = "PPEngineCmp"

    def __init__(self, option=None):
        self._option = option

    @property
    def option(self):
        return self._option

    @option.setter
    def option(self, value):
        if value != self.option:
            self._option = value
            self._reset()
