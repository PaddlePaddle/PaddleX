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


class BatchSizeSetMixin:
    @property
    def batch_size(self):
        return self.components["ReadCmp"].batch_size

    @batch_size.setter
    def batch_size(self, batch_size):
        self.components["ReadCmp"].batch_size = batch_size


class DeviceSetMixin:
    @property
    def device(self):
        return self._pp_option.device

    @device.setter
    def device(self, device):
        self._pp_option.device = device


class PPOptionSetMixin:
    @property
    def pp_option(self):
        return self._pp_option

    @pp_option.setter
    def pp_option(self, pp_option):
        self._pp_option = pp_option
