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
    def set_batch_size(self, batch_size):
        self.components["ReadCmp"].batch_size = batch_size


class DeviceSetMixin:
    def set_device(self, device):
        self.pp_option.set_device(device)
        self.components["PPEngineCmp"].option = self.pp_option


class PPOptionSetMixin:
    def set_pp_option(self, pp_option):
        self.pp_option = pp_option
        self.components["PPEngineCmp"].option = self.pp_option
