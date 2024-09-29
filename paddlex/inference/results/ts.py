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

from .utils.mixin import JsonMixin, CSVMixin
from .base import BaseResult


class _BaseTSResult(BaseResult, CSVMixin):
    def __init__(self, data):
        super().__init__(data)
        CSVMixin.__init__(self)


class TSFcResult(_BaseTSResult):
    def _to_csv(self, save_path):
        return self["forecast"]


class TSClsResult(_BaseTSResult):
    def save_to_csv(self, save_path):
        return self["classification"]


class TSAdResult(_BaseTSResult):
    def save_to_csv(self, save_path):
        return self["anomaly"]
