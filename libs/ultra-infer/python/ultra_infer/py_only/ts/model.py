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

from ..base import PyOnlyUltraInferModel


class PyOnlyTSModel(PyOnlyUltraInferModel):
    @abc.abstractmethod
    def batch_predict(self, ts_list):
        raise NotImplementedError

    def predict(self, ts):
        return self.batch_predict([ts])[0]
