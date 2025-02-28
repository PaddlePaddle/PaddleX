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

import numpy as np

from ...utils.benchmark import benchmark


@benchmark.timeit
class NormalizeFeatures:
    """Normalize Features Transform"""

    def _normalize(self, preds):
        """normalize"""
        feas_norm = np.sqrt(np.sum(np.square(preds[0]), axis=0, keepdims=True))
        features = np.divide(preds[0], feas_norm)
        return features

    def __call__(self, preds):
        normalized_features = [self._normalize(feature) for feature in preds]
        return normalized_features
