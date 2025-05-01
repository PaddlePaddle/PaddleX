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

import numpy as np

from ....utils.deps import class_requires_deps, is_dep_available
from ...utils.benchmark import benchmark

if is_dep_available("scikit-image"):
    from skimage import morphology


@benchmark.timeit
@class_requires_deps("scikit-image")
class MapToMask:
    """Map_to_mask"""

    def __init__(self):
        """
        Initialize the instance.
        """
        super().__init__()

    def __call__(self, preds, *args):
        """apply"""
        return [self.apply(pred) for pred in preds]

    def apply(
        self,
        pred,
    ):
        """apply"""
        score_map = pred[0]
        thred = 0.01
        mask = score_map[0]
        mask[mask > thred] = 255
        mask[mask <= thred] = 0
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask = mask.astype(np.uint8)

        return mask[None, :, :]
