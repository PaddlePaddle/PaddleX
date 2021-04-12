# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import cv2
try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence
from .operators import Transform, Resize


class BatchRandomResize(Transform):
    """
    Resize image to target size randomly. random target_size and interpolation method
    Args:
        target_size (int, list, tuple): image target size, if random size is True, must be list or tuple
        keep_ratio (bool): whether keep_raio or not, default true
        interp (int): the interpolation method
        random_size (bool): whether random select target size of image
    """

    def __init__(self,
                 target_size,
                 keep_ratio,
                 interp=cv2.INTER_NEAREST,
                 random_size=True):
        super(BatchRandomResize, self).__init__()
        self.keep_ratio = keep_ratio
        self.interp = interp
        assert isinstance(target_size, (
            int, Sequence)), "target_size must be int, list or tuple"
        if random_size and not isinstance(target_size, list):
            raise TypeError(
                "Type of target_size is invalid when random_size is True. Must be List, now is {}".
                format(type(target_size)))
        self.target_size = target_size
        self.random_size = random_size

    def __call__(self, samples):
        if self.random_size:
            height, width = np.random.choice(self.target_size)
        else:
            height, width = self.target_size

        resizer = Resize(
            height=height,
            width=width,
            keep_ratio=self.keep_ratio,
            interp=self.interp)

        return [resizer(sample) for sample in samples]
