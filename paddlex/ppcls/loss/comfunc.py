#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def rerange_index(batch_size, samples_each_class):
    tmp = np.arange(0, batch_size * batch_size)
    tmp = tmp.reshape(-1, batch_size)
    rerange_index = []

    for i in range(batch_size):
        step = i // samples_each_class
        start = step * samples_each_class
        end = (step + 1) * samples_each_class

        pos_idx = []
        neg_idx = []
        for j, k in enumerate(tmp[i]):
            if j >= start and j < end:
                if j == i:
                    pos_idx.insert(0, k)
                else:
                    pos_idx.append(k)
            else:
                neg_idx.append(k)
        rerange_index += (pos_idx + neg_idx)

    rerange_index = np.array(rerange_index).astype(np.int32)
    return rerange_index
