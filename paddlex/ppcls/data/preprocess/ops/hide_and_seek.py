# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

# This code is based on https://github.com/kkanshul/Hide-and-Seek

import numpy as np
import random


class HideAndSeek(object):
    def __init__(self):
        # possible grid size, 0 means no hiding
        self.grid_sizes = [0, 16, 32, 44, 56]
        # hiding probability
        self.hide_prob = 0.5

    def __call__(self, img):
        # randomly choose one grid size
        grid_size = np.random.choice(self.grid_sizes)

        _, h, w = img.shape

        # hide the patches
        if grid_size == 0:
            return img
        for x in range(0, w, grid_size):
            for y in range(0, h, grid_size):
                x_end = min(w, x + grid_size)
                y_end = min(h, y + grid_size)
                if (random.random() <= self.hide_prob):
                    img[:, x:x_end, y:y_end] = 0

        return img
