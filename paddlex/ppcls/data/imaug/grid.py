# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

# This code is based on https://github.com/akuxcw/GridMask

import numpy as np
from PIL import Image
import pdb

# curr
CURR_EPOCH = 0
# epoch for the prob to be the upper limit
NUM_EPOCHS = 240


class GridMask(object):
    def __init__(self, d1=96, d2=224, rotate=1, ratio=0.5, mode=0, prob=1.):
        self.d1 = d1
        self.d2 = d2
        self.rotate = rotate
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob
        self.last_prob = -1

    def set_prob(self):
        global CURR_EPOCH
        global NUM_EPOCHS
        self.prob = self.st_prob * min(1, 1.0 * CURR_EPOCH / NUM_EPOCHS)

    def __call__(self, img):
        self.set_prob()
        if abs(self.last_prob - self.prob) > 1e-10:
            global CURR_EPOCH
            global NUM_EPOCHS
            print(
                "self.prob is updated, self.prob={}, CURR_EPOCH: {}, NUM_EPOCHS: {}".
                format(self.prob, CURR_EPOCH, NUM_EPOCHS))
            self.last_prob = self.prob
        # print("CURR_EPOCH: {}, NUM_EPOCHS: {}, self.prob is set as: {}".format(CURR_EPOCH, NUM_EPOCHS, self.prob) )
        if np.random.rand() > self.prob:
            return img
        _, h, w = img.shape
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(self.d1, self.d2)
        #d = self.d
        self.l = int(d * self.ratio + 0.5)
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        for i in range(-1, hh // d + 1):
            s = d * i + st_h
            t = s + self.l
            s = max(min(s, hh), 0)
            t = max(min(t, hh), 0)
            mask[s:t, :] *= 0
        for i in range(-1, ww // d + 1):
            s = d * i + st_w
            t = s + self.l
            s = max(min(s, ww), 0)
            t = max(min(t, ww), 0)
            mask[:, s:t] *= 0
        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (ww - w) // 2:(ww - w) //
                    2 + w]

        if self.mode == 1:
            mask = 1 - mask

        mask = np.expand_dims(mask, axis=0)
        img = (img * mask).astype(img.dtype)

        return img
