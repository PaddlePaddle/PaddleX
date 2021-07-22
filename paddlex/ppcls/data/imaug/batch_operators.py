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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from .fmix import sample_mask


class BatchOperator(object):
    """ BatchOperator """

    def __init__(self, *args, **kwargs):
        pass

    def _unpack(self, batch):
        """ _unpack """
        assert isinstance(batch, list), \
                'batch should be a list filled with tuples (img, label)'
        bs = len(batch)
        assert bs > 0, 'size of the batch data should > 0'
        imgs, labels = list(zip(*batch))
        return np.array(imgs), np.array(labels), bs

    def __call__(self, batch):
        return batch


class MixupOperator(BatchOperator):
    """ Mixup operator """

    def __init__(self, alpha=0.2):
        assert alpha > 0., \
                'parameter alpha[%f] should > 0.0' % (alpha)
        self._alpha = alpha

    def __call__(self, batch):
        imgs, labels, bs = self._unpack(batch)
        idx = np.random.permutation(bs)
        lam = np.random.beta(self._alpha, self._alpha)
        lams = np.array([lam] * bs, dtype=np.float32)
        imgs = lam * imgs + (1 - lam) * imgs[idx]
        return list(zip(imgs, labels, labels[idx], lams))


class CutmixOperator(BatchOperator):
    """ Cutmix operator """

    def __init__(self, alpha=0.2):
        assert alpha > 0., \
                'parameter alpha[%f] should > 0.0' % (alpha)
        self._alpha = alpha

    def _rand_bbox(self, size, lam):
        """ _rand_bbox """
        w = size[2]
        h = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(w * cut_rat)
        cut_h = np.int(h * cut_rat)

        # uniform
        cx = np.random.randint(w)
        cy = np.random.randint(h)

        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)

        return bbx1, bby1, bbx2, bby2

    def __call__(self, batch):
        imgs, labels, bs = self._unpack(batch)
        idx = np.random.permutation(bs)
        lam = np.random.beta(self._alpha, self._alpha)

        bbx1, bby1, bbx2, bby2 = self._rand_bbox(imgs.shape, lam)
        imgs[:, :, bbx1:bbx2, bby1:bby2] = imgs[idx, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - (float(bbx2 - bbx1) * (bby2 - bby1) /
                   (imgs.shape[-2] * imgs.shape[-1]))
        lams = np.array([lam] * bs, dtype=np.float32)
        return list(zip(imgs, labels, labels[idx], lams))


class FmixOperator(BatchOperator):
    """ Fmix operator """

    def __init__(self, alpha=1, decay_power=3, max_soft=0., reformulate=False):
        self._alpha = alpha
        self._decay_power = decay_power
        self._max_soft = max_soft
        self._reformulate = reformulate

    def __call__(self, batch):
        imgs, labels, bs = self._unpack(batch)
        idx = np.random.permutation(bs)
        size = (imgs.shape[2], imgs.shape[3])
        lam, mask = sample_mask(self._alpha, self._decay_power, \
                size, self._max_soft, self._reformulate)
        imgs = mask * imgs + (1 - mask) * imgs[idx]
        return list(zip(imgs, labels, labels[idx], [lam] * bs))
