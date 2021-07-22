# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import collections
import numpy as np
import datetime


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window.
    """

    def __init__(self, window_size=20):
        if window_size is None:
            self.deque = list()
        else:
            self.deque = collections.deque(maxlen=window_size)

    def update(self, value):
        self.deque.append(value)

    def avg(self):
        return np.mean(self.deque)


class TrainingStats(object):
    def __init__(self, window_size=None, delimiter=', '):
        self.meters = None
        self.window_size = window_size
        self.delimiter = delimiter

    def update(self, stats):
        if self.meters is None:
            self.meters = {
                k: SmoothedValue(self.window_size)
                for k in stats.keys()
            }
        for k, v in self.meters.items():
            v.update(stats[k].numpy())

    def get(self, extras=None):
        stats = collections.OrderedDict()
        if extras:
            for k, v in extras.items():
                stats[k] = v
        for k, v in self.meters.items():
            stats[k] = v.avg()

        return stats

    def log(self, extras=None):
        d = self.get(extras)
        strs = []
        for k, v in d.items():
            strs.append("{}={}".format(k, str(v).format('8.6f')))
        return self.delimiter.join(strs)
