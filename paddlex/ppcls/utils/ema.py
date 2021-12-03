# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

import paddle
import numpy as np


class ExponentialMovingAverage():
    """
    Exponential Moving Average
    Code was heavily based on https://github.com/Wanger-SJTU/SegToolbox.Pytorch/blob/master/lib/utils/ema.py
    """

    def __init__(self, model, decay, thres_steps=True):
        self._model = model
        self._decay = decay
        self._thres_steps = thres_steps
        self._shadow = {}
        self._backup = {}

    def register(self):
        self._update_step = 0
        for name, param in self._model.named_parameters():
            if param.stop_gradient is False:
                self._shadow[name] = param.numpy().copy()

    def update(self):
        decay = min(self._decay, (1 + self._update_step) / (
            10 + self._update_step)) if self._thres_steps else self._decay
        for name, param in self._model.named_parameters():
            if param.stop_gradient is False:
                assert name in self._shadow
                new_val = np.array(param.numpy().copy())
                old_val = np.array(self._shadow[name])
                new_average = decay * old_val + (1 - decay) * new_val
                self._shadow[name] = new_average
        self._update_step += 1
        return decay

    def apply(self):
        for name, param in self._model.named_parameters():
            if param.stop_gradient is False:
                assert name in self._shadow
                self._backup[name] = np.array(param.numpy().copy())
                param.set_value(np.array(self._shadow[name]))

    def restore(self):
        for name, param in self._model.named_parameters():
            if param.stop_gradient is False:
                assert name in self._backup
                param.set_value(self._backup[name])
        self._backup = {}
