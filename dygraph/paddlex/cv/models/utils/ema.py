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

import paddle


class ExponentialMovingAverage(object):
    def __init__(self, decay, model, use_thres_step=False):
        self.step = 0
        self.decay = decay
        self.shadow = dict()
        for k, v in model.state_dict().items():
            self.shadow[k] = paddle.zeros_like(v)
        self.use_thres_step = use_thres_step

    def update(self, model):
        if self.use_thres_step:
            decay = min(self.decay, (1 + self.step) / (10 + self.step))
        else:
            decay = self.decay
        self._decay = decay
        model_dict = model.state_dict()
        for k, v in self.shadow.items():
            v = decay * v + (1 - decay) * model_dict[k]
            v.stop_gradient = True
            self.shadow[k] = v
        self.step += 1

    def apply(self):
        if self.step == 0:
            return self.shadow
        state_dict = dict()
        for k, v in self.shadow.items():
            v = v / (1 - self._decay**self.step)
            v.stop_gradient = True
            state_dict[k] = v
        return state_dict
