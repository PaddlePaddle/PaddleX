# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import math

from paddle.optimizer.lr import LinearWarmup
from paddle.optimizer.lr import PiecewiseDecay
from paddle.optimizer.lr import CosineAnnealingDecay
from paddle.optimizer.lr import ExponentialDecay

__all__ = ['LearningRateBuilder']


class Cosine(CosineAnnealingDecay):
    """
    Cosine learning rate decay
    lr = 0.05 * (math.cos(epoch * (math.pi / epochs)) + 1)

    Args:
        lr(float): initial learning rate
        step_each_epoch(int): steps each epoch
        epochs(int): total training epochs
    """

    def __init__(self, lr, step_each_epoch, epochs, **kwargs):
        super(Cosine, self).__init__(
            learning_rate=lr,
            T_max=step_each_epoch * epochs, )

        self.update_specified = False


class Piecewise(PiecewiseDecay):
    """
    Piecewise learning rate decay

    Args:
        lr(float): initial learning rate
        step_each_epoch(int): steps each epoch
        decay_epochs(list): piecewise decay epochs
        gamma(float): decay factor
    """

    def __init__(self, lr, step_each_epoch, decay_epochs, gamma=0.1, **kwargs):
        boundaries = [step_each_epoch * e for e in decay_epochs]
        lr_values = [lr * (gamma**i) for i in range(len(boundaries) + 1)]
        super(Piecewise, self).__init__(
            boundaries=boundaries, values=lr_values)

        self.update_specified = False


class CosineWarmup(LinearWarmup):
    """
    Cosine learning rate decay with warmup
    [0, warmup_epoch): linear warmup
    [warmup_epoch, epochs): cosine decay

    Args:
        lr(float): initial learning rate
        step_each_epoch(int): steps each epoch
        epochs(int): total training epochs
        warmup_epoch(int): epoch num of warmup
    """

    def __init__(self, lr, step_each_epoch, epochs, warmup_epoch=5, **kwargs):
        assert epochs > warmup_epoch, "total epoch({}) should be larger than warmup_epoch({}) in CosineWarmup.".format(
            epochs, warmup_epoch)
        warmup_step = warmup_epoch * step_each_epoch
        start_lr = 0.0
        end_lr = lr
        lr_sch = Cosine(lr, step_each_epoch, epochs - warmup_epoch)

        super(CosineWarmup, self).__init__(
            learning_rate=lr_sch,
            warmup_steps=warmup_step,
            start_lr=start_lr,
            end_lr=end_lr)

        self.update_specified = False


class ExponentialWarmup(LinearWarmup):
    """
    Exponential learning rate decay with warmup
    [0, warmup_epoch): linear warmup
    [warmup_epoch, epochs): Exponential decay

    Args:
        lr(float): initial learning rate
        step_each_epoch(int): steps each epoch
        decay_epochs(float): decay epochs
        decay_rate(float): decay rate
        warmup_epoch(int): epoch num of warmup
    """

    def __init__(self,
                 lr,
                 step_each_epoch,
                 decay_epochs=2.4,
                 decay_rate=0.97,
                 warmup_epoch=5,
                 **kwargs):
        warmup_step = warmup_epoch * step_each_epoch
        start_lr = 0.0
        end_lr = lr
        lr_sch = ExponentialDecay(lr, decay_rate)

        super(ExponentialWarmup, self).__init__(
            learning_rate=lr_sch,
            warmup_steps=warmup_step,
            start_lr=start_lr,
            end_lr=end_lr)

        # NOTE: hac method to update exponential lr scheduler
        self.update_specified = True
        self.update_start_step = warmup_step
        self.update_step_interval = int(decay_epochs * step_each_epoch)
        self.step_each_epoch = step_each_epoch


class LearningRateBuilder():
    """
    Build learning rate variable
    https://www.paddlepaddle.org.cn/documentation/docs/zh/api_cn/layers_cn.html

    Args:
        function(str): class name of learning rate
        params(dict): parameters used for init the class
    """

    def __init__(self,
                 function='Linear',
                 params={'lr': 0.1,
                         'steps': 100,
                         'end_lr': 0.0}):
        self.function = function
        self.params = params

    def __call__(self):
        mod = sys.modules[__name__]
        lr = getattr(mod, self.function)(**self.params)
        return lr
