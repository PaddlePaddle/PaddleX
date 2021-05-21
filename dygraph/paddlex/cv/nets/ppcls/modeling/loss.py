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
import paddle.nn.functional as F

__all__ = ['CELoss', 'JSDivLoss']


class Loss(object):
    """
    Loss
    """

    def __init__(self, class_dim=1000, epsilon=None):
        assert class_dim > 1, "class_dim=%d is not larger than 1" % (class_dim)
        self._class_dim = class_dim
        if epsilon is not None and epsilon >= 0.0 and epsilon <= 1.0:
            self._epsilon = epsilon
            self._label_smoothing = True
        else:
            self._epsilon = None
            self._label_smoothing = False

    def _labelsmoothing(self, target):
        if target.shape[-1] != self._class_dim:
            one_hot_target = F.one_hot(target, self._class_dim)
        else:
            one_hot_target = target
        soft_target = F.label_smooth(one_hot_target, epsilon=self._epsilon)
        soft_target = paddle.reshape(soft_target, shape=[-1, self._class_dim])
        return soft_target

    def _crossentropy(self, input, target, use_pure_fp16=False):
        if self._label_smoothing:
            target = self._labelsmoothing(target)
            input = -F.log_softmax(input, axis=-1)
            cost = paddle.sum(target * input, axis=-1)
        else:
            cost = F.cross_entropy(input=input, label=target)
        if use_pure_fp16:
            avg_cost = paddle.sum(cost)
        else:
            avg_cost = paddle.mean(cost)
        return avg_cost

    def _kldiv(self, input, target, name=None):
        eps = 1.0e-10
        cost = target * paddle.log(
            (target + eps) / (input + eps)) * self._class_dim
        return cost

    def _jsdiv(self, input, target):
        input = F.softmax(input)
        target = F.softmax(target)
        cost = self._kldiv(input, target) + self._kldiv(target, input)
        cost = cost / 2
        avg_cost = paddle.mean(cost)
        return avg_cost

    def __call__(self, input, target):
        pass


class CELoss(Loss):
    """
    Cross entropy loss
    """

    def __init__(self, class_dim=1000, epsilon=None):
        super(CELoss, self).__init__(class_dim, epsilon)

    def __call__(self, input, target, use_pure_fp16=False):
        cost = self._crossentropy(input, target, use_pure_fp16)
        return cost


class JSDivLoss(Loss):
    """
    JSDiv loss
    """

    def __init__(self, class_dim=1000, epsilon=None):
        super(JSDivLoss, self).__init__(class_dim, epsilon)

    def __call__(self, input, target):
        cost = self._jsdiv(input, target)
        return cost
