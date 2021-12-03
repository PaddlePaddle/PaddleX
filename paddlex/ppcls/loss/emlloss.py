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

import math
import paddle
import numpy as np
from .comfunc import rerange_index


class EmlLoss(paddle.nn.Layer):
    def __init__(self, batch_size=40, samples_each_class=2):
        super(EmlLoss, self).__init__()
        assert (batch_size % samples_each_class == 0)
        self.samples_each_class = samples_each_class
        self.batch_size = batch_size
        self.rerange_index = rerange_index(batch_size, samples_each_class)
        self.thresh = 20.0
        self.beta = 100000

    def surrogate_function(self, beta, theta, bias):
        x = theta * paddle.exp(bias)
        output = paddle.log(1 + beta * x) / math.log(1 + beta)
        return output

    def surrogate_function_approximate(self, beta, theta, bias):
        output = (
            paddle.log(theta) + bias + math.log(beta)) / math.log(1 + beta)
        return output

    def surrogate_function_stable(self, beta, theta, target, thresh):
        max_gap = paddle.to_tensor(thresh, dtype='float32')
        max_gap.stop_gradient = True

        target_max = paddle.maximum(target, max_gap)
        target_min = paddle.minimum(target, max_gap)

        loss1 = self.surrogate_function(beta, theta, target_min)
        loss2 = self.surrogate_function_approximate(beta, theta, target_max)
        bias = self.surrogate_function(beta, theta, max_gap)
        loss = loss1 + loss2 - bias
        return loss

    def forward(self, input, target=None):
        features = input["features"]
        samples_each_class = self.samples_each_class
        batch_size = self.batch_size
        rerange_index = self.rerange_index

        #calc distance
        diffs = paddle.unsqueeze(
            features, axis=1) - paddle.unsqueeze(
                features, axis=0)
        similary_matrix = paddle.sum(paddle.square(diffs), axis=-1)

        tmp = paddle.reshape(similary_matrix, shape=[-1, 1])
        rerange_index = paddle.to_tensor(rerange_index)
        tmp = paddle.gather(tmp, index=rerange_index)
        similary_matrix = paddle.reshape(tmp, shape=[-1, batch_size])

        ignore, pos, neg = paddle.split(
            similary_matrix,
            num_or_sections=[
                1, samples_each_class - 1, batch_size - samples_each_class
            ],
            axis=1)
        ignore.stop_gradient = True

        pos_max = paddle.max(pos, axis=1, keepdim=True)
        pos = paddle.exp(pos - pos_max)
        pos_mean = paddle.mean(pos, axis=1, keepdim=True)

        neg_min = paddle.min(neg, axis=1, keepdim=True)
        neg = paddle.exp(neg_min - neg)
        neg_mean = paddle.mean(neg, axis=1, keepdim=True)

        bias = pos_max - neg_min
        theta = paddle.multiply(neg_mean, pos_mean)

        loss = self.surrogate_function_stable(self.beta, theta, bias,
                                              self.thresh)
        loss = paddle.mean(loss)
        return {"emlloss": loss}
