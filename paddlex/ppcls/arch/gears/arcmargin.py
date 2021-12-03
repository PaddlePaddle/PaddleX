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

import paddle
import paddle.nn as nn
import math


class ArcMargin(nn.Layer):
    def __init__(self,
                 embedding_size,
                 class_num,
                 margin=0.5,
                 scale=80.0,
                 easy_margin=False):
        super().__init__()
        self.embedding_size = embedding_size
        self.class_num = class_num
        self.margin = margin
        self.scale = scale
        self.easy_margin = easy_margin
        self.weight = self.create_parameter(
            shape=[self.embedding_size, self.class_num],
            is_bias=False,
            default_initializer=paddle.nn.initializer.XavierNormal())

    def forward(self, input, label=None):
        input_norm = paddle.sqrt(
            paddle.sum(paddle.square(input), axis=1, keepdim=True))
        input = paddle.divide(input, input_norm)

        weight_norm = paddle.sqrt(
            paddle.sum(paddle.square(self.weight), axis=0, keepdim=True))
        weight = paddle.divide(self.weight, weight_norm)

        cos = paddle.matmul(input, weight)
        if not self.training or label is None:
            return cos
        sin = paddle.sqrt(1.0 - paddle.square(cos) + 1e-6)
        cos_m = math.cos(self.margin)
        sin_m = math.sin(self.margin)
        phi = cos * cos_m - sin * sin_m

        th = math.cos(self.margin) * (-1)
        mm = math.sin(self.margin) * self.margin
        if self.easy_margin:
            phi = self._paddle_where_more_than(cos, 0, phi, cos)
        else:
            phi = self._paddle_where_more_than(cos, th, phi, cos - mm)

        one_hot = paddle.nn.functional.one_hot(label, self.class_num)
        one_hot = paddle.squeeze(one_hot, axis=[1])
        output = paddle.multiply(one_hot, phi) + paddle.multiply(
            (1.0 - one_hot), cos)
        output = output * self.scale
        return output

    def _paddle_where_more_than(self, target, limit, x, y):
        mask = paddle.cast(x=(target > limit), dtype='float32')
        output = paddle.multiply(mask, x) + paddle.multiply((1.0 - mask), y)
        return output
