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
import math
import paddle.nn as nn


class CosMargin(paddle.nn.Layer):
    def __init__(self, embedding_size, class_num, margin=0.35, scale=64.0):
        super(CosMargin, self).__init__()
        self.scale = scale
        self.margin = margin
        self.embedding_size = embedding_size
        self.class_num = class_num

        self.weight = self.create_parameter(
            shape=[self.embedding_size, self.class_num],
            is_bias=False,
            default_initializer=paddle.nn.initializer.XavierNormal())

    def forward(self, input, label):
        label.stop_gradient = True

        input_norm = paddle.sqrt(
            paddle.sum(paddle.square(input), axis=1, keepdim=True))
        input = paddle.divide(input, input_norm)

        weight_norm = paddle.sqrt(
            paddle.sum(paddle.square(self.weight), axis=0, keepdim=True))
        weight = paddle.divide(self.weight, weight_norm)

        cos = paddle.matmul(input, weight)
        if not self.training or label is None:
            return cos

        cos_m = cos - self.margin

        one_hot = paddle.nn.functional.one_hot(label, self.class_num)
        one_hot = paddle.squeeze(one_hot, axis=[1])
        output = paddle.multiply(one_hot, cos_m) + paddle.multiply(
            (1.0 - one_hot), cos)
        output = output * self.scale
        return output
