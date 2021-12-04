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

import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class CircleMargin(nn.Layer):
    def __init__(self, embedding_size, class_num, margin, scale):
        super(CircleMargin, self).__init__()
        self.scale = scale
        self.margin = margin
        self.embedding_size = embedding_size
        self.class_num = class_num

        self.weight = self.create_parameter(
            shape=[self.embedding_size, self.class_num],
            is_bias=False,
            default_initializer=paddle.nn.initializer.XavierNormal())

    def forward(self, input, label):
        feat_norm = paddle.sqrt(
            paddle.sum(paddle.square(input), axis=1, keepdim=True))
        input = paddle.divide(input, feat_norm)

        weight_norm = paddle.sqrt(
            paddle.sum(paddle.square(self.weight), axis=0, keepdim=True))
        weight = paddle.divide(self.weight, weight_norm)

        logits = paddle.matmul(input, weight)
        if not self.training or label is None:
            return logits

        alpha_p = paddle.clip(-logits.detach() + 1 + self.margin, min=0.)
        alpha_n = paddle.clip(logits.detach() + self.margin, min=0.)
        delta_p = 1 - self.margin
        delta_n = self.margin

        m_hot = F.one_hot(label.reshape([-1]), num_classes=logits.shape[1])

        logits_p = alpha_p * (logits - delta_p)
        logits_n = alpha_n * (logits - delta_n)
        pre_logits = logits_p * m_hot + logits_n * (1 - m_hot)
        pre_logits = self.scale * pre_logits

        return pre_logits
