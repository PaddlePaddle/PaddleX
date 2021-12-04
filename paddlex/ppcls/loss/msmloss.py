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
import paddle
from .comfunc import rerange_index


class MSMLoss(paddle.nn.Layer):
    """
    MSMLoss Loss, based on triplet loss. USE P * K samples.
    the batch size is fixed. Batch_size = P * K;  but the K may vary between batches.
    same label gather together

            supported_metrics = [
            'euclidean',
            'sqeuclidean',
            'cityblock',
        ]
    only consider samples_each_class = 2
    """

    def __init__(self, batch_size=120, samples_each_class=2, margin=0.1):
        super(MSMLoss, self).__init__()
        self.margin = margin
        self.samples_each_class = samples_each_class
        self.batch_size = batch_size
        self.rerange_index = rerange_index(batch_size, samples_each_class)

    def forward(self, input, target=None):
        #normalization
        features = input["features"]
        features = self._nomalize(features)
        samples_each_class = self.samples_each_class
        rerange_index = paddle.to_tensor(self.rerange_index)

        #calc sm
        diffs = paddle.unsqueeze(
            features, axis=1) - paddle.unsqueeze(
                features, axis=0)
        similary_matrix = paddle.sum(paddle.square(diffs), axis=-1)

        #rerange
        tmp = paddle.reshape(similary_matrix, shape=[-1, 1])
        tmp = paddle.gather(tmp, index=rerange_index)
        similary_matrix = paddle.reshape(tmp, shape=[-1, self.batch_size])

        #split
        ignore, pos, neg = paddle.split(
            similary_matrix,
            num_or_sections=[1, samples_each_class - 1, -1],
            axis=1)
        ignore.stop_gradient = True

        hard_pos = paddle.max(pos)
        hard_neg = paddle.min(neg)

        loss = hard_pos + self.margin - hard_neg
        loss = paddle.nn.ReLU()(loss)
        return {"msmloss": loss}

    def _nomalize(self, input):
        input_norm = paddle.sqrt(
            paddle.sum(paddle.square(input), axis=1, keepdim=True))
        return paddle.divide(input, input_norm)
