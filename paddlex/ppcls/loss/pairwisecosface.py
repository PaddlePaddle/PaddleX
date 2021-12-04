#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.nn as nn
import paddle.nn.functional as F


class PairwiseCosface(nn.Layer):
    def __init__(self, margin, gamma):
        super(PairwiseCosface, self).__init__()
        self.margin = margin
        self.gamma = gamma

    def forward(self, embedding, targets):
        if isinstance(embedding, dict):
            embedding = embedding['features']
        # Normalize embedding features
        embedding = F.normalize(embedding, axis=1)
        dist_mat = paddle.matmul(embedding, embedding, transpose_y=True)

        N = dist_mat.shape[0]
        is_pos = targets.reshape([N, 1]).expand([N, N]).equal(
            paddle.t(targets.reshape([N, 1]).expand([N, N]))).astype('float')
        is_neg = targets.reshape([N, 1]).expand([N, N]).not_equal(
            paddle.t(targets.reshape([N, 1]).expand([N, N]))).astype('float')

        # Mask scores related to itself
        is_pos = is_pos - paddle.eye(N, N)

        s_p = dist_mat * is_pos
        s_n = dist_mat * is_neg

        logit_p = -self.gamma * s_p + (-99999999.) * (1 - is_pos)
        logit_n = self.gamma * (s_n + self.margin) + (-99999999.) * (1 - is_neg
                                                                     )

        loss = F.softplus(
            paddle.logsumexp(
                logit_p, axis=1) + paddle.logsumexp(
                    logit_n, axis=1)).mean()

        return {"PairwiseCosface": loss}
