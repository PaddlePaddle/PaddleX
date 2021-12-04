#copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import paddle
import paddle.nn as nn


class DSHSDLoss(nn.Layer):
    """
    # DSHSD(IEEE ACCESS 2019)
    # paper [Deep Supervised Hashing Based on Stable Distribution](https://ieeexplore.ieee.org/document/8648432/)
    # [DSHSD] epoch:70,  bit:48,  dataset:cifar10-1,  MAP:0.809, Best MAP: 0.809
    # [DSHSD] epoch:250, bit:48,  dataset:nuswide_21, MAP:0.809, Best MAP: 0.815
    # [DSHSD] epoch:135, bit:48,  dataset:imagenet,   MAP:0.647, Best MAP: 0.647
    """

    def __init__(self, alpha, multi_label=False):
        super(DSHSDLoss, self).__init__()
        self.alpha = alpha
        self.multi_label = multi_label

    def forward(self, input, label):
        feature = input["features"]
        logits = input["logits"]

        dist = paddle.sum(paddle.square(
            (paddle.unsqueeze(feature, 1) - paddle.unsqueeze(feature, 0))),
                          axis=2)

        # label to ont-hot
        label = paddle.flatten(label)
        n_class = logits.shape[1]
        label = paddle.nn.functional.one_hot(label, n_class).astype("float32")

        s = (paddle.matmul(
            label, label, transpose_y=True) == 0).astype("float32")
        margin = 2 * feature.shape[1]
        Ld = (1 - s) / 2 * dist + s / 2 * (margin - dist).clip(min=0)
        Ld = Ld.mean()

        if self.multi_label:
            # multiple labels classification loss
            Lc = (logits - label * logits + (
                (1 + (-logits).exp()).log())).sum(axis=1).mean()
        else:
            # single labels classification loss
            Lc = (-paddle.nn.functional.softmax(logits).log() * label).sum(
                axis=1).mean()

        return {"dshsdloss": Lc + Ld * self.alpha}


class LCDSHLoss(nn.Layer):
    """
    # paper [Locality-Constrained Deep Supervised Hashing for Image Retrieval](https://www.ijcai.org/Proceedings/2017/0499.pdf)
    # [LCDSH] epoch:145, bit:48, dataset:cifar10-1,  MAP:0.798, Best MAP: 0.798
    # [LCDSH] epoch:183, bit:48, dataset:nuswide_21, MAP:0.833, Best MAP: 0.834
    """

    def __init__(self, n_class, _lambda):
        super(LCDSHLoss, self).__init__()
        self._lambda = _lambda
        self.n_class = n_class

    def forward(self, input, label):
        feature = input["features"]

        # label to ont-hot
        label = paddle.flatten(label)
        label = paddle.nn.functional.one_hot(label,
                                             self.n_class).astype("float32")

        s = 2 * (paddle.matmul(
            label, label, transpose_y=True) > 0).astype("float32") - 1
        inner_product = paddle.matmul(feature, feature, transpose_y=True) * 0.5

        inner_product = inner_product.clip(min=-50, max=50)
        L1 = paddle.log(1 + paddle.exp(-s * inner_product)).mean()

        b = feature.sign()
        inner_product_ = paddle.matmul(b, b, transpose_y=True) * 0.5
        sigmoid = paddle.nn.Sigmoid()
        L2 = (sigmoid(inner_product) - sigmoid(inner_product_)).pow(2).mean()

        return {"lcdshloss": L1 + self._lambda * L2}
