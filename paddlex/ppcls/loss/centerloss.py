from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class CenterLoss(nn.Layer):
    def __init__(self, num_classes=5013, feat_dim=2048):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = paddle.randn(
            shape=[self.num_classes, self.feat_dim]).astype(
                "float64")  #random center

    def __call__(self, input, target):
        """
        inputs: network output: {"features: xxx", "logits": xxxx}
        target: image label
        """
        feats = input["features"]
        labels = target
        batch_size = feats.shape[0]

        #calc feat * feat
        dist1 = paddle.sum(paddle.square(feats), axis=1, keepdim=True)
        dist1 = paddle.expand(dist1, [batch_size, self.num_classes])

        #dist2 of centers
        dist2 = paddle.sum(paddle.square(self.centers), axis=1,
                           keepdim=True)  #num_classes
        dist2 = paddle.expand(dist2,
                              [self.num_classes, batch_size]).astype("float64")
        dist2 = paddle.transpose(dist2, [1, 0])

        #first x * x + y * y
        distmat = paddle.add(dist1, dist2)
        tmp = paddle.matmul(feats, paddle.transpose(self.centers, [1, 0]))
        distmat = distmat - 2.0 * tmp

        #generate the mask
        classes = paddle.arange(self.num_classes).astype("int64")
        labels = paddle.expand(
            paddle.unsqueeze(labels, 1), (batch_size, self.num_classes))
        mask = paddle.equal(
            paddle.expand(classes, [batch_size, self.num_classes]),
            labels).astype("float64")  #get mask

        dist = paddle.multiply(distmat, mask)
        loss = paddle.sum(paddle.clip(dist, min=1e-12, max=1e+12)) / batch_size

        return {'CenterLoss': loss}
