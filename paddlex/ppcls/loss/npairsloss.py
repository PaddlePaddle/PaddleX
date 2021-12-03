from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import paddle


class NpairsLoss(paddle.nn.Layer):
    def __init__(self, reg_lambda=0.01):
        super(NpairsLoss, self).__init__()
        self.reg_lambda = reg_lambda

    def forward(self, input, target=None):
        """
        anchor and positive(should include label)
        """
        features = input["features"]
        reg_lambda = self.reg_lambda
        batch_size = features.shape[0]
        fea_dim = features.shape[1]
        num_class = batch_size // 2

        #reshape
        out_feas = paddle.reshape(features, shape=[-1, 2, fea_dim])
        anc_feas, pos_feas = paddle.split(out_feas, num_or_sections=2, axis=1)
        anc_feas = paddle.squeeze(anc_feas, axis=1)
        pos_feas = paddle.squeeze(pos_feas, axis=1)

        #get simi matrix
        similarity_matrix = paddle.matmul(
            anc_feas, pos_feas, transpose_y=True)  #get similarity matrix
        sparse_labels = paddle.arange(0, num_class, dtype='int64')
        xentloss = paddle.nn.CrossEntropyLoss()(
            similarity_matrix, sparse_labels)  #by default: mean

        #l2 norm
        reg = paddle.mean(paddle.sum(paddle.square(features), axis=1))
        l2loss = 0.5 * reg_lambda * reg
        return {"npairsloss": xentloss + l2loss}
