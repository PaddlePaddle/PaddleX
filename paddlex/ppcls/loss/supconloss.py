import paddle
from paddle import nn


class SupConLoss(nn.Layer):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self,
                 views=16,
                 temperature=0.07,
                 contrast_mode='all',
                 base_temperature=0.07,
                 normalize_feature=True):
        super(SupConLoss, self).__init__()
        self.temperature = paddle.to_tensor(temperature)
        self.contrast_mode = contrast_mode
        self.base_temperature = paddle.to_tensor(base_temperature)
        self.num_ids = None
        self.views = views
        self.normalize_feature = normalize_feature

    def forward(self, features, labels, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        features = features["features"]
        if self.num_ids is None:
            self.num_ids = int(features.shape[0] / self.views)

        if self.normalize_feature:
            features = 1. * features / (paddle.expand_as(
                paddle.norm(
                    features, p=2, axis=-1, keepdim=True), features) + 1e-12)
        features = features.reshape([self.num_ids, self.views, -1])
        labels = labels.reshape([self.num_ids, self.views])[:, 0]

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.reshape(
                [features.shape[0], features.shape[1], -1])

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = paddle.eye(batch_size, dtype='float32')
        elif labels is not None:
            labels = labels.reshape([-1, 1])
            if labels.shape[0] != batch_size:
                raise ValueError(
                    'Num of labels does not match num of features')
            mask = paddle.cast(
                paddle.equal(labels, paddle.t(labels)), 'float32')
        else:
            mask = paddle.cast(mask, 'float32')

        contrast_count = features.shape[1]
        contrast_feature = paddle.concat(
            paddle.unbind(
                features, axis=1), axis=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = paddle.divide(
            paddle.matmul(anchor_feature, paddle.t(contrast_feature)),
            self.temperature)
        # for numerical stability
        logits_max = paddle.max(anchor_dot_contrast, axis=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = paddle.tile(mask, [anchor_count, contrast_count])

        logits_mask = 1 - paddle.eye(batch_size * anchor_count)
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = paddle.exp(logits) * logits_mask
        log_prob = logits - paddle.log(
            paddle.sum(exp_logits, axis=1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = paddle.sum((mask * log_prob),
                                       axis=1) / paddle.sum(mask, axis=1)

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = paddle.mean(loss.reshape([anchor_count, batch_size]))

        return {"SupConLoss": loss}
