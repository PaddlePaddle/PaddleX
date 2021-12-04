import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class MultiLabelLoss(nn.Layer):
    """
    Multi-label loss
    """

    def __init__(self, epsilon=None):
        super().__init__()
        if epsilon is not None and (epsilon <= 0 or epsilon >= 1):
            epsilon = None
        self.epsilon = epsilon

    def _labelsmoothing(self, target, class_num):
        if target.ndim == 1 or target.shape[-1] != class_num:
            one_hot_target = F.one_hot(target, class_num)
        else:
            one_hot_target = target
        soft_target = F.label_smooth(one_hot_target, epsilon=self.epsilon)
        soft_target = paddle.reshape(soft_target, shape=[-1, class_num])
        return soft_target

    def _binary_crossentropy(self, input, target, class_num):
        if self.epsilon is not None:
            target = self._labelsmoothing(target, class_num)
            cost = F.binary_cross_entropy_with_logits(
                logit=input, label=target)
        else:
            cost = F.binary_cross_entropy_with_logits(
                logit=input, label=target)

        return cost

    def forward(self, x, target):
        if isinstance(x, dict):
            x = x["logits"]
        class_num = x.shape[-1]
        loss = self._binary_crossentropy(x, target, class_num)
        loss = loss.mean()
        return {"MultiLabelLoss": loss}
