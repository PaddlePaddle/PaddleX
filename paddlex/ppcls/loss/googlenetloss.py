# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class GoogLeNetLoss(nn.Layer):
    """
    Cross entropy loss used after googlenet
    """

    def __init__(self, epsilon=None):
        super().__init__()
        assert (epsilon is None or epsilon <= 0 or
                epsilon >= 1), "googlenet is not support label_smooth"

    def forward(self, inputs, label):
        input0, input1, input2 = inputs
        if isinstance(input0, dict):
            input0 = input0["logits"]
        if isinstance(input1, dict):
            input1 = input1["logits"]
        if isinstance(input2, dict):
            input2 = input2["logits"]

        loss0 = F.cross_entropy(input0, label=label, soft_label=False)
        loss1 = F.cross_entropy(input1, label=label, soft_label=False)
        loss2 = F.cross_entropy(input2, label=label, soft_label=False)
        loss = loss0 + 0.3 * loss1 + 0.3 * loss2
        loss = loss.mean()
        return {"GooleNetLoss": loss}
