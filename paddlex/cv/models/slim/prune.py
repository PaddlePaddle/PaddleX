# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
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

import numpy as np
import paddle
import paddleslim.dygraph.filter_pruner
from paddleslim.dygraph import FilterPruner

FILTER_DIM = paddleslim.dygraph.filter_pruner.FILTER_DIM


def _pruner_eval_fn(model, eval_dataset, batch_size):
    metric = model.evaluate(eval_dataset, batch_size=batch_size)
    return metric[list(metric.keys())[0]]


def _pruner_template_input(sample, model_type):
    if model_type == 'detector':
        template_input = [{
            "image": paddle.ones(
                shape=[1, 3] + list(sample["image"].shape[:2]),
                dtype='float32'),
            "im_shape": paddle.full(
                [1, 2], 640, dtype='float32'),
            "scale_factor": paddle.ones(
                shape=[1, 2], dtype='float32')
        }]
    else:
        template_input = [1] + list(sample[0].shape)

    return template_input


class XFilterPruner(FilterPruner):
    def __init__(self, model, inputs, sen_file=None):
        super(XFilterPruner, self).__init__(model, inputs, sen_file)

    def _sensitive_prune(self, pruned_flops, skip_vars=[], align=None):
        """
        This method is exactly the same as paddleslim.FilterPruner.sensitive_prune()
        except that this function returns ratios besides the pruning plan.

        """
        # skip depthwise convolutions
        for layer in self.model.sublayers():
            if isinstance(layer,
                          paddle.nn.layer.conv.Conv2D) and layer._groups > 1:
                for param in layer.parameters(include_sublayers=False):
                    skip_vars.append(param.name)
        self.restore()
        ratios, pruned_flops = self.get_ratios_by_sensitivity(
            pruned_flops, align=align, dims=FILTER_DIM, skip_vars=skip_vars)
        self.plan = self.prune_vars(ratios, FILTER_DIM)
        self.plan._pruned_flops = pruned_flops
        return self.plan, ratios


class L1NormFilterPruner(XFilterPruner):
    def __init__(self, model, inputs, sen_file=None):
        super(L1NormFilterPruner, self).__init__(
            model, inputs, sen_file=sen_file)

    def cal_mask(self, var_name, pruned_ratio, group):
        for _item in group[var_name]:
            if _item['pruned_dims'] == [0]:
                value = _item['value']
                pruned_dims = _item['pruned_dims']
        reduce_dims = [
            i for i in range(len(value.shape)) if i not in pruned_dims
        ]
        l1norm = np.mean(np.abs(value), axis=tuple(reduce_dims))
        sorted_idx = l1norm.argsort()
        pruned_num = int(round(len(sorted_idx) * pruned_ratio))
        pruned_idx = sorted_idx[:pruned_num]
        mask_shape = [value.shape[i] for i in pruned_dims]
        mask = np.ones(mask_shape, dtype="int32")
        mask[pruned_idx] = 0
        return mask


class FPGMFilterPruner(XFilterPruner):
    def __init__(self, model, inputs, sen_file=None):
        super(FPGMFilterPruner, self).__init__(
            model, inputs, sen_file=sen_file)

    def cal_mask(self, var_name, pruned_ratio, group):
        for _item in group[var_name]:
            if _item['pruned_dims'] == [0]:
                value = _item['value']
                pruned_dims = _item['pruned_dims']
        dist_sum_list = []
        for out_i in range(value.shape[0]):
            dist_sum = self.get_distance_sum(value, out_i)
            dist_sum_list.append(dist_sum)
        scores = np.array(dist_sum_list)

        sorted_idx = scores.argsort()
        pruned_num = int(round(len(sorted_idx) * pruned_ratio))
        pruned_idx = sorted_idx[:pruned_num]
        mask_shape = [value.shape[i] for i in pruned_dims]
        mask = np.ones(mask_shape, dtype="int32")
        mask[pruned_idx] = 0
        return mask

    def get_distance_sum(self, value, out_idx):
        w = value.view()
        w.shape = value.shape[0], np.product(value.shape[1:])
        selected_filter = np.tile(w[out_idx], (w.shape[0], 1))
        x = w - selected_filter
        x = np.sqrt(np.sum(x * x, -1))
        return x.sum()
