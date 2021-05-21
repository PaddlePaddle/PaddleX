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

import paddle
import paddleslim

FILTER_DIM = paddleslim.dygraph.prune.filter_pruner.FILTER_DIM


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


def sensitive_prune(pruner, pruned_flops, skip_vars=[], align=None):
    # skip depthwise convolutions
    for layer in pruner.model.sublayers():
        if isinstance(layer,
                      paddle.nn.layer.conv.Conv2D) and layer._groups > 1:
            for param in layer.parameters(include_sublayers=False):
                skip_vars.append(param.name)
    pruner.restore()
    ratios, pruned_flops = pruner.get_ratios_by_sensitivity(
        pruned_flops, align=align, dims=FILTER_DIM, skip_vars=skip_vars)
    pruner.plan = pruner.prune_vars(ratios, FILTER_DIM)
    pruner.plan._pruned_flops = pruned_flops
    return pruner.plan, ratios
