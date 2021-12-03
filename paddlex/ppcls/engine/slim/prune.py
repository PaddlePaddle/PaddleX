# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import, division, print_function
import paddle
from paddlex.ppcls.utils import logger


def get_pruner(config, model):
    if config.get("Slim", False) and config["Slim"].get("prune", False):
        import paddleslim
        prune_method_name = config["Slim"]["prune"]["name"].lower()
        assert prune_method_name in [
            "fpgm", "l1_norm"
        ], "The prune methods only support 'fpgm' and 'l1_norm'"
        if prune_method_name == "fpgm":
            pruner = paddleslim.dygraph.FPGMFilterPruner(
                model, [1] + config["Global"]["image_shape"])
        else:
            pruner = paddleslim.dygraph.L1NormFilterPruner(
                model, [1] + config["Global"]["image_shape"])

        # prune model
        _prune_model(pruner, config, model)
    else:
        pruner = None

    return pruner


def _prune_model(pruner, config, model):
    from paddleslim.analysis import dygraph_flops as flops
    logger.info("FLOPs before pruning: {}GFLOPs".format(
        flops(model, [1] + config["Global"]["image_shape"]) / 1e9))
    model.eval()

    params = []
    for sublayer in model.sublayers():
        for param in sublayer.parameters(include_sublayers=False):
            if isinstance(sublayer, paddle.nn.Conv2D):
                params.append(param.name)
    ratios = {}
    for param in params:
        ratios[param] = config["Slim"]["prune"]["pruned_ratio"]
    plan = pruner.prune_vars(ratios, [0])

    logger.info("FLOPs after pruning: {}GFLOPs; pruned ratio: {}".format(
        flops(model, [1] + config["Global"]["image_shape"]) / 1e9,
        plan.pruned_flops))

    for param in model.parameters():
        if "conv2d" in param.name:
            logger.info("{}\t{}".format(param.name, param.shape))

    model.train()
