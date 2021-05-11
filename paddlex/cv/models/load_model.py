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

import os.path as osp
import copy

import paddleslim.dygraph.filter_pruner
import yaml
import paddlex
import paddlex.utils.logging as logging
from paddlex.cv.transforms import build_transforms
from .slim.prune import L1NormFilterPruner, FPGMFilterPruner


def load_model(model_dir):
    if not osp.exists(model_dir):
        logging.error("model_dir '{}' does not exists!".format(model_dir))
    if not osp.exists(osp.join(model_dir, "model.yml")):
        raise Exception("There's no model.yml in {}".format(model_dir))
    with open(osp.join(model_dir, "model.yml")) as f:
        model_info = yaml.load(f.read(), Loader=yaml.Loader)
    f.close()

    if not hasattr(paddlex.cv.models, model_info['Model']):
        raise Exception("There's no attribute {} in paddlex.cv.models".format(
            model_info['Model']))
    if 'model_name' in model_info['_init_params']:
        del model_info['_init_params']['model_name']
    model = getattr(paddlex.cv.models, model_info['Model'])(
        **model_info['_init_params'])

    if 'Transforms' in model_info:
        model.test_transforms = build_transforms(model_info['Transforms'])

    if '_Attributes' in model_info:
        for k, v in model_info['_Attributes'].items():
            if k in model.__dict__:
                model.__dict__[k] = v

    if 'status' in model_info:
        status = model_info['status']
        if status == 'Pruned':
            with open(osp.join(model_dir, "prune.yml")) as f:
                pruning_info = yaml.load(f.read(), Loader=yaml.Loader)
                inputs = pruning_info['pruner_inputs']
                if pruning_info['criterion'] == 'l1_norm':
                    pruner = L1NormFilterPruner(model.net, inputs=inputs)
                elif pruning_info['criterion'] == 'fpgm':
                    pruner = FPGMFilterPruner(model.net, inputs=inputs)
                else:
                    raise Exception(
                        "The pruning criterion {} is not supported.".format(
                            pruning_info['criterion']))
                pruning_ratios = pruning_info['pruning_ratios']
                pruner.prune_vars(
                    ratios=pruning_ratios,
                    axis=paddleslim.dygraph.filter_pruner.FILTER_DIM)

    # load weights
    model.net_initialize(pretrain_weights=osp.join(model_dir,
                                                   'model.pdparams'))

    logging.info("Model[{}] loaded.".format(model_info['Model']))
    model.trainable = False
    model.status = status
    return model
