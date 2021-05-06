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
import yaml
import paddlex
import paddlex.utils.logging as logging
from paddlex.cv.transforms import build_transforms


def load_model(model_dir):
    if not osp.exists(model_dir):
        logging.error("model_dir '{}' is not exists!".format(model_dir))
    if not osp.exists(osp.join(model_dir, "model.yml")):
        raise Exception("There's not model.yml in {}".format(model_dir))
    with open(osp.join(model_dir, "model.yml")) as f:
        info = yaml.load(f.read(), Loader=yaml.Loader)

    if 'status' in info:
        status = info['status']

    if not hasattr(paddlex.cv.models, info['Model']):
        raise Exception("There's no attribute {} in paddlex.cv.models".format(
            info['Model']))
    if 'model_name' in info['_init_params']:
        del info['_init_params']['model_name']
    model = getattr(paddlex.cv.models, info['Model'])(**info['_init_params'])

    if 'Transforms' in info:
        model.test_transforms = build_transforms(info['Transforms'])

    if '_Attributes' in info:
        for k, v in info['_Attributes'].items():
            if k in model.__dict__:
                model.__dict__[k] = v

    # load weights
    model.net_initialize(pretrain_weights=osp.join(model_dir,
                                                   'model.pdparams'))

    logging.info("Model[{}] loaded.".format(info['Model']))
    model.trainable = False
    model.status = status
    return model
