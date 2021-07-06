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
import numpy as np
import yaml
import paddle
import paddleslim
import paddlex
import paddlex.utils.logging as logging
from paddlex.cv.transforms import build_transforms


def load_rcnn_inference_model(model_dir):
    paddle.enable_static()
    exe = paddle.static.Executor(paddle.CPUPlace())
    path_prefix = osp.join(model_dir, "model")
    prog, _, _ = paddle.static.load_inference_model(path_prefix, exe)
    paddle.disable_static()
    extra_var_info = paddle.load(osp.join(model_dir, "model.pdiparams.info"))

    net_state_dict = dict()
    static_state_dict = dict()

    for name, var in prog.state_dict().items():
        static_state_dict[name] = np.array(var)
    for var_name in static_state_dict:
        if var_name not in extra_var_info:
            continue
        structured_name = extra_var_info[var_name].get('structured_name', None)
        if structured_name is None:
            continue
        net_state_dict[structured_name] = static_state_dict[var_name]
    return net_state_dict


def load_model(model_dir):
    """
    Load saved model from a given directory.
    Args:
        model_dir(str): The directory where the model is saved.

    Returns:
        The model loaded from the directory.
    """
    if not osp.exists(model_dir):
        logging.error("model_dir '{}' does not exists!".format(model_dir))
    if not osp.exists(osp.join(model_dir, "model.yml")):
        raise Exception("There's no model.yml in {}".format(model_dir))
    with open(osp.join(model_dir, "model.yml")) as f:
        model_info = yaml.load(f.read(), Loader=yaml.Loader)
    f.close()

    version = model_info['version']
    if int(version.split('.')[0]) < 2:
        raise Exception(
            'Current version is {}, a model trained by PaddleX={} cannot be load.'.
            format(paddlex.__version__, version))

    status = model_info['status']

    if not hasattr(paddlex.cv.models, model_info['Model']):
        raise Exception("There's no attribute {} in paddlex.cv.models".format(
            model_info['Model']))
    if 'model_name' in model_info['_init_params']:
        del model_info['_init_params']['model_name']

    with paddle.utils.unique_name.guard():
        model = getattr(paddlex.cv.models, model_info['Model'])(
            **model_info['_init_params'])

        if 'Transforms' in model_info:
            model.test_transforms = build_transforms(model_info['Transforms'])

        if '_Attributes' in model_info:
            for k, v in model_info['_Attributes'].items():
                if k in model.__dict__:
                    model.__dict__[k] = v

        if status == 'Pruned' or osp.exists(osp.join(model_dir, "prune.yml")):
            with open(osp.join(model_dir, "prune.yml")) as f:
                pruning_info = yaml.load(f.read(), Loader=yaml.Loader)
                inputs = pruning_info['pruner_inputs']
                if model.model_type == 'detector':
                    inputs = [{
                        k: paddle.to_tensor(v)
                        for k, v in inputs.items()
                    }]
                    model.net.eval()
                model.pruner = getattr(paddleslim, pruning_info['pruner'])(
                    model.net, inputs=inputs)
                model.pruning_ratios = pruning_info['pruning_ratios']
                model.pruner.prune_vars(
                    ratios=model.pruning_ratios,
                    axis=paddleslim.dygraph.prune.filter_pruner.FILTER_DIM)

        if status == 'Quantized':
            with open(osp.join(model_dir, "quant.yml")) as f:
                quant_info = yaml.load(f.read(), Loader=yaml.Loader)
                model.quant_config = quant_info['quant_config']
                model.quantizer = paddleslim.QAT(model.quant_config)
                model.quantizer.quantize(model.net)

        if status == 'Infer':
            if model_info['Model'] in ['FasterRCNN', 'MaskRCNN']:
                net_state_dict = load_rcnn_inference_model(model_dir)
            else:
                net_state_dict = paddle.load(osp.join(model_dir, 'model'))
        else:
            net_state_dict = paddle.load(osp.join(model_dir, 'model.pdparams'))
        model.net.set_state_dict(net_state_dict)

        logging.info("Model[{}] loaded.".format(model_info['Model']))
        model.status = status
    return model
