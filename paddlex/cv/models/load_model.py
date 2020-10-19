# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import yaml
import os.path as osp
import six
import copy
from collections import OrderedDict
import paddle.fluid as fluid
from paddle.fluid.framework import Parameter
import paddlex
import paddlex.utils.logging as logging
from paddlex.cv.transforms import build_transforms, build_transforms_v1


def load_model(model_dir, fixed_input_shape=None):
    model_scope = fluid.Scope()
    if not osp.exists(model_dir):
        logging.error("model_dir '{}' is not exists!".format(model_dir))
    if not osp.exists(osp.join(model_dir, "model.yml")):
        raise Exception("There's not model.yml in {}".format(model_dir))
    with open(osp.join(model_dir, "model.yml")) as f:
        info = yaml.load(f.read(), Loader=yaml.Loader)

    if 'status' in info:
        status = info['status']
    elif 'save_method' in info:
        # 兼容老版本PaddleX
        status = info['save_method']

    if not hasattr(paddlex.cv.models, info['Model']):
        raise Exception("There's no attribute {} in paddlex.cv.models".format(
            info['Model']))
    if 'model_name' in info['_init_params']:
        del info['_init_params']['model_name']
    model = getattr(paddlex.cv.models, info['Model'])(**info['_init_params'])

    model.fixed_input_shape = fixed_input_shape
    if '_Attributes' in info:
        if 'fixed_input_shape' in info['_Attributes']:
            fixed_input_shape = info['_Attributes']['fixed_input_shape']
            if fixed_input_shape is not None:
                logging.info("Model already has fixed_input_shape with {}".
                             format(fixed_input_shape))
                model.fixed_input_shape = fixed_input_shape

    with fluid.scope_guard(model_scope):
        if status == "Normal" or \
                status == "Prune" or status == "fluid.save":
            startup_prog = fluid.Program()
            model.test_prog = fluid.Program()
            with fluid.program_guard(model.test_prog, startup_prog):
                with fluid.unique_name.guard():
                    model.test_inputs, model.test_outputs = model.build_net(
                        mode='test')
            model.test_prog = model.test_prog.clone(for_test=True)
            model.exe.run(startup_prog)
            if status == "Prune":
                from .slim.prune import update_program
                model.test_prog = update_program(
                    model.test_prog,
                    model_dir,
                    model.places[0],
                    scope=model_scope)
            import pickle
            with open(osp.join(model_dir, 'model.pdparams'), 'rb') as f:
                load_dict = pickle.load(f)
            fluid.io.set_program_state(model.test_prog, load_dict)

        elif status == "Infer" or \
                status == "Quant" or status == "fluid.save_inference_model":
            [prog, input_names, outputs] = fluid.io.load_inference_model(
                model_dir, model.exe, params_filename='__params__')
            model.test_prog = prog
            test_outputs_info = info['_ModelInputsOutputs']['test_outputs']
            model.test_inputs = OrderedDict()
            model.test_outputs = OrderedDict()
            for name in input_names:
                model.test_inputs[name] = model.test_prog.global_block().var(
                    name)
            for i, out in enumerate(outputs):
                var_desc = test_outputs_info[i]
                model.test_outputs[var_desc[0]] = out
    if 'Transforms' in info:
        transforms_mode = info.get('TransformsMode', 'RGB')
        # 固定模型的输入shape
        fix_input_shape(info, fixed_input_shape=model.fixed_input_shape)
        if transforms_mode == 'RGB':
            to_rgb = True
        else:
            to_rgb = False
        if 'BatchTransforms' in info:
            # 兼容老版本PaddleX模型
            model.test_transforms = build_transforms_v1(
                model.model_type, info['Transforms'], info['BatchTransforms'])
            model.eval_transforms = copy.deepcopy(model.test_transforms)
        else:
            model.test_transforms = build_transforms(
                model.model_type, info['Transforms'], to_rgb)
            model.eval_transforms = copy.deepcopy(model.test_transforms)

    if '_Attributes' in info:
        for k, v in info['_Attributes'].items():
            if k in model.__dict__:
                model.__dict__[k] = v

    logging.info("Model[{}] loaded.".format(info['Model']))
    model.scope = model_scope
    model.trainable = False
    model.status = status
    return model


def fix_input_shape(info, fixed_input_shape=None):
    if fixed_input_shape is not None:
        input_channel = 3
        if 'input_channel' in info['_init_params']:
            input_channel = info['_init_params']['input_channel']
        resize = {'ResizeByShort': {}}
        padding = {'Padding': {}}
        if info['_Attributes']['model_type'] == 'classifier':
            pass
        else:
            resize['ResizeByShort']['short_size'] = min(fixed_input_shape)
            resize['ResizeByShort']['max_size'] = max(fixed_input_shape)
            padding['Padding']['target_size'] = list(fixed_input_shape)
            if info['_Attributes']['model_type'] == 'segmenter':
                padding['Padding']['im_padding_value'] = [0.] * input_channel
            info['Transforms'].append(resize)
            info['Transforms'].append(padding)
