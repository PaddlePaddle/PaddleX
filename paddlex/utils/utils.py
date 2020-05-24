# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

import sys
import time
import os
import os.path as osp
import numpy as np
import six
import yaml
import math
from . import logging


def seconds_to_hms(seconds):
    h = math.floor(seconds / 3600)
    m = math.floor((seconds - h * 3600) / 60)
    s = int(seconds - h * 3600 - m * 60)
    hms_str = "{}:{}:{}".format(h, m, s)
    return hms_str


def get_environ_info():
    import paddle.fluid as fluid
    info = dict()
    info['place'] = 'cpu'
    info['num'] = int(os.environ.get('CPU_NUM', 1))
    if os.environ.get('CUDA_VISIBLE_DEVICES', None) != "":
        if hasattr(fluid.core, 'get_cuda_device_count'):
            gpu_num = 0
            try:
                gpu_num = fluid.core.get_cuda_device_count()
            except:
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
                pass
            if gpu_num > 0:
                info['place'] = 'cuda'
                info['num'] = fluid.core.get_cuda_device_count()
    return info


def parse_param_file(param_file, return_shape=True):
    from paddle.fluid.proto.framework_pb2 import VarType
    f = open(param_file, 'rb')
    version = np.fromstring(f.read(4), dtype='int32')
    lod_level = np.fromstring(f.read(8), dtype='int64')
    for i in range(int(lod_level)):
        _size = np.fromstring(f.read(8), dtype='int64')
        _ = f.read(_size)
    version = np.fromstring(f.read(4), dtype='int32')
    tensor_desc = VarType.TensorDesc()
    tensor_desc_size = np.fromstring(f.read(4), dtype='int32')
    tensor_desc.ParseFromString(f.read(int(tensor_desc_size)))
    tensor_shape = tuple(tensor_desc.dims)
    if return_shape:
        f.close()
        return tuple(tensor_desc.dims)
    if tensor_desc.data_type != 5:
        raise Exception(
            "Unexpected data type while parse {}".format(param_file))
    data_size = 4
    for i in range(len(tensor_shape)):
        data_size *= tensor_shape[i]
    weight = np.fromstring(f.read(data_size), dtype='float32')
    f.close()
    return np.reshape(weight, tensor_shape)


def fuse_bn_weights(exe, main_prog, weights_dir):
    import paddle.fluid as fluid
    logging.info("Try to fuse weights of batch_norm...")
    bn_vars = list()
    for block in main_prog.blocks:
        ops = list(block.ops)
        for op in ops:
            if op.type == 'affine_channel':
                scale_name = op.input('Scale')[0]
                bias_name = op.input('Bias')[0]
                prefix = scale_name[:-5]
                mean_name = prefix + 'mean'
                variance_name = prefix + 'variance'
                if not osp.exists(osp.join(
                        weights_dir, mean_name)) or not osp.exists(
                            osp.join(weights_dir, variance_name)):
                    logging.info(
                        "There's no batch_norm weight found to fuse, skip fuse_bn."
                    )
                    return

                bias = block.var(bias_name)
                pretrained_shape = parse_param_file(
                    osp.join(weights_dir, bias_name))
                actual_shape = tuple(bias.shape)
                if pretrained_shape != actual_shape:
                    continue
                bn_vars.append(
                    [scale_name, bias_name, mean_name, variance_name])
    eps = 1e-5
    for names in bn_vars:
        scale_name, bias_name, mean_name, variance_name = names
        scale = parse_param_file(
            osp.join(weights_dir, scale_name), return_shape=False)
        bias = parse_param_file(
            osp.join(weights_dir, bias_name), return_shape=False)
        mean = parse_param_file(
            osp.join(weights_dir, mean_name), return_shape=False)
        variance = parse_param_file(
            osp.join(weights_dir, variance_name), return_shape=False)
        bn_std = np.sqrt(np.add(variance, eps))
        new_scale = np.float32(np.divide(scale, bn_std))
        new_bias = bias - mean * new_scale
        scale_tensor = fluid.global_scope().find_var(scale_name).get_tensor()
        bias_tensor = fluid.global_scope().find_var(bias_name).get_tensor()
        scale_tensor.set(new_scale, exe.place)
        bias_tensor.set(new_bias, exe.place)
    if len(bn_vars) == 0:
        logging.info(
            "There's no batch_norm weight found to fuse, skip fuse_bn.")
    else:
        logging.info("There's {} batch_norm ops been fused.".format(
            len(bn_vars)))


def load_pdparams(exe, main_prog, model_dir):
    import paddle.fluid as fluid
    from paddle.fluid.proto.framework_pb2 import VarType
    from paddle.fluid.framework import Program

    vars_to_load = list()
    import pickle
    with open(osp.join(model_dir, 'model.pdparams'), 'rb') as f:
        params_dict = pickle.load(f) if six.PY2 else pickle.load(
            f, encoding='latin1')
    unused_vars = list()
    for var in main_prog.list_vars():
        if not isinstance(var, fluid.framework.Parameter):
            continue
        if var.name not in params_dict:
            raise Exception("{} is not in saved paddlex model".format(
                var.name))
        if var.shape != params_dict[var.name].shape:
            unused_vars.append(var.name)
            logging.warning(
                "[SKIP] Shape of pretrained weight {} doesn't match.(Pretrained: {}, Actual: {})"
                .format(var.name, params_dict[var.name].shape, var.shape))
            continue
        vars_to_load.append(var)
        logging.debug("Weight {} will be load".format(var.name))
    for var_name in unused_vars:
        del params_dict[var_name]
    fluid.io.set_program_state(main_prog, params_dict)

    if len(vars_to_load) == 0:
        logging.warning(
            "There is no pretrain weights loaded, maybe you should check you pretrain model!"
        )
    else:
        logging.info("There are {} varaibles in {} are loaded.".format(
            len(vars_to_load), model_dir))


def is_persistable(var):
    import paddle.fluid as fluid
    from paddle.fluid.proto.framework_pb2 import VarType

    if var.desc.type() == fluid.core.VarDesc.VarType.FEED_MINIBATCH or \
        var.desc.type() == fluid.core.VarDesc.VarType.FETCH_LIST or \
        var.desc.type() == fluid.core.VarDesc.VarType.READER:
        return False
    return var.persistable


def is_belong_to_optimizer(var):
    import paddle.fluid as fluid
    from paddle.fluid.proto.framework_pb2 import VarType

    if not (isinstance(var, fluid.framework.Parameter)
            or var.desc.need_check_feed()):
        return is_persistable(var)
    return False


def load_pdopt(exe, main_prog, model_dir):
    import paddle.fluid as fluid

    optimizer_var_list = list()
    vars_to_load = list()
    import pickle
    with open(osp.join(model_dir, 'model.pdopt'), 'rb') as f:
        opt_dict = pickle.load(f) if six.PY2 else pickle.load(
            f, encoding='latin1')
    optimizer_var_list = list(
        filter(is_belong_to_optimizer, main_prog.list_vars()))
    exception_message = "the training process can not be resumed due to optimizer set now and last time is different. Recommend to use `pretrain_weights` instead of `resume_checkpoint`"
    if len(optimizer_var_list) > 0:
        for var in optimizer_var_list:
            if var.name not in opt_dict:
                raise Exception(
                    "{} is not in saved paddlex optimizer, {}".format(
                        var.name, exception_message))
            if var.shape != opt_dict[var.name].shape:
                raise Exception(
                    "Shape of optimizer variable {} doesn't match.(Last: {}, Now: {}), {}"
                    .format(var.name, opt_dict[var.name].shape,
                            var.shape), exception_message)
        optimizer_varname_list = [var.name for var in optimizer_var_list]
        for k, v in opt_dict.items():
            if k not in optimizer_varname_list:
                raise Exception(
                    "{} in saved paddlex optimizer is not in the model, {}".
                    format(k, exception_message))
        fluid.io.set_program_state(main_prog, opt_dict)

    if len(optimizer_var_list) == 0:
        raise Exception(
            "There is no optimizer parameters in the model, please set the optimizer!"
        )
    else:
        logging.info(
            "There are {} optimizer parameters in {} are loaded.".format(
                len(optimizer_var_list), model_dir))


def load_pretrain_weights(exe,
                          main_prog,
                          weights_dir,
                          fuse_bn=False,
                          resume=False):
    if not osp.exists(weights_dir):
        raise Exception("Path {} not exists.".format(weights_dir))
    if osp.exists(osp.join(weights_dir, "model.pdparams")):
        load_pdparams(exe, main_prog, weights_dir)
        if resume:
            if osp.exists(osp.join(weights_dir, "model.pdopt")):
                load_pdopt(exe, main_prog, weights_dir)
            else:
                raise Exception(
                    "Optimizer file {} does not exist. Stop resumming training. Recommend to use `pretrain_weights` instead of `resume_checkpoint`"
                    .format(osp.join(weights_dir, "model.pdopt")))
        return
    import paddle.fluid as fluid
    vars_to_load = list()
    for var in main_prog.list_vars():
        if not isinstance(var, fluid.framework.Parameter):
            continue
        if not osp.exists(osp.join(weights_dir, var.name)):
            logging.debug(
                "[SKIP] Pretrained weight {}/{} doesn't exist".format(
                    weights_dir, var.name))
            continue
        pretrained_shape = parse_param_file(osp.join(weights_dir, var.name))
        actual_shape = tuple(var.shape)
        if pretrained_shape != actual_shape:
            logging.warning(
                "[SKIP] Shape of pretrained weight {}/{} doesn't match.(Pretrained: {}, Actual: {})"
                .format(weights_dir, var.name, pretrained_shape, actual_shape))
            continue
        vars_to_load.append(var)
        logging.debug("Weight {} will be load".format(var.name))

    params_dict = fluid.io.load_program_state(
        weights_dir, var_list=vars_to_load)
    fluid.io.set_program_state(main_prog, params_dict)
    if len(vars_to_load) == 0:
        logging.warning(
            "There is no pretrain weights loaded, maybe you should check you pretrain model!"
        )
    else:
        logging.info("There are {} varaibles in {} are loaded.".format(
            len(vars_to_load), weights_dir))
    if fuse_bn:
        fuse_bn_weights(exe, main_prog, weights_dir)
    if resume:
        exception_message = "the training process can not be resumed due to optimizer set now and last time is different. Recommend to use `pretrain_weights` instead of `resume_checkpoint`"
        optimizer_var_list = list(
            filter(is_belong_to_optimizer, main_prog.list_vars()))
        if len(optimizer_var_list) > 0:
            for var in optimizer_var_list:
                if not osp.exists(osp.join(weights_dir, var.name)):
                    raise Exception(
                        "Optimizer parameter {} doesn't exist, {}".format(
                            osp.join(weights_dir, var.name),
                            exception_message))
                pretrained_shape = parse_param_file(
                    osp.join(weights_dir, var.name))
                actual_shape = tuple(var.shape)
                if pretrained_shape != actual_shape:
                    raise Exception(
                        "Shape of optimizer variable {} doesn't match.(Last: {}, Now: {}), {}"
                        .format(var.name, pretrained_shape,
                                actual_shape), exception_message)
            optimizer_varname_list = [var.name for var in optimizer_var_list]
            if os.exists(osp.join(weights_dir, 'learning_rate')
                         ) and 'learning_rate' not in optimizer_varname_list:
                raise Exception(
                    "Optimizer parameter {}/learning_rate is not in the model, {}"
                    .format(weights_dir, exception_message))
            fluid.io.load_vars(
                executor=exe,
                dirname=weights_dir,
                main_program=main_prog,
                vars=optimizer_var_list)

        if len(optimizer_var_list) == 0:
            raise Exception(
                "There is no optimizer parameters in the model, please set the optimizer!"
            )
        else:
            logging.info(
                "There are {} optimizer parameters in {} are loaded.".format(
                    len(optimizer_var_list), weights_dir))


class EarlyStop:
    def __init__(self, patience, thresh):
        self.patience = patience
        self.counter = 0
        self.score = None
        self.max = 0
        self.thresh = thresh
        if patience < 1:
            raise Exception("Argument patience should be a positive integer.")

    def __call__(self, current_score):
        if self.score is None:
            self.score = current_score
            return False
        elif current_score > self.max:
            self.counter = 0
            self.score = current_score
            self.max = current_score
            return False
        else:
            if (abs(self.score - current_score) < self.thresh
                    or current_score < self.score):
                self.counter += 1
                self.score = current_score
                logging.debug(
                    "EarlyStopping: %i / %i" % (self.counter, self.patience))
                if self.counter >= self.patience:
                    logging.info("EarlyStopping: Stop training")
                    return True
                return False
            else:
                self.counter = 0
                self.score = current_score
                return False
