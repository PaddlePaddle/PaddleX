# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import errno
import os
import re
import shutil
import tempfile

import paddle
from paddle.static import load_program_state

from paddlex.ppcls.utils import logger

__all__ = ['init_model', 'save_model', 'load_dygraph_pretrain']


def _mkdir_if_not_exist(path):
    """
    mkdir if not exists, ignore the exception when multiprocess mkdir together
    """
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno == errno.EEXIST and os.path.isdir(path):
                logger.warning(
                    'be happy if some process has already created {}'.format(
                        path))
            else:
                raise OSError('Failed to mkdir {}'.format(path))


def load_dygraph_pretrain(model, path=None, load_static_weights=False):
    if not (os.path.isdir(path) or os.path.exists(path + '.pdparams')):
        raise ValueError("Model pretrain path {} does not "
                         "exists.".format(path))
    if load_static_weights:
        pre_state_dict = load_program_state(path)
        param_state_dict = {}
        model_dict = model.state_dict()
        for key in model_dict.keys():
            weight_name = model_dict[key].name
            if weight_name in pre_state_dict.keys():
                logger.info('Load weight: {}, shape: {}'.format(
                    weight_name, pre_state_dict[weight_name].shape))
                param_state_dict[key] = pre_state_dict[weight_name]
            else:
                param_state_dict[key] = model_dict[key]
        model.set_dict(param_state_dict)
        return

    param_state_dict = paddle.load(path + ".pdparams")
    model.set_dict(param_state_dict)
    return


def load_distillation_model(model, pretrained_model, load_static_weights):
    logger.info("In distillation mode, teacher model will be "
                "loaded firstly before student model.")

    if not isinstance(pretrained_model, list):
        pretrained_model = [pretrained_model]

    if not isinstance(load_static_weights, list):
        load_static_weights = [load_static_weights] * len(pretrained_model)

    teacher = model.teacher if hasattr(model,
                                       "teacher") else model._layers.teacher
    student = model.student if hasattr(model,
                                       "student") else model._layers.student
    load_dygraph_pretrain(
        teacher,
        path=pretrained_model[0],
        load_static_weights=load_static_weights[0])
    logger.info("Finish initing teacher model from {}".format(
        pretrained_model))
    # load student model
    if len(pretrained_model) >= 2:
        load_dygraph_pretrain(
            student,
            path=pretrained_model[1],
            load_static_weights=load_static_weights[1])
        logger.info("Finish initing student model from {}".format(
            pretrained_model))


def init_model(config, net, optimizer=None):
    """
    load model from checkpoint or pretrained_model
    """
    checkpoints = config.get('checkpoints')
    if checkpoints and optimizer is not None:
        assert os.path.exists(checkpoints + ".pdparams"), \
            "Given dir {}.pdparams not exist.".format(checkpoints)
        assert os.path.exists(checkpoints + ".pdopt"), \
            "Given dir {}.pdopt not exist.".format(checkpoints)
        para_dict = paddle.load(checkpoints + ".pdparams")
        opti_dict = paddle.load(checkpoints + ".pdopt")
        net.set_dict(para_dict)
        optimizer.set_state_dict(opti_dict)
        logger.info("Finish load checkpoints from {}".format(checkpoints))
        return

    pretrained_model = config.get('pretrained_model')
    load_static_weights = config.get('load_static_weights', False)
    use_distillation = config.get('use_distillation', False)
    if pretrained_model:
        if use_distillation:
            load_distillation_model(net, pretrained_model, load_static_weights)
        else:  # common load
            load_dygraph_pretrain(
                net,
                path=pretrained_model,
                load_static_weights=load_static_weights)
            logger.info(
                logger.coloring("Finish load pretrained model from {}".format(
                    pretrained_model), "HEADER"))


def _save_student_model(net, model_prefix):
    """
    save student model if the net is the network contains student
    """
    student_model_prefix = model_prefix + "_student.pdparams"
    if hasattr(net, "_layers"):
        net = net._layers
    if hasattr(net, "student"):
        paddle.save(net.student.state_dict(), student_model_prefix)
        logger.info("Already save student model in {}".format(
            student_model_prefix))


def save_model(net, optimizer, model_path, epoch_id, prefix='ppcls'):
    """
    save model to the target path
    """
    if paddle.distributed.get_rank() != 0:
        return
    model_path = os.path.join(model_path, str(epoch_id))
    _mkdir_if_not_exist(model_path)
    model_prefix = os.path.join(model_path, prefix)

    _save_student_model(net, model_prefix)

    paddle.save(net.state_dict(), model_prefix + ".pdparams")
    paddle.save(optimizer.state_dict(), model_prefix + ".pdopt")
    logger.info("Already save model in {}".format(model_path))
