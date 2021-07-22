# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import paddle
from paddle import is_compiled_with_cuda

from paddlex.ppcls.modeling import get_architectures
from paddlex.ppcls.modeling import similar_architectures
from paddlex.ppcls.modeling import get_blacklist_model_in_static_mode
from paddlex.ppcls.utils import logger


def check_version():
    """
    Log error and exit when the installed version of paddlepaddle is
    not satisfied.
    """
    err = "PaddlePaddle version 1.8.0 or higher is required, " \
          "or a suitable develop version is satisfied as well. \n" \
          "Please make sure the version is good with your code."
    try:
        pass
        # paddle.utils.require_version('0.0.0')
    except Exception:
        logger.error(err)
        sys.exit(1)


def check_gpu():
    """
    Log error and exit when using paddlepaddle cpu version.
    """
    err = "You are using paddlepaddle cpu version! Please try to " \
          "install paddlepaddle-gpu to run model on GPU."

    try:
        assert is_compiled_with_cuda()
    except AssertionError:
        logger.error(err)
        sys.exit(1)


def check_architecture(architecture):
    """
    check architecture and recommend similar architectures
    """
    assert isinstance(architecture, dict), \
        ("the type of architecture({}) should be dict". format(architecture))
    assert "name" in architecture, \
        ("name must be in the architecture keys, just contains: {}". format(
            architecture.keys()))

    similar_names = similar_architectures(architecture["name"],
                                          get_architectures())
    model_list = ', '.join(similar_names)
    err = "Architecture [{}] is not exist! Maybe you want: [{}]" \
          "".format(architecture["name"], model_list)
    try:
        assert architecture["name"] in similar_names
    except AssertionError:
        logger.error(err)
        sys.exit(1)


def check_model_with_running_mode(architecture):
    """
    check whether the model is consistent with the operating mode 
    """
    # some model are not supported in the static mode
    blacklist = get_blacklist_model_in_static_mode()
    if not paddle.in_dynamic_mode() and architecture["name"] in blacklist:
        logger.error("Model: {} is not supported in the staic mode.".format(
            architecture["name"]))
        sys.exit(1)
    return


def check_mix(architecture, use_mix=False):
    """
    check mix parameter
    """
    err = "Cannot use mix processing in GoogLeNet, " \
          "please set use_mix = False."
    try:
        if architecture["name"] == "GoogLeNet":
            assert use_mix is not True
    except AssertionError:
        logger.error(err)
        sys.exit(1)


def check_classes_num(classes_num):
    """
    check classes_num
    """
    err = "classes_num({}) should be a positive integer" \
        "and larger than 1".format(classes_num)
    try:
        assert isinstance(classes_num, int)
        assert classes_num > 1
    except AssertionError:
        logger.error(err)
        sys.exit(1)


def check_data_dir(path):
    """
    check cata_dir
    """
    err = "Data path is not exist, please given a right path" \
          "".format(path)
    try:
        assert os.isdir(path)
    except AssertionError:
        logger.error(err)
        sys.exit(1)


def check_function_params(config, key):
    """
    check specify config
    """
    k_config = config.get(key)
    assert k_config is not None, \
        ('{} is required in config'.format(key))

    assert k_config.get('function'), \
        ('function is required {} config'.format(key))
    params = k_config.get('params')
    assert params is not None, \
        ('params is required in {} config'.format(key))
    assert isinstance(params, dict), \
        ('the params in {} config should be a dict'.format(key))
