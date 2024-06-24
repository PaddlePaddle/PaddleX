# !/usr/bin/env python3
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2024 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Author: PaddlePaddle Authors
"""


class ClsKeys(object):
    """
    This class defines a set of keys used for communication of Cls predictors
    and transforms. Both predictors and transforms accept a dict or a list of
    dicts as input, and they get the objects of their interest from the dict, or
    put the generated objects into the dict, all based on these keys.
    """

    # Common keys
    IMAGE = 'image'
    IM_PATH = 'input_path'
    # Suite-specific keys
    CLS_PRED = 'cls_pred'
    CLS_RESULT = 'cls_result'
