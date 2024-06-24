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


class InstanceSegKeys(object):
    """
    This class defines a set of keys used for communication of Det predictors
    and transforms. Both predictors and transforms accept a dict or a list of
    dicts as input, and they get the objects of their interest from the dict, or
    put the generated objects into the dict, all based on these keys.
    """

    # Common keys
    IMAGE = 'image'
    IM_PATH = 'input_path'
    IM_SIZE = 'image_size'
    SCALE_FACTOR = 'scale_factors'
    # Suite-specific keys
    BOXES = 'boxes'
    MASKS = 'masks'
