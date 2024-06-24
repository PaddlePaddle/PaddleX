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


class TextDetKeys(object):
    """
    This class defines a set of keys used for communication of TextDet predictors
    and transforms. Both predictors and transforms accept a dict or a list of
    dicts as input, and they get the objects of their interest from the dict, or
    put the generated objects into the dict, all based on these keys.
    """

    # Common keys
    IMAGE = 'image'
    IM_PATH = 'input_path'
    IM_SIZE = 'original_image_size'
    ORI_IM = 'original_image'
    SHAPE = 'shape'
    PROB_MAP = 'prob_map'
    # Suite-specific keys
    DT_SCORES = 'dt_scores'
    DT_POLYS = 'dt_polys'
    SUB_IMGS = 'sub_imgs'