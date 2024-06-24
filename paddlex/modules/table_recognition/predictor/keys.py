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


class TableRecKeys(object):
    """
    This class defines a set of keys used for communication of TableRec predictors
    and transforms. Both predictors and transforms accept a dict or a list of
    dicts as input, and they get the objects of their interest from the dict, or
    put the generated objects into the dict, all based on these keys.
    """

    # Common keys
    IMAGE = 'image'
    IM_SIZE = 'image_size'
    IM_PATH = 'input_path'
    ORI_IM = 'original_image'
    ORI_IM_SIZE = 'original_image_size'
    # Suite-specific keys
    SHAPE_LIST = 'shape_list'
    STRUCTURE_PROB = 'structure_probs'
    LOC_PROB = 'loc_probs'
    BBOX_RES = 'cell_bbox'
    HTML_RES = 'html'
