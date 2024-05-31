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
import os
import os.path as osp

from ...base.register import register_model_info, register_suite_info
from .model import TextDetModel
from .runner import TextDetRunner
from .config import TextDetConfig

REPO_ROOT_PATH = os.environ.get('PADDLE_PDX_PADDLEOCR_PATH')
PDX_CONFIG_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', 'configs'))

register_suite_info({
    'suite_name': 'TextDet',
    'model': TextDetModel,
    'runner': TextDetRunner,
    'config': TextDetConfig,
    'runner_root_path': REPO_ROOT_PATH
})

################ Models Using Universal Config ################
register_model_info({
    'model_name': 'PP-OCRv4_mobile_det',
    'suite': 'TextDet',
    'config_path': osp.join(PDX_CONFIG_DIR, 'PP-OCRv4_mobile_det.yaml'),
    'supported_apis': ['train', 'evaluate', 'predict', 'export', 'infer']
})

register_model_info({
    'model_name': 'PP-OCRv4_server_det',
    'suite': 'TextDet',
    'config_path': osp.join(PDX_CONFIG_DIR, 'PP-OCRv4_server_det.yaml'),
    'supported_apis': ['train', 'evaluate', 'predict', 'export', 'infer']
})