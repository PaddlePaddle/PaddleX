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
from .model import TextRecModel
from .runner import TextRecRunner
from .config import TextRecConfig

REPO_ROOT_PATH = os.environ.get('PADDLE_PDX_PADDLEOCR_PATH')
PDX_CONFIG_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', 'configs'))

register_suite_info({
    'suite_name': 'TextRec',
    'model': TextRecModel,
    'runner': TextRecRunner,
    'config': TextRecConfig,
    'runner_root_path': REPO_ROOT_PATH
})

register_model_info({
    'model_name': 'PP-OCRv4_mobile_rec',
    'suite': 'TextRec',
    'config_path': osp.join(PDX_CONFIG_DIR, 'PP-OCRv4_mobile_rec.yaml'),
    'supported_apis': ['train', 'evaluate', 'predict', 'export', 'infer']
})

register_model_info({
    'model_name': 'PP-OCRv4_server_rec',
    'suite': 'TextRec',
    'config_path': osp.join(PDX_CONFIG_DIR, 'PP-OCRv4_server_rec.yaml'),
    'supported_apis': ['train', 'evaluate', 'predict', 'export', 'infer']
})
