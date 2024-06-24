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
from .model import TableRecModel
from .runner import TableRecRunner
from .config import TableRecConfig

REPO_ROOT_PATH = os.environ.get('PADDLE_PDX_PADDLEOCR_PATH')
PDX_CONFIG_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', 'configs'))

register_suite_info({
    'suite_name': 'TableRec',
    'model': TableRecModel,
    'runner': TableRecRunner,
    'config': TableRecConfig,
    'runner_root_path': REPO_ROOT_PATH
})

register_model_info({
    'model_name': 'SLANet',
    'suite': 'TableRec',
    'config_path': osp.join(PDX_CONFIG_DIR, 'SLANet.yaml'),
    'supported_apis': ['train', 'evaluate', 'predict', 'export', 'infer']
})
