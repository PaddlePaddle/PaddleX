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
from ..ts_base.model import TSModel
from .runner import TSCLSRunner
from .config import TSClassifyConfig

REPO_ROOT_PATH = os.environ.get('PADDLE_PDX_PADDLETS_PATH')
PDX_CONFIG_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', 'configs'))

register_suite_info({
    'suite_name': 'TSClassify',
    'model': TSModel,
    'runner': TSCLSRunner,
    'config': TSClassifyConfig,
    'runner_root_path': REPO_ROOT_PATH
})

################ Models Using Universal Config ################

# timesnet
TimesNetCLS_CFG_PATH = osp.join(PDX_CONFIG_DIR, 'TimesNet_cls.yaml')
register_model_info({
    'model_name': 'TimesNet_cls',
    'suite': 'TSClassify',
    'config_path': TimesNetCLS_CFG_PATH,
    'supported_apis': ['train', 'evaluate', 'predict'],
    'supported_train_opts': {
        'device': ['cpu', 'gpu_n1cx', 'xpu', 'npu', 'mlu'],
        'dy2st': False,
        'amp': []
    },
    'supported_evaluate_opts': {
        'device': ['cpu', 'gpu_n1cx', 'xpu', 'npu', 'mlu'],
        'amp': []
    },
    'supported_predict_opts': {
        'device': ['cpu', 'gpu', 'xpu', 'npu', 'mlu']
    },
    'supported_infer_opts': {
        'device': ['cpu', 'gpu', 'xpu', 'npu', 'mlu']
    },
})
