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
from ..ts_base.runner import TSRunner
from .config import LongForecastConfig

REPO_ROOT_PATH = os.environ.get('PADDLE_PDX_PADDLETS_PATH')
PDX_CONFIG_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', 'configs'))

register_suite_info({
    'suite_name': 'LongForecast',
    'model': TSModel,
    'runner': TSRunner,
    'config': LongForecastConfig,
    'runner_root_path': REPO_ROOT_PATH
})

################ Models Using Universal Config ################

# DLinear
DLinear_CFG_PATH = osp.join(PDX_CONFIG_DIR, 'DLinear.yaml')
register_model_info({
    'model_name': 'DLinear',
    'suite': 'LongForecast',
    'config_path': DLinear_CFG_PATH,
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
        'device': ['cpu', 'gpu']
    },
})

DLinear_CFG_PATH = osp.join(PDX_CONFIG_DIR, 'RLinear.yaml')
register_model_info({
    'model_name': 'RLinear',
    'suite': 'LongForecast',
    'config_path': DLinear_CFG_PATH,
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

DLinear_CFG_PATH = osp.join(PDX_CONFIG_DIR, 'NLinear.yaml')
register_model_info({
    'model_name': 'NLinear',
    'suite': 'LongForecast',
    'config_path': DLinear_CFG_PATH,
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

# TiDE
TiDE_CFG_PATH = osp.join(PDX_CONFIG_DIR, 'TiDE.yaml')
register_model_info({
    'model_name': 'TiDE',
    'suite': 'LongForecast',
    'config_path': TiDE_CFG_PATH,
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

# PatchTST
PatchTST_CFG_PATH = osp.join(PDX_CONFIG_DIR, 'PatchTST.yaml')
register_model_info({
    'model_name': 'PatchTST',
    'suite': 'LongForecast',
    'config_path': PatchTST_CFG_PATH,
    'supported_apis': ['train', 'evaluate', 'predict'],
    'supported_train_opts': {
        'device': ['cpu', 'gpu_n1cx', 'xpu', 'npu', 'mlu'],
        'dy2st': False,
        'amp': []
    },
    'supported_evaluate_opts': {
        'device': ['cpu', 'gpu_n1cx'],
        'amp': []
    },
    'supported_predict_opts': {
        'device': ['cpu', 'gpu']
    },
    'supported_infer_opts': {
        'device': ['cpu', 'gpu']
    },
})

# Non-stationary
Nonstationary_CFG_PATH = osp.join(PDX_CONFIG_DIR, 'Nonstationary.yaml')
register_model_info({
    'model_name': 'Nonstationary',
    'suite': 'LongForecast',
    'config_path': Nonstationary_CFG_PATH,
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

# timesnet
TimesNet_CFG_PATH = osp.join(PDX_CONFIG_DIR, 'TimesNet.yaml')
register_model_info({
    'model_name': 'TimesNet',
    'suite': 'LongForecast',
    'config_path': TimesNet_CFG_PATH,
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