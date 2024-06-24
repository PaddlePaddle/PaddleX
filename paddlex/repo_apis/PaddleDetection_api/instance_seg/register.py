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
from .model import InstanceSegModel
from .config import InstanceSegConfig
from .runner import InstanceSegRunner

REPO_ROOT_PATH = os.environ.get('PADDLE_PDX_PADDLEDETECTION_PATH')
PDX_CONFIG_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', 'configs'))

register_suite_info({
    'suite_name': 'InstanceSeg',
    'model': InstanceSegModel,
    'runner': InstanceSegRunner,
    'config': InstanceSegConfig,
    'runner_root_path': REPO_ROOT_PATH
})

################ Models Using Universal Config ################
register_model_info({
    'model_name': 'Mask-RT-DETR-L',
    'suite': 'InstanceSeg',
    'config_path': osp.join(PDX_CONFIG_DIR, 'Mask-RT-DETR-L.yaml'),
    'supported_apis': ['train', 'evaluate', 'predict', 'export', 'infer'],
    'supported_dataset_types': ['COCOInstSegDataset'],
    'supported_train_opts': {
        'device': ['cpu', 'gpu_nxcx', 'xpu', 'npu', 'mlu'],
        'dy2st': False,
        'amp': ["OFF"]
    },
})

register_model_info({
    'model_name': 'Mask-RT-DETR-H',
    'suite': 'InstanceSeg',
    'config_path': osp.join(PDX_CONFIG_DIR, 'Mask-RT-DETR-H.yaml'),
    'supported_apis': ['train', 'evaluate', 'predict', 'export', 'infer'],
    'supported_dataset_types': ['COCOInstSegDataset'],
    'supported_train_opts': {
        'device': ['cpu', 'gpu_nxcx', 'xpu', 'npu', 'mlu'],
        'dy2st': False,
        'amp': ["OFF"]
    },
})
