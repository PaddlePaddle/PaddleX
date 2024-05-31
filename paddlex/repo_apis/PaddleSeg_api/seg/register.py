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
from .model import SegModel
from .runner import SegRunner
from .config import SegConfig

REPO_ROOT_PATH = os.environ.get('PADDLE_PDX_PADDLESEG_PATH')
PDX_CONFIG_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', 'configs'))

register_suite_info({
    'suite_name': 'Seg',
    'model': SegModel,
    'runner': SegRunner,
    'config': SegConfig,
    'runner_root_path': REPO_ROOT_PATH
})

################ Models Using Universal Config ################
# OCRNet
register_model_info({
    'model_name': 'OCRNet_HRNet-W48',
    'suite': 'Seg',
    'config_path': osp.join(PDX_CONFIG_DIR, 'OCRNet_HRNet-W48.yaml'),
    'supported_apis': ['train', 'evaluate', 'predict', 'export', 'infer']
})

# PP-LiteSeg
register_model_info({
    'model_name': 'PP-LiteSeg-T',
    'suite': 'Seg',
    'config_path': osp.join(PDX_CONFIG_DIR, 'PP-LiteSeg-T.yaml'),
    'supported_apis': ['train', 'evaluate', 'predict', 'export', 'infer'],
    'supported_train_opts': {
        'device': ['cpu', 'gpu_nxcx', 'xpu', 'npu', 'mlu'],
        'dy2st': True,
        'amp': ['O1', 'O2']
    },
    'supported_evaluate_opts': {
        'device': ['cpu', 'gpu_nxcx', 'xpu', 'npu', 'mlu'],
        'amp': []
    },
    'supported_predict_opts': {
        'device': ['cpu', 'gpu', 'xpu', 'npu', 'mlu']
    },
    'supported_infer_opts': {
        'device': ['cpu', 'gpu', 'xpu', 'npu', 'mlu']
    },
    'supported_dataset_types': []
})

# OCRNet
register_model_info({
    'model_name': 'Deeplabv3-R50',
    'suite': 'Seg',
    'config_path': osp.join(PDX_CONFIG_DIR, 'Deeplabv3-R50.yaml'),
    'supported_apis': ['train', 'evaluate', 'predict', 'export', 'infer']
})

register_model_info({
    'model_name': 'Deeplabv3-R101',
    'suite': 'Seg',
    'config_path': osp.join(PDX_CONFIG_DIR, 'Deeplabv3-R101.yaml'),
    'supported_apis': ['train', 'evaluate', 'predict', 'export', 'infer']
})

register_model_info({
    'model_name': 'Deeplabv3_Plus-R50',
    'suite': 'Seg',
    'config_path': osp.join(PDX_CONFIG_DIR, 'Deeplabv3_Plus-R50.yaml'),
    'supported_apis': ['train', 'evaluate', 'predict', 'export', 'infer']
})

register_model_info({
    'model_name': 'Deeplabv3_Plus-R101',
    'suite': 'Seg',
    'config_path': osp.join(PDX_CONFIG_DIR, 'Deeplabv3_Plus-R101.yaml'),
    'supported_apis': ['train', 'evaluate', 'predict', 'export', 'infer']
})


# For compatibility
def _set_alias(model_name, alias):
    from ...base.register import get_registered_model_info

    record = get_registered_model_info(model_name)
    record = dict(**record)
    record['model_name'] = alias
    register_model_info(record)


_set_alias('OCRNet_HRNet-W48', 'ocrnet_hrnetw48')
_set_alias('PP-LiteSeg-T', 'pp_liteseg_stdc1')
