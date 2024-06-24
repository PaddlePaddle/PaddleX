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
from .model import ClsModel
from .runner import ClsRunner
from .config import ClsConfig

REPO_ROOT_PATH = os.environ.get('PADDLE_PDX_PADDLECLAS_PATH')
PDX_CONFIG_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', 'configs'))

register_suite_info({
    'suite_name': 'Cls',
    'model': ClsModel,
    'runner': ClsRunner,
    'config': ClsConfig,
    'runner_root_path': REPO_ROOT_PATH
})

################ Models Using Universal Config ################
register_model_info({
    'model_name': 'SwinTransformer_base_patch4_window7_224',
    'suite': 'Cls',
    'config_path': osp.join(PDX_CONFIG_DIR,
                            'SwinTransformer_base_patch4_window7_224.yaml'),
    'supported_apis': ['train', 'evaluate', 'predict', 'export', 'infer'],
    'infer_config': 'deploy/configs/inference_cls.yaml'
})

register_model_info({
    'model_name': 'PP-LCNet_x0_25',
    'suite': 'Cls',
    'config_path': osp.join(PDX_CONFIG_DIR, 'PP-LCNet_x0_25.yaml'),
    'supported_apis': ['train', 'evaluate', 'predict', 'export', 'infer'],
    'infer_config': 'deploy/configs/inference_cls.yaml'
})

register_model_info({
    'model_name': 'PP-LCNet_x0_35',
    'suite': 'Cls',
    'config_path': osp.join(PDX_CONFIG_DIR, 'PP-LCNet_x0_35.yaml'),
    'supported_apis': ['train', 'evaluate', 'predict', 'export', 'infer'],
    'infer_config': 'deploy/configs/inference_cls.yaml'
})

register_model_info({
    'model_name': 'PP-LCNet_x0_5',
    'suite': 'Cls',
    'config_path': osp.join(PDX_CONFIG_DIR, 'PP-LCNet_x0_5.yaml'),
    'supported_apis': ['train', 'evaluate', 'predict', 'export', 'infer'],
    'infer_config': 'deploy/configs/inference_cls.yaml'
})

register_model_info({
    'model_name': 'PP-LCNet_x0_75',
    'suite': 'Cls',
    'config_path': osp.join(PDX_CONFIG_DIR, 'PP-LCNet_x0_75.yaml'),
    'supported_apis': ['train', 'evaluate', 'predict', 'export', 'infer'],
    'infer_config': 'deploy/configs/inference_cls.yaml'
})

register_model_info({
    'model_name': 'PP-LCNet_x1_0',
    'suite': 'Cls',
    'config_path': osp.join(PDX_CONFIG_DIR, 'PP-LCNet_x1_0.yaml'),
    'supported_apis': ['train', 'evaluate', 'predict', 'export', 'infer'],
    'infer_config': 'deploy/configs/inference_cls.yaml'
})

register_model_info({
    'model_name': 'PP-LCNet_x1_5',
    'suite': 'Cls',
    'config_path': osp.join(PDX_CONFIG_DIR, 'PP-LCNet_x1_5.yaml'),
    'supported_apis': ['train', 'evaluate', 'predict', 'export', 'infer'],
    'infer_config': 'deploy/configs/inference_cls.yaml'
})

register_model_info({
    'model_name': 'PP-LCNet_x2_0',
    'suite': 'Cls',
    'config_path': osp.join(PDX_CONFIG_DIR, 'PP-LCNet_x2_0.yaml'),
    'supported_apis': ['train', 'evaluate', 'predict', 'export', 'infer'],
    'infer_config': 'deploy/configs/inference_cls.yaml'
})

register_model_info({
    'model_name': 'PP-LCNet_x2_5',
    'suite': 'Cls',
    'config_path': osp.join(PDX_CONFIG_DIR, 'PP-LCNet_x2_5.yaml'),
    'supported_apis': ['train', 'evaluate', 'predict', 'export', 'infer'],
    'infer_config': 'deploy/configs/inference_cls.yaml'
})

register_model_info({
    'model_name': 'CLIP_vit_base_patch16_224',
    'suite': 'Cls',
    'config_path': osp.join(PDX_CONFIG_DIR, 'CLIP_vit_base_patch16_224.yaml'),
    'supported_apis': ['train', 'evaluate', 'predict', 'export', 'infer'],
    'infer_config': 'deploy/configs/inference_cls.yaml'
})

register_model_info({
    'model_name': 'CLIP_vit_large_patch14_224',
    'suite': 'Cls',
    'config_path': osp.join(PDX_CONFIG_DIR, 'CLIP_vit_large_patch14_224.yaml'),
    'supported_apis': ['train', 'evaluate', 'predict', 'export', 'infer'],
    'infer_config': 'deploy/configs/inference_cls.yaml'
})

register_model_info({
    'model_name': 'PP-HGNet_small',
    'suite': 'Cls',
    'config_path': osp.join(PDX_CONFIG_DIR, 'PP-HGNet_small.yaml'),
    'supported_apis': ['train', 'evaluate', 'predict', 'export', 'infer'],
    'infer_config': 'deploy/configs/inference_cls.yaml'
})

register_model_info({
    'model_name': 'PP-HGNetV2-B0',
    'suite': 'Cls',
    'config_path': osp.join(PDX_CONFIG_DIR, 'PP-HGNetV2-B0.yaml'),
    'supported_apis': ['train', 'evaluate', 'predict', 'export', 'infer'],
    'infer_config': 'deploy/configs/inference_cls.yaml'
})

register_model_info({
    'model_name': 'PP-HGNetV2-B4',
    'suite': 'Cls',
    'config_path': osp.join(PDX_CONFIG_DIR, 'PP-HGNetV2-B4.yaml'),
    'supported_apis': ['train', 'evaluate', 'predict', 'export', 'infer'],
    'infer_config': 'deploy/configs/inference_cls.yaml'
})

register_model_info({
    'model_name': 'PP-HGNetV2-B6',
    'suite': 'Cls',
    'config_path': osp.join(PDX_CONFIG_DIR, 'PP-HGNetV2-B6.yaml'),
    'supported_apis': ['train', 'evaluate', 'predict', 'export', 'infer'],
    'infer_config': 'deploy/configs/inference_cls.yaml'
})

register_model_info({
    'model_name': 'ResNet18',
    'suite': 'Cls',
    'config_path': osp.join(PDX_CONFIG_DIR, 'ResNet18.yaml'),
    'supported_apis': ['train', 'evaluate', 'predict', 'export', 'infer'],
    'infer_config': 'deploy/configs/inference_cls.yaml'
})

register_model_info({
    'model_name': 'ResNet34',
    'suite': 'Cls',
    'config_path': osp.join(PDX_CONFIG_DIR, 'ResNet34.yaml'),
    'supported_apis': ['train', 'evaluate', 'predict', 'export', 'infer'],
    'infer_config': 'deploy/configs/inference_cls.yaml'
})

register_model_info({
    'model_name': 'ResNet50',
    'suite': 'Cls',
    'config_path': osp.join(PDX_CONFIG_DIR, 'ResNet50.yaml'),
    'supported_apis': ['train', 'evaluate', 'predict', 'export', 'infer'],
    'infer_config': 'deploy/configs/inference_cls.yaml'
})

register_model_info({
    'model_name': 'ResNet101',
    'suite': 'Cls',
    'config_path': osp.join(PDX_CONFIG_DIR, 'ResNet101.yaml'),
    'supported_apis': ['train', 'evaluate', 'predict', 'export', 'infer'],
    'infer_config': 'deploy/configs/inference_cls.yaml'
})

register_model_info({
    'model_name': 'ResNet152',
    'suite': 'Cls',
    'config_path': osp.join(PDX_CONFIG_DIR, 'ResNet152.yaml'),
    'supported_apis': ['train', 'evaluate', 'predict', 'export', 'infer'],
    'infer_config': 'deploy/configs/inference_cls.yaml'
})

register_model_info({
    'model_name': 'MobileNetV2_x0_25',
    'suite': 'Cls',
    'config_path': osp.join(PDX_CONFIG_DIR, 'MobileNetV2_x0_25.yaml'),
    'supported_apis': ['train', 'evaluate', 'predict', 'export', 'infer'],
    'infer_config': 'deploy/configs/inference_cls.yaml'
})

register_model_info({
    'model_name': 'MobileNetV2_x0_5',
    'suite': 'Cls',
    'config_path': osp.join(PDX_CONFIG_DIR, 'MobileNetV2_x0_5.yaml'),
    'supported_apis': ['train', 'evaluate', 'predict', 'export', 'infer'],
    'infer_config': 'deploy/configs/inference_cls.yaml'
})

register_model_info({
    'model_name': 'MobileNetV2_x1_0',
    'suite': 'Cls',
    'config_path': osp.join(PDX_CONFIG_DIR, 'MobileNetV2_x1_0.yaml'),
    'supported_apis': ['train', 'evaluate', 'predict', 'export', 'infer'],
    'infer_config': 'deploy/configs/inference_cls.yaml'
})

register_model_info({
    'model_name': 'MobileNetV2_x1_5',
    'suite': 'Cls',
    'config_path': osp.join(PDX_CONFIG_DIR, 'MobileNetV2_x1_5.yaml'),
    'supported_apis': ['train', 'evaluate', 'predict', 'export', 'infer'],
    'infer_config': 'deploy/configs/inference_cls.yaml'
})

register_model_info({
    'model_name': 'MobileNetV2_x2_0',
    'suite': 'Cls',
    'config_path': osp.join(PDX_CONFIG_DIR, 'MobileNetV2_x2_0.yaml'),
    'supported_apis': ['train', 'evaluate', 'predict', 'export', 'infer'],
    'infer_config': 'deploy/configs/inference_cls.yaml'
})

register_model_info({
    'model_name': 'MobileNetV3_large_x0_35',
    'suite': 'Cls',
    'config_path': osp.join(PDX_CONFIG_DIR, 'MobileNetV3_large_x0_35.yaml'),
    'supported_apis': ['train', 'evaluate', 'predict', 'export', 'infer'],
    'infer_config': 'deploy/configs/inference_cls.yaml'
})

register_model_info({
    'model_name': 'MobileNetV3_large_x0_5',
    'suite': 'Cls',
    'config_path': osp.join(PDX_CONFIG_DIR, 'MobileNetV3_large_x0_5.yaml'),
    'supported_apis': ['train', 'evaluate', 'predict', 'export', 'infer'],
    'infer_config': 'deploy/configs/inference_cls.yaml'
})

register_model_info({
    'model_name': 'MobileNetV3_large_x0_75',
    'suite': 'Cls',
    'config_path': osp.join(PDX_CONFIG_DIR, 'MobileNetV3_large_x0_75.yaml'),
    'supported_apis': ['train', 'evaluate', 'predict', 'export', 'infer'],
    'infer_config': 'deploy/configs/inference_cls.yaml'
})

register_model_info({
    'model_name': 'MobileNetV3_large_x1_0',
    'suite': 'Cls',
    'config_path': osp.join(PDX_CONFIG_DIR, 'MobileNetV3_large_x1_0.yaml'),
    'supported_apis': ['train', 'evaluate', 'predict', 'export', 'infer'],
    'infer_config': 'deploy/configs/inference_cls.yaml'
})

register_model_info({
    'model_name': 'MobileNetV3_large_x1_25',
    'suite': 'Cls',
    'config_path': osp.join(PDX_CONFIG_DIR, 'MobileNetV3_large_x1_25.yaml'),
    'supported_apis': ['train', 'evaluate', 'predict', 'export', 'infer'],
    'infer_config': 'deploy/configs/inference_cls.yaml'
})

register_model_info({
    'model_name': 'MobileNetV3_small_x0_35',
    'suite': 'Cls',
    'config_path': osp.join(PDX_CONFIG_DIR, 'MobileNetV3_small_x0_35.yaml'),
    'supported_apis': ['train', 'evaluate', 'predict', 'export', 'infer'],
    'infer_config': 'deploy/configs/inference_cls.yaml'
})

register_model_info({
    'model_name': 'MobileNetV3_small_x0_5',
    'suite': 'Cls',
    'config_path': osp.join(PDX_CONFIG_DIR, 'MobileNetV3_small_x0_5.yaml'),
    'supported_apis': ['train', 'evaluate', 'predict', 'export', 'infer'],
    'infer_config': 'deploy/configs/inference_cls.yaml'
})

register_model_info({
    'model_name': 'MobileNetV3_small_x0_75',
    'suite': 'Cls',
    'config_path': osp.join(PDX_CONFIG_DIR, 'MobileNetV3_small_x0_75.yaml'),
    'supported_apis': ['train', 'evaluate', 'predict', 'export', 'infer'],
    'infer_config': 'deploy/configs/inference_cls.yaml'
})

register_model_info({
    'model_name': 'MobileNetV3_small_x1_0',
    'suite': 'Cls',
    'config_path': osp.join(PDX_CONFIG_DIR, 'MobileNetV3_small_x1_0.yaml'),
    'supported_apis': ['train', 'evaluate', 'predict', 'export', 'infer'],
    'infer_config': 'deploy/configs/inference_cls.yaml'
})

register_model_info({
    'model_name': 'MobileNetV3_small_x1_25',
    'suite': 'Cls',
    'config_path': osp.join(PDX_CONFIG_DIR, 'MobileNetV3_small_x1_25.yaml'),
    'supported_apis': ['train', 'evaluate', 'predict', 'export', 'infer'],
    'infer_config': 'deploy/configs/inference_cls.yaml'
})

register_model_info({
    'model_name': 'ConvNeXt_tiny',
    'suite': 'Cls',
    'config_path': osp.join(PDX_CONFIG_DIR, 'ConvNeXt_tiny.yaml'),
    'supported_apis': ['train', 'evaluate', 'predict', 'export', 'infer'],
    'infer_config': 'deploy/configs/inference_cls.yaml'
})
