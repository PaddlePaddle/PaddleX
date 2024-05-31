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
from ..object_detection import DetEvaluator
from .support_models import SUPPORT_MODELS


class InstanceSegEvaluator(DetEvaluator):
    """ Instance Segmentation Model Evaluator """
    support_models = SUPPORT_MODELS

    def update_config(self):
        """update evalution config
        """
        if self.eval_config.log_interval:
            self.pdx_config.update_log_interval(self.eval_config.log_interval)
        self.pdx_config.update_dataset(self.global_config.dataset_dir,
                                       "COCOInstSegDataset")
        self.pdx_config.update_weights(self.eval_config.weight_path)
