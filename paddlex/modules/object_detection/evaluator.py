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
from ..base.evaluator import BaseEvaluator
from .support_models import SUPPORT_MODELS


class DetEvaluator(BaseEvaluator):
    """ Object Detection Model Evaluator """
    support_models = SUPPORT_MODELS

    def update_config(self):
        """update evalution config
        """
        if self.eval_config.log_interval:
            self.pdx_config.update_log_interval(self.eval_config.log_interval)
        self.pdx_config.update_dataset(self.global_config.dataset_dir,
                                       "COCODetDataset")
        self.pdx_config.update_weights(self.eval_config.weight_path)

    def get_eval_kwargs(self) -> dict:
        """get key-value arguments of model evalution function

        Returns:
            dict: the arguments of evaluation function.
        """
        return {
            "weight_path": self.eval_config.weight_path,
            "device": self.get_device(using_gpu_number=1)
        }
