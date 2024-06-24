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
from ..base import BaseEvaluator
from .model_list import MODELS


class TableRecEvaluator(BaseEvaluator):
    """ Table Recognition Model Evaluator """
    entities = MODELS

    def update_config(self):
        """update evalution config
        """
        if self.eval_config.log_interval:
            self.pdx_config.update_log_interval(self.eval_config.log_interval)

        self.pdx_config.update_dataset(self.global_config.dataset_dir,
                                       "PubTabTableRecDataset")

    def get_eval_kwargs(self) -> dict:
        """get key-value arguments of model evalution function

        Returns:
            dict: the arguments of evaluation function.
        """
        return {
            "weight_path": self.eval_config.weight_path,
            "device": self.get_device()
        }
