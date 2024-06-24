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
from pathlib import Path
from ..base import BaseEvaluator
from .model_list import MODELS


class SegEvaluator(BaseEvaluator):
    """ Semantic Segmentation Model Evaluator """
    entities = MODELS

    def update_config(self):
        """update evalution config
        """
        self.pdx_config.update_dataset(self.global_config.dataset_dir,
                                       "SegDataset")
        self.pdx_config.update_pretrained_weights(None, is_backbone=True)

    def get_config_path(self, weight_path):
        """
        get config path

        Args:
            weight_path (str): The path to the weight

        Returns:
            config_path (str): The path to the config

        """

        config_path = Path(weight_path).parent.parent / "config.yaml"

        return config_path

    def get_eval_kwargs(self) -> dict:
        """get key-value arguments of model evalution function

        Returns:
            dict: the arguments of evaluation function.
        """
        return {
            "weight_path": self.eval_config.weight_path,
            "device": self.get_device(),
        }
