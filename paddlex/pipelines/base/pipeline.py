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
from abc import ABC, abstractmethod

from ...utils.misc import AutoRegisterABCMetaClass


def build_pipeline(
        pipeline_name: str,
        model_list: list,
        output: str,
        device: str, ) -> "BasePipeline":
    """build model evaluater

    Args:
        pipeline_name (str): the pipeline name, that is name of pipeline class

    Returns:
        BasePipeline: the pipeline, which is subclass of BasePipeline.
    """
    pipeline = BasePipeline.get(pipeline_name)(output=output, device=device)
    pipeline.update_model_name(model_list)
    pipeline.load_model()
    return pipeline


class BasePipeline(ABC, metaclass=AutoRegisterABCMetaClass):
    """Base Pipeline
    """
    __is_base = True

    def __init__(self):
        super().__init__()

    @abstractmethod
    def load_model(self):
        """load model predictor
        """
        raise NotImplementedError

    @abstractmethod
    def update_model_name(self, model_list: list) -> dict:
        """update model name and re

        Args:
            model_list (list): list of model name.
        """
        raise NotImplementedError

    @abstractmethod
    def get_input_keys(self):
        """get dict keys of input argument input
        """
        raise NotImplementedError
