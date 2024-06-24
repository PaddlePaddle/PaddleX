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
from ..base import BasePipeline
from ...modules.instance_segmentation.model_list import MODELS
from ...modules import create_model, PaddleInferenceOption
from ...modules.object_detection import transforms as T


class InstanceSegPipeline(BasePipeline):
    """InstanceSeg Pipeline
    """
    entities = "instance_segmentation"

    def __init__(self,
                 model_name=None,
                 model_dir=None,
                 output="./output",
                 kernel_option=None,
                 device="gpu",
                 **kwargs):
        self.model_name = model_name
        self.model_dir = model_dir
        self.output = output
        self.device = device
        self.kernel_option = kernel_option
        if self.model_name is not None:
            self.load_model()

    def check_model_name(self):
        """ check that model name is valid
        """
        assert self.model_name in MODELS, f"The model name({self.model_name}) error. Only support: {MODELS}."

    def load_model(self):
        """load model predictor
        """
        self.check_model_name()
        kernel_option = self.get_kernel_option(
        ) if self.kernel_option is None else self.kernel_option
        self.model = create_model(
            model_name=self.model_name,
            model_dir=self.model_dir,
            output=self.output,
            kernel_option=kernel_option)

    def predict(self, input):
        """predict
        """
        return self.model.predict(input)

    def get_kernel_option(self):
        """get kernel option
        """
        kernel_option = PaddleInferenceOption()
        kernel_option.set_device(self.device)
        return kernel_option

    def update_model_name(self, model_name_list):
        """update model name and re

        Args:
            model_list (list): list of model name.
        """
        assert len(model_name_list) == 1
        self.model_name = model_name_list[0]

    def get_input_keys(self):
        """get dict keys of input argument input
        """
        return self.model.get_input_keys()
