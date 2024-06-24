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
__all__ = [
    'build_pipeline', 'BasePipeline', 'OCRPipeline', 'ClsPipeline',
    'DetPipeline', 'InstanceSegPipeline', 'SegPipeline'
]

from .base import build_pipeline, BasePipeline
from .PPOCR import OCRPipeline
from .image_classification import ClsPipeline
from .object_detection import DetPipeline
from .instance_segmentation import InstanceSegPipeline
from .semantic_segmentation import SegPipeline
