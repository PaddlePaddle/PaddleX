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
from .image_classification import ClsDatasetChecker, ClsTrainer, ClsEvaluator
from .object_detection import COCODatasetChecker, DetTrainer, DetEvaluator
from .text_detection import TextDetDatasetChecker, TextDetTrainer, TextDetEvaluator
from .text_recognition import TextRecDatasetChecker, TextRecTrainer, TextRecEvaluator
from .table_recognition import TableRecDatasetChecker, TableRecTrainer, TableRecEvaluator
from .semantic_segmentation import SegDatasetChecker, SegTrainer, SegEvaluator
from .instance_segmentation import COCOInstSegDatasetChecker, InstanceSegTrainer, InstanceSegEvaluator
from .ts_anomaly_detection import TSADDatasetChecker, TSADTrainer, TSADEvaluator
from .ts_classification import TSCLSDatasetChecker, TSCLSTrainer, TSCLSEvaluator
from .ts_forecast import TSFCDatasetChecker, TSFCTrainer, TSFCEvaluator
