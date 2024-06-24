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
from .base import build_dataset_checker, build_trainer, build_evaluater, build_predictor, create_model, \
PaddleInferenceOption
from .image_classification import ClsDatasetChecker, ClsTrainer, ClsEvaluator, ClsPredictor
from .object_detection import COCODatasetChecker, DetTrainer, DetEvaluator, DetPredictor
from .text_detection import TextDetDatasetChecker, TextDetTrainer, TextDetEvaluator, TextDetPredictor
from .text_recognition import TextRecDatasetChecker, TextRecTrainer, TextRecEvaluator, TextRecPredictor
from .table_recognition import TableRecDatasetChecker, TableRecTrainer, TableRecEvaluator, TableRecPredictor
from .semantic_segmentation import SegDatasetChecker, SegTrainer, SegEvaluator, SegPredictor
from .instance_segmentation import COCOInstSegDatasetChecker, InstanceSegTrainer, InstanceSegEvaluator, \
InstanceSegPredictor
from .ts_anomaly_detection import TSADDatasetChecker, TSADTrainer, TSADEvaluator, TSADPredictor
from .ts_classification import TSCLSDatasetChecker, TSCLSTrainer, TSCLSEvaluator, TSCLSPredictor
from .ts_forecast import TSFCDatasetChecker, TSFCTrainer, TSFCEvaluator, TSFCPredictor

from .base.predictor.transforms import image_common
from .image_classification import transforms as cls_transforms
from .object_detection import transforms as det_transforms
from .text_detection import transforms as text_det_transforms
from .text_recognition import transforms as text_rec_transforms
from .table_recognition import transforms as table_rec_transforms
from .semantic_segmentation import transforms as seg_transforms
from .instance_segmentation import transforms as instance_seg_transforms
