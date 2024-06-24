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

from operator import le
import os

import numpy as np
from . import transforms as T
from ....utils import logging
from ...base import BasePredictor
from ...base.predictor.transforms import image_common
from .keys import TextDetKeys as K
from ..model_list import MODELS


class TextDetPredictor(BasePredictor):
    """ TextDetPredictor """
    entities = MODELS

    @classmethod
    def get_input_keys(cls):
        """ get input keys """
        return [[K.IMAGE], [K.IM_PATH]]

    @classmethod
    def get_output_keys(cls):
        """ get output keys """
        return [K.PROB_MAP, K.SHAPE]

    def _run(self, batch_input):
        """ _run """
        if len(batch_input) != 1:
            raise ValueError(
                f"For `{self.__class__.__name__}`, batch size can only be set to 1."
            )
        images = [data[K.IMAGE] for data in batch_input]
        input_ = np.stack(images, axis=0)
        if input_.ndim == 3:
            input_ = input_[:, np.newaxis]
        input_ = input_.astype(dtype=np.float32, copy=False)
        outputs = self._predictor.predict([input_])

        pred = batch_input
        pred[0][K.PROB_MAP] = outputs

        return pred

    def _get_pre_transforms_from_config(self):
        """ get preprocess transforms """
        return [
            image_common.ReadImage(), T.DetResizeForTest(
                limit_side_len=960, limit_type="max"), T.NormalizeImage(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    scale=1. / 255,
                    order='hwc'), image_common.ToCHWImage()
        ]

    def _get_post_transforms_from_config(self):
        """ get postprocess transforms """
        post_transforms = [
            T.DBPostProcess(
                thresh=0.3,
                box_thresh=0.6,
                max_candidates=1000,
                unclip_ratio=1.5,
                use_dilation=False,
                score_mode='fast',
                box_type='quad'), T.SaveTextDetResults(self.output),
            T.PrintResult()
        ]
        return post_transforms
