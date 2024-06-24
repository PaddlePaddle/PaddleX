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

import os
import numpy as np
from pathlib import Path

from ...base import BasePredictor
from ...base.predictor.transforms import image_common
from .keys import ClsKeys as K
from .utils import InnerConfig
from ....utils import logging
from . import transforms as T
from ..model_list import MODELS


class ClsPredictor(BasePredictor):
    """ Clssification Predictor """
    entities = MODELS

    def load_other_src(self):
        """ load the inner config file """
        infer_cfg_file_path = os.path.join(self.model_dir, 'inference.yml')
        if not os.path.exists(infer_cfg_file_path):
            raise FileNotFoundError(
                f"Cannot find config file: {infer_cfg_file_path}")
        return InnerConfig(infer_cfg_file_path)

    @classmethod
    def get_input_keys(cls):
        """ get input keys """
        return [[K.IMAGE], [K.IM_PATH]]

    @classmethod
    def get_output_keys(cls):
        """ get output keys """
        return [K.CLS_PRED]

    def _run(self, batch_input):
        """ run """
        input_dict = {}
        input_dict[K.IMAGE] = np.stack(
            [data[K.IMAGE] for data in batch_input], axis=0).astype(
                dtype=np.float32, copy=False)
        input_ = [input_dict[K.IMAGE]]
        outputs = self._predictor.predict(input_)
        cls_outs = outputs[0]
        # In-place update
        pred = batch_input
        for dict_, cls_out in zip(pred, cls_outs):
            dict_[K.CLS_PRED] = cls_out
        return pred

    def _get_pre_transforms_from_config(self):
        """ get preprocess transforms """
        logging.info(
            f"Transformation operators for data preprocessing will be inferred from config file."
        )
        pre_transforms = self.other_src.pre_transforms
        pre_transforms.insert(0, image_common.ReadImage(format='RGB'))
        return pre_transforms

    def _get_post_transforms_from_config(self):
        """ get postprocess transforms """
        post_transforms = self.other_src.post_transforms
        post_transforms.extend([
            T.PrintResult(), T.SaveClsResults(self.output,
                                              self.other_src.labels)
        ])
        return post_transforms
