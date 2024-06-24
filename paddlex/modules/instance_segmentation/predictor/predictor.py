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
import numpy as np
from ...object_detection import DetPredictor
from .keys import InstanceSegKeys as K
from ..model_list import MODELS


class InstanceSegPredictor(DetPredictor):
    """ Instance Seg Predictor """
    entities = MODELS

    def _run(self, batch_input):
        """ run """
        input_dict = {}
        input_dict["image"] = np.stack(
            [data[K.IMAGE] for data in batch_input], axis=0).astype(
                dtype=np.float32, copy=False)
        input_dict["scale_factor"] = np.stack(
            [data[K.SCALE_FACTOR][::-1] for data in batch_input],
            axis=0).astype(
                dtype=np.float32, copy=False)
        input_dict["im_shape"] = np.stack(
            [data[K.IM_SIZE][::-1] for data in batch_input], axis=0).astype(
                dtype=np.float32, copy=False)

        input_ = [input_dict[i] for i in self._predictor.get_input_names()]

        batch_np_boxes, batch_np_boxes_num, batch_np_masks = self._predictor.predict(
            input_)

        pred = batch_input
        box_idx_start = 0
        for idx in range(len(batch_input)):
            np_boxes_num = batch_np_boxes_num[idx]
            box_idx_end = box_idx_start + np_boxes_num
            np_boxes = batch_np_boxes[box_idx_start:box_idx_end]
            np_masks = batch_np_masks[box_idx_start:box_idx_end]
            box_idx_start = box_idx_end

            batch_input[idx][K.BOXES] = np_boxes
            batch_input[idx][K.MASKS] = np_masks
        return pred

    @classmethod
    def get_output_keys(cls):
        """ get output keys """
        return [K.BOXES, K.MASKS]
