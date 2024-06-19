# copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
from ...object_detection import DetPredictor
from .keys import InstanceSegKeys as K
from ..support_models import SUPPORT_MODELS


class InstanceSegPredictor(DetPredictor):
    """ Instance Seg Predictor """
    support_models = SUPPORT_MODELS

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
            [data[K.IMAGE_SHAPE][::-1] for data in batch_input], axis=0).astype(
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
