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

import os

import numpy as np

from ....utils import logging
from ...base.predictor.transforms import image_common
from ...base import BasePredictor
from .keys import TableRecKeys as K
from . import transforms as T
from ..model_list import MODELS


class TableRecPredictor(BasePredictor):
    """ TableRecPredictor """
    entities = MODELS

    def __init__(self,
                 model_name,
                 model_dir,
                 kernel_option,
                 output,
                 pre_transforms=None,
                 post_transforms=None,
                 table_max_len=488):
        self.table_max_len = table_max_len
        super().__init__(
            model_name=model_name,
            model_dir=model_dir,
            kernel_option=kernel_option,
            output=output,
            pre_transforms=pre_transforms,
            post_transforms=post_transforms)

    @classmethod
    def get_input_keys(cls):
        """ get input keys """
        return [[K.IMAGE, K.ORI_IM_SIZE], [K.IM_PATH, K.ORI_IM_SIZE]]

    @classmethod
    def get_output_keys(cls):
        """ get output keys """
        return [K.STRUCTURE_PROB, K.LOC_PROB, K.SHAPE_LIST]

    def _run(self, batch_input):
        """ run """
        images = [data[K.IMAGE] for data in batch_input]
        input_ = np.stack(images, axis=0)
        if input_.ndim == 3:
            input_ = input_[:, np.newaxis]
        input_ = input_.astype(dtype=np.float32, copy=False)
        outputs = self._predictor.predict([input_])
        struc_probs = outputs[1]
        bbox_probs = outputs[0]
        for data in batch_input:
            data[K.SHAPE_LIST] = [data[K.ORI_IM_SIZE]]
        # In-place update
        pred = batch_input
        for dict_, struc_prob, bbox_prob in zip(pred, struc_probs, bbox_probs):
            dict_[K.STRUCTURE_PROB] = struc_prob[np.newaxis, :]
            dict_[K.LOC_PROB] = bbox_prob[np.newaxis, :]
        return pred

    def _get_pre_transforms_from_config(self):
        """ _get_pre_transforms_from_config """
        return [
            image_common.ReadImage(),
            image_common.ResizeByLong(target_long_edge=self.table_max_len),
            image_common.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            image_common.Pad(target_size=self.table_max_len, val=0.0),
            image_common.ToCHWImage()
        ]

    def _get_post_transforms_from_config(self):
        """get postprocess transforms"""
        post_transforms = [T.TableLabelDecode()]
        if not self.disable_save:
            post_transforms.append(T.SaveTableResults(self.output))
        return post_transforms
