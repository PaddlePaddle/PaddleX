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



from operator import le
import os

import numpy as np
from . import transforms as T
from ....utils import logging
from ...base import BasePredictor
from ...base.predictor.transforms import image_common
from .keys import TextDetKeys as K
from ..support_models import SUPPORT_MODELS


class TextDetPredictor(BasePredictor):
    """ TextDetPredictor """
    support_models = SUPPORT_MODELS

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

    def _get_pre_transforms_for_data(self, data):
        """ get preprocess transforms """
        if K.IMAGE not in data and K.IM_PATH not in data:
            raise KeyError(
                f"Key {repr(K.IMAGE)} or {repr(K.IM_PATH)} is required, but not found."
            )
        pre_transforms = []
        if K.IMAGE not in data:
            pre_transforms.append(image_common.ReadImage())
        pre_transforms.append(
            T.DetResizeForTest(
                limit_side_len=960, limit_type="max"))
        pre_transforms.append(
            T.NormalizeImage(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                scale=1. / 255,
                order='hwc'))
        pre_transforms.append(image_common.ToCHWImage())
        return pre_transforms

    def _get_post_transforms_for_data(self, data):
        """ get postprocess transforms """
        post_transforms = [
            T.DBPostProcess(
                thresh=0.3,
                box_thresh=0.6,
                max_candidates=1000,
                unclip_ratio=1.5,
                use_dilation=False,
                score_mode='fast',
                box_type='quad'),
        ]
        if data.get('cli_flag', False):
            output_dir = data.get("output_dir", "./")
            post_transforms.append(T.SaveTextDetResults(output_dir))
        return post_transforms
