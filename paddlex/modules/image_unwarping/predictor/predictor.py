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
from pathlib import Path

from ...base import BasePredictor
from ...base.predictor.transforms import image_common
from .keys import WarpKeys as K
from . import transforms as T
from ..model_list import MODELS


class WarpPredictor(BasePredictor):
    """Clssification Predictor"""

    entities = MODELS


    @classmethod
    def get_input_keys(cls):
        """get input keys"""
        return [[K.IMAGE], [K.IM_PATH]]

    @classmethod
    def get_output_keys(cls):
        """get output keys"""
        return [K.DOCTR_IMG]

    def _run(self, batch_input):
        """run"""
        input_dict = {}
        input_dict[K.IMAGE] = np.stack(
            [data[K.IMAGE] for data in batch_input], axis=0
        ).astype(dtype=np.float32, copy=False)
        input_ = [input_dict[K.IMAGE]]
        outputs = self._predictor.predict(input_)
        Warp_outs = outputs[0]
        # In-place update
        pred = batch_input
        for dict_, Warp_out in zip(pred, Warp_outs):
            dict_[K.DOCTR_IMG] = Warp_out
        return pred

    def _get_pre_transforms_from_config(self):
        """get preprocess transforms"""
        pre_transforms = [
            image_common.ReadImage(format='RGB'),
            image_common.Normalize(scale=1./255, mean=0.0, std=1.0),
            image_common.ToCHWImage()
        ]
        
        return pre_transforms

    def _get_post_transforms_from_config(self):
        """get postprocess transforms"""
        post_transforms = [
            T.DocTrPostProcess(scale=255.),
            T.SaveDocTrResults(self.output)
        ] # yapf: disable
        return post_transforms
