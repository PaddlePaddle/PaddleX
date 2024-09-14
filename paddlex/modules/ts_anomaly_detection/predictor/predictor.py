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
from ...base.predictor.transforms import ts_common
from ...base import BasePredictor
from .keys import TSFCKeys as K
from . import transforms as T
from .utils import InnerConfig
from ..model_list import MODELS


class TSADPredictor(BasePredictor):
    """SegPredictor"""

    entities = MODELS

    def __init__(
        self,
        model_name,
        model_dir,
        kernel_option,
        output,
        pre_transforms=None,
        post_transforms=None,
    ):
        super().__init__(
            model_name=model_name,
            model_dir=model_dir,
            kernel_option=kernel_option,
            output=output,
            pre_transforms=pre_transforms,
            post_transforms=post_transforms,
        )

    def load_other_src(self):
        """load the inner config file"""
        infer_cfg_file_path = os.path.join(self.model_dir, "inference.yml")
        if not os.path.exists(infer_cfg_file_path):
            raise FileNotFoundError(f"Cannot find config file: {infer_cfg_file_path}")
        return InnerConfig(infer_cfg_file_path, self.model_dir)

    @classmethod
    def get_input_keys(cls):
        """get input keys"""
        return [[K.TS], [K.TS_PATH]]

    @classmethod
    def get_output_keys(cls):
        """get output keys"""
        return [K.PRED]

    def _run(self, batch_input):
        """run"""
        n = len(batch_input[0][K.TS])
        input_ = [
            np.stack([lst[i] for lst in [data[K.TS] for data in batch_input]], axis=0)
            for i in range(n)
        ]

        outputs = self._predictor.predict(input_)
        batch_output = outputs[0]
        # In-place update
        for dict_, output in zip(batch_input, batch_output):
            dict_[K.PRED] = output
        return batch_input

    def _get_pre_transforms_from_config(self):
        """_get_pre_transforms_from_config"""
        logging.info(
            f"Transformation operators for data preprocessing will be inferred from config file."
        )
        pre_transforms = self.other_src.pre_transforms
        pre_transforms.insert(0, ts_common.ReadTS())

        return pre_transforms

    def _get_post_transforms_from_config(self):
        """_get_post_transforms_from_config"""
        post_transforms = self.other_src.post_transforms
        post_transforms.append(T.SaveTSResults(self.output))
        return post_transforms
