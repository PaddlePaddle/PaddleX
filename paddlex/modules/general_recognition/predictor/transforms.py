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

from .keys import ShiTuRecKeys as K
from ...base import BaseTransform
from ....utils import logging

__all__ = [
    "NormalizeFeatures",
    "PrintShiTuRecResult"
]


class NormalizeFeatures(BaseTransform):
    """Normalize Features Transform"""

    def apply(self, data):
        """apply"""
        x = data[K.SHITU_REC_PRED]
        feas_norm = np.sqrt(np.sum(np.square(x), axis=0, keepdims=True))
        x = np.divide(x, feas_norm)
        data[K.SHITU_REC_RESULT] = x
        return data

    @classmethod
    def get_input_keys(cls):
        """get input keys"""
        return [K.IM_PATH, K.SHITU_REC_PRED]

    @classmethod
    def get_output_keys(cls):
        """get output keys"""
        return [K.SHITU_REC_RESULT]


class PrintShiTuRecResult(BaseTransform):
    """Print Result Transform"""

    def apply(self, data):
        """apply"""
        logging.info("The prediction result is:")
        logging.info(data[K.SHITU_REC_RESULT])
        return data

    @classmethod
    def get_input_keys(cls):
        """get input keys"""
        return [K.SHITU_REC_RESULT]

    @classmethod
    def get_output_keys(cls):
        """get output keys"""
        return []