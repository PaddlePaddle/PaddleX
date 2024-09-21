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
from .keys import ClsKeys as K
from .utils import InnerConfig
from ....utils import logging
from . import transforms as T
from .predictor import ClsPredictor
from ..model_list import ML_MODELS


class MLClsPredictor(ClsPredictor, BasePredictor):
    """ MLClssification Predictor """
    entities = ML_MODELS

    def _get_post_transforms_from_config(self):
        """ get postprocess transforms """
        post_transforms = self.other_src.post_transforms
        post_transforms.extend([
            T.PrintResult(), T.SaveMLClsResults(self.output,
                                                self.other_src.labels)
        ])
        return post_transforms
