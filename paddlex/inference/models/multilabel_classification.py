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

from ...utils.func_register import FuncRegister
from ...modules.multilabel_classification.model_list import MODELS
from ..components import *
from ..results import MLClassResult
from ..utils.process_hook import batchable_method
from .image_classification import ClasPredictor


class MLClasPredictor(ClasPredictor):

    entities = [*MODELS]

    def _pack_res(self, single):
        keys = ["input_path", "class_ids", "scores"]
        if "label_names" in single:
            keys.append("label_names")
        return MLClassResult({key: single[key] for key in keys})
