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

from ...modules.formula_recognition.model_list import MODELS
from ..components import *
from ..results import TextRecResult
from .base import BasicPredictor


class LaTeXOCRPredictor(BasicPredictor):

    entities = MODELS

    def _build_components(self):
        self._add_component(
            [
                ReadImage(format="RGB"),
                LaTeXOCRReisizeNormImg(),
            ]
        )

        predictor = ImagePredictor(
            model_dir=self.model_dir,
            model_prefix=self.MODEL_FILE_PREFIX,
            option=self.pp_option,
        )
        self._add_component(predictor)

        op = self.build_postprocess(**self.config["PostProcess"])
        self._add_component(op)

    def build_postprocess(self, **kwargs):
        if kwargs.get("name") == "LaTeXOCRDecode":
            return LaTeXOCRDecode(
                character_list=kwargs.get("character_dict"),
            )
        else:
            raise Exception()

    def _pack_res(self, single):
        keys = ["input_path", "rec_text"]
        return TextRecResult({key: single[key] for key in keys})
