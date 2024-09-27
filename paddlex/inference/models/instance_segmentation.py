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

from .object_detection import DetPredictor
from ...utils.func_register import FuncRegister
from ...modules.instance_segmentation.model_list import MODELS
from ..components import *
from ..results import InstanceSegResult


class InstanceSegPredictor(DetPredictor):

    entities = MODELS

    def _build_components(self):
        self._add_component(ReadImage(format="RGB"))
        for cfg in self.config["Preprocess"]:
            tf_key = cfg["type"]
            func = self._FUNC_MAP[tf_key]
            cfg.pop("type")
            args = cfg
            op = func(self, **args) if args else func(self)
            self._add_component(op)

        predictor = ImageDetPredictor(
            model_dir=self.model_dir,
            model_prefix=self.MODEL_FILE_PREFIX,
            option=self.pp_option,
        )
        if "RT-DETR" in self.model_name:
            predictor.set_inputs(
                {"img": "img", "scale_factors": "scale_factors", "img_size": "img_size"}
            )
        self._add_component(
            [
                ("Predictor", predictor),
                InstanceSegPostProcess(
                    threshold=self.config["draw_threshold"],
                    labels=self.config["label_list"],
                ),
            ]
        )

    def _pack_res(self, single):
        keys = ["img_path", "boxes", "masks"]
        return InstanceSegResult({key: single[key] for key in keys})
