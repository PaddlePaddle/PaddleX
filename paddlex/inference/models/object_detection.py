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
from ...modules.object_detection.model_list import MODELS
from ..components import *
from ..results import DetResult
from ..utils.process_hook import batchable_method
from .base import CVPredictor


class DetPredictor(CVPredictor):

    entities = MODELS

    _FUNC_MAP = {}
    register = FuncRegister(_FUNC_MAP)

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
        if "DETR" in self.model_name or "RCNN" in self.model_name:
            predictor.set_inputs(
                {
                    "img": "img",
                    "scale_factors": "scale_factors",
                    "img_size": "img_size",
                }
            )

        self._add_component(
            [
                ("Predictor", predictor),
                DetPostProcess(
                    threshold=self.config["draw_threshold"],
                    labels=self.config["label_list"],
                ),
            ]
        )

    @register("Resize")
    def build_resize(self, target_size, keep_ratio=False, interp=2):
        assert target_size
        if isinstance(interp, int):
            interp = {
                0: "NEAREST",
                1: "LINEAR",
                2: "CUBIC",
                3: "AREA",
                4: "LANCZOS4",
            }[interp]
        op = Resize(target_size=target_size, keep_ratio=keep_ratio, interp=interp)
        return op

    @register("NormalizeImage")
    def build_normalize(
        self,
        norm_type=None,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        is_scale=None,
    ):
        if is_scale:
            scale = 1.0 / 255.0
        else:
            scale = 1
        if not norm_type or norm_type == "none":
            norm_type = "mean_std"
        if norm_type != "mean_std":
            mean = 0
            std = 1
        return Normalize(mean=mean, std=std)

    @register("Permute")
    def build_to_chw(self):
        return ToCHWImage()

    @register("Pad")
    def build_pad(self, fill_value=None, size=None):
        if fill_value is None:
            fill_value = [127.5, 127.5, 127.5]
        if size is None:
            size = [3, 640, 640]
        return DetPad(size=size, fill_value=fill_value)

    @register("PadStride")
    def build_pad_stride(self, stride=32):
        return PadStride(stride=stride)

    def _pack_res(self, single):
        keys = ["img_path", "boxes"]
        return DetResult({key: single[key] for key in keys})
