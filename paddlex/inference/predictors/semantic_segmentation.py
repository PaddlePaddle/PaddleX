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
from ...modules.semantic_segmentation.model_list import MODELS
from ..components import *
from ..results import SegResult
from ..utils.process_hook import batchable_method
from .base import BasicPredictor


class SegPredictor(BasicPredictor):

    entities = MODELS

    _FUNC_MAP = {}
    register = FuncRegister(_FUNC_MAP)

    def _check_args(self, kwargs):
        assert set(kwargs.keys()).issubset(set(["batch_size"]))
        return kwargs

    def _build_components(self):
        ops = {}
        ops["ReadImage"] = ReadImage(
            batch_size=self.kwargs.get("batch_size", 1), format="RGB"
        )
        ops["ToCHWImage"] = ToCHWImage()
        for cfg in self.config["Deploy"]["transforms"]:
            tf_key = cfg["type"]
            func = self._FUNC_MAP.get(tf_key)
            cfg.pop("type")
            args = cfg
            op = func(self, **args) if args else func(self)
            ops[tf_key] = op

        predictor = ImagePredictor(
            model_dir=self.model_dir,
            model_prefix=self.MODEL_FILE_PREFIX,
            option=self.pp_option,
        )
        ops["predictor"] = predictor
        return ops

    @register("Resize")
    def build_resize(
        self, target_size, keep_ratio=False, size_divisor=None, interp="LINEAR"
    ):
        assert target_size
        op = Resize(
            target_size=target_size,
            keep_ratio=keep_ratio,
            size_divisor=size_divisor,
            interp=interp,
        )
        return op

    @register("ResizeByLong")
    def build_resizebylong(self, long_size):
        assert long_size
        return ResizeByLong(
            target_long_edge=long_size, size_divisor=size_divisor, interp=interp
        )

    @register("ResizeByShort")
    def build_resizebylong(self, short_size):
        assert short_size
        return ResizeByLong(
            target_long_edge=short_size, size_divisor=size_divisor, interp=interp
        )

    @register("Normalize")
    def build_normalize(
        self,
        mean=0.5,
        std=0.5,
    ):
        return Normalize(mean=mean, std=std)

    def _pack_res(self, single):
        keys = ["img_path", "pred"]
        return SegResult({key: single[key] for key in keys})
