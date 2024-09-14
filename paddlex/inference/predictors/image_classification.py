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
from ...modules.image_classification.model_list import MODELS
from ..components import *
from ..results import TopkResult
from ..utils.process_hook import batchable_method
from .base import BasePredictor


class ClasPredictor(BasePredictor):

    entities = MODELS

    INPUT_KEYS = "x"
    OUTPUT_KEYS = "topk_res"
    DEAULT_INPUTS = {"x": "x"}
    DEAULT_OUTPUTS = {"topk_res": "topk_res"}

    _FUNC_MAP = {}
    register = FuncRegister(_FUNC_MAP)

    def _build_components(self):
        ops = {}
        ops["ReadImage"] = ReadImage(batch_size=self.kwargs.get("batch_size", 1))
        for cfg in self.config["PreProcess"]["transform_ops"]:
            tf_key = list(cfg.keys())[0]
            func = self._FUNC_MAP.get(tf_key)
            args = cfg.get(tf_key, {})
            op = func(self, **args) if args else func(self)
            ops[tf_key] = op

        kernel_option = PaddlePredictorOption()
        kernel_option.set_device(self.device)
        predictor = ImagePredictor(
            model_dir=self.model_dir,
            model_prefix=self.MODEL_FILE_PREFIX,
            option=kernel_option,
        )
        ops["predictor"] = predictor

        post_processes = self.config["PostProcess"]
        for key in post_processes:
            func = self._FUNC_MAP.get(key)
            args = post_processes.get(key, {})
            op = func(self, **args) if args else func(self)
            ops[key] = op
        return ops

    @register("ResizeImage")
    def build_resize(self, resize_short=None, size=None):
        assert resize_short or size
        if resize_short:
            op = ResizeByShort(
                target_short_edge=resize_short, size_divisor=None, interp="LINEAR"
            )
        else:
            op = Resize(target_size=size)
        return op

    @register("CropImage")
    def build_crop(self, size=224):
        return Crop(crop_size=size)

    @register("NormalizeImage")
    def build_normalize(
        self,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        scale=1 / 255,
        order="",
        channel_num=3,
    ):
        assert channel_num == 3
        assert order == ""
        return Normalize(mean=mean, std=std)

    @register("ToCHWImage")
    def build_to_chw(self):
        return ToCHWImage()

    @register("Topk")
    def build_topk(self, topk, label_list=None):
        return Topk(topk=int(topk), class_ids=label_list)

    @batchable_method
    def _pack_res(self, data):
        keys = ["img_path", "class_ids", "scores"]
        if "label_names" in data:
            keys.append("label_names")
        return {"topk_res": TopkResult({key: data[key] for key in keys})}
