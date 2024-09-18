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
from ...modules.text_recognition.model_list import MODELS
from ..components import *
from ..results import TextRecResult
from ..utils.process_hook import batchable_method
from .base import BasePredictor


class TextRecPredictor(BasePredictor):

    entities = MODELS

    _FUNC_MAP = {}
    register = FuncRegister(_FUNC_MAP)

    def _build_components(self):
        ops = {}
        for cfg in self.config["PreProcess"]["transform_ops"]:
            tf_key = list(cfg.keys())[0]
            assert tf_key in self._FUNC_MAP
            func = self._FUNC_MAP.get(tf_key)
            args = cfg.get(tf_key, {})
            op = func(self, **args) if args else func(self)
            if op:
                ops[tf_key] = op

        predictor = ImagePredictor(
            model_dir=self.model_dir,
            model_prefix=self.MODEL_FILE_PREFIX,
            option=self.pp_option,
        )
        ops["predictor"] = predictor

        key, op = self.build_postprocess(**self.config["PostProcess"])
        ops[key] = op
        return ops

    @register("DecodeImage")
    def build_readimg(self, channel_first, img_mode):
        assert channel_first == False
        return ReadImage(format=img_mode, batch_size=self.kwargs.get("batch_size", 1))

    @register("RecResizeImg")
    def build_resize(self, image_shape):
        return OCRReisizeNormImg(rec_image_shape=image_shape)

    def build_postprocess(self, **kwargs):
        if kwargs.get("name") == "CTCLabelDecode":
            return "CTCLabelDecode", CTCLabelDecode(
                character_list=kwargs.get("character_dict"),
            )
        else:
            raise Exception()

    @register("MultiLabelEncode")
    def foo(self, *args, **kwargs):
        return None

    @register("KeepKeys")
    def foo(self, *args, **kwargs):
        return None

    @batchable_method
    def _pack_res(self, data):
        keys = ["img_path", "rec_text", "rec_score"]
        return {"result": TextRecResult({key: data[key] for key in keys})}
