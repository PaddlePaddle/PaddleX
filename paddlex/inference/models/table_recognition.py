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
from ...modules.table_recognition.model_list import MODELS
from ..components import *
from ..results import TableRecResult
from ..utils.process_hook import batchable_method
from .base import CVPredictor


class TablePredictor(CVPredictor):
    """table recognition predictor"""

    entities = MODELS

    _FUNC_MAP = {}
    register = FuncRegister(_FUNC_MAP)

    def _build_components(self):
        for cfg in self.config["PreProcess"]["transform_ops"]:
            tf_key = list(cfg.keys())[0]
            func = self._FUNC_MAP.get(tf_key)
            args = cfg.get(tf_key, {})
            op = func(self, **args) if args else func(self)
            if op:
                self._add_component(op)

        predictor = ImagePredictor(
            model_dir=self.model_dir,
            model_prefix=self.MODEL_FILE_PREFIX,
            option=self.pp_option,
        )
        self._add_component(("Predictor", predictor))

        op = self.build_postprocess(**self.config["PostProcess"])
        self._add_component(op)

    def build_postprocess(self, **kwargs):
        if kwargs.get("name") == "TableLabelDecode":
            return TableLabelDecode(
                merge_no_span_structure=kwargs.get("merge_no_span_structure"),
                dict_character=kwargs.get("character_dict"),
            )
        else:
            raise Exception()

    @register("DecodeImage")
    def build_readimg(self, *args, **kwargs):
        return ReadImage(*args, **kwargs)

    @register("TableLabelEncode")
    def foo(self, *args, **kwargs):
        return None

    @register("TableBoxEncode")
    def foo(self, *args, **kwargs):
        return None

    @register("ResizeTableImage")
    def build_resize_table(self, max_len=488):
        return ResizeByLong(target_long_edge=max_len)

    @register("NormalizeImage")
    def build_normalize(
        self,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        scale=1 / 255,
        order="hwc",
    ):
        return Normalize(mean=mean, std=std)

    @register("PaddingTableImage")
    def build_padding(self, size=[488, 448], pad_value=0):
        return Pad(target_size=size[0], val=pad_value)

    @register("ToCHWImage")
    def build_to_chw(self):
        return ToCHWImage()

    @register("KeepKeys")
    def foo(self, *args, **kwargs):
        return None

    def _pack_res(self, single):
        keys = ["img_path", "bbox", "structure"]
        return TableRecResult({key: single[key] for key in keys})
