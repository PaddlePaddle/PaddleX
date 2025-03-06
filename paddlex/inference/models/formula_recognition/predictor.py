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

from ....utils import logging
from ....utils.func_register import FuncRegister
from ....modules.formula_recognition.model_list import MODELS
from ...common.batch_sampler import ImageBatchSampler
from ...common.reader import ReadImage
from ..common import (
    StaticInfer,
)
from ..base import BasicPredictor
from .processors import (
    MinMaxResize,
    LatexTestTransform,
    LatexImageFormat,
    LaTeXOCRDecode,
    NormalizeImage,
    ToBatch,
    UniMERNetImgDecode,
    UniMERNetDecode,
    UniMERNetTestTransform,
    UniMERNetImageFormat,
)

from .result import FormulaRecResult


class FormulaRecPredictor(BasicPredictor):

    entities = MODELS

    _FUNC_MAP = {}
    register = FuncRegister(_FUNC_MAP)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_tfs, self.infer, self.post_op = self._build()

    def _build_batch_sampler(self):
        return ImageBatchSampler()

    def _get_result_class(self):
        return FormulaRecResult

    def _build(self):
        pre_tfs = {"Read": ReadImage(format="RGB")}
        for cfg in self.config["PreProcess"]["transform_ops"]:
            tf_key = list(cfg.keys())[0]
            assert tf_key in self._FUNC_MAP
            func = self._FUNC_MAP[tf_key]
            args = cfg.get(tf_key, {})
            name, op = func(self, **args) if args else func(self)
            if op:
                pre_tfs[name] = op
        pre_tfs["ToBatch"] = ToBatch()

        infer = StaticInfer(
            model_dir=self.model_dir,
            model_prefix=self.MODEL_FILE_PREFIX,
            option=self.pp_option,
        )

        post_op = self.build_postprocess(**self.config["PostProcess"])
        return pre_tfs, infer, post_op

    def process(self, batch_data):
        batch_raw_imgs = self.pre_tfs["Read"](imgs=batch_data.instances)
        if self.model_name in ("LaTeX_OCR_rec"):
            batch_imgs = self.pre_tfs["MinMaxResize"](imgs=batch_raw_imgs)
            batch_imgs = self.pre_tfs["LatexTestTransform"](imgs=batch_imgs)
            batch_imgs = self.pre_tfs["NormalizeImage"](imgs=batch_imgs)
            batch_imgs = self.pre_tfs["LatexImageFormat"](imgs=batch_imgs)
        elif self.model_name in ("UniMERNet"):
            batch_imgs = self.pre_tfs["UniMERNetImgDecode"](imgs=batch_raw_imgs)
            batch_imgs = self.pre_tfs["UniMERNetTestTransform"](imgs=batch_imgs)
            batch_imgs = self.pre_tfs["UniMERNetImageFormat"](imgs=batch_imgs)
        elif self.model_name in ("PP-FormulaNet-S", "PP-FormulaNet-L"):
            batch_imgs = self.pre_tfs["UniMERNetImgDecode"](imgs=batch_raw_imgs)
            batch_imgs = self.pre_tfs["UniMERNetTestTransform"](imgs=batch_imgs)
            batch_imgs = self.pre_tfs["LatexImageFormat"](imgs=batch_imgs)

        x = self.pre_tfs["ToBatch"](imgs=batch_imgs)
        batch_preds = self.infer(x=x)
        batch_preds = [p.reshape([-1]) for p in batch_preds[0]]
        rec_formula = self.post_op(batch_preds)
        return {
            "input_path": batch_data.input_paths,
            "page_index": batch_data.page_indexes,
            "input_img": batch_raw_imgs,
            "rec_formula": rec_formula,
        }

    @register("DecodeImage")
    def build_readimg(self, channel_first, img_mode):
        assert channel_first == False
        return "Read", ReadImage(format=img_mode)

    @register("MinMaxResize")
    def build_min_max_resize(self, min_dimensions, max_dimensions):
        return "MinMaxResize", MinMaxResize(
            min_dimensions=min_dimensions, max_dimensions=max_dimensions
        )

    @register("LatexTestTransform")
    def build_latex_test_transform(
        self,
    ):
        return "LatexTestTransform", LatexTestTransform()

    @register("NormalizeImage")
    def build_normalize(self, mean, std, order="chw"):
        return "NormalizeImage", NormalizeImage(mean=mean, std=std, order=order)

    @register("LatexImageFormat")
    def build_latexocr_imageformat(self):
        return "LatexImageFormat", LatexImageFormat()

    @register("UniMERNetImgDecode")
    def build_unimernet_decode(self, input_size):
        return "UniMERNetImgDecode", UniMERNetImgDecode(input_size)

    def build_postprocess(self, **kwargs):
        if kwargs.get("name") == "LaTeXOCRDecode":
            return LaTeXOCRDecode(
                character_list=kwargs.get("character_dict"),
            )
        elif kwargs.get("name") == "UniMERNetDecode":
            return UniMERNetDecode(
                character_list=kwargs.get("character_dict"),
            )
        else:
            raise Exception()

    @register("UniMERNetTestTransform")
    def build_unimernet_imageformat(self):
        return "UniMERNetTestTransform", UniMERNetTestTransform()

    @register("UniMERNetImageFormat")
    def build_unimernet_imageformat(self):
        return "UniMERNetImageFormat", UniMERNetImageFormat()

    @register("UniMERNetLabelEncode")
    def foo(self, *args, **kwargs):
        return None, None

    @register("KeepKeys")
    def foo(self, *args, **kwargs):
        return None, None
