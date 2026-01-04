# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from ....modules.formula_recognition.model_list import MODELS
from ....utils import logging
from ....utils.func_register import FuncRegister
from ...common.batch_sampler import ImageBatchSampler
from ...common.reader import ReadImage
from ..base import BasePredictor
from .processors import (
    LatexImageFormat,
    LaTeXOCRDecode,
    LatexTestTransform,
    MinMaxResize,
    NormalizeImage,
    ToBatch,
    UniMERNetDecode,
    UniMERNetImageFormat,
    UniMERNetImgDecode,
    UniMERNetTestTransform,
)
from .result import FormulaRecResult


class FormulaRecPredictor(BasePredictor):
    """FormulaRecPredictor that inherits from BasePredictor."""

    entities = MODELS

    _FUNC_MAP = {}
    register = FuncRegister(_FUNC_MAP)

    def __init__(self, *args, **kwargs):
        """Initializes FormulaRecPredictor.
        Args:
            *args: Arbitrary positional arguments passed to the superclass.
            **kwargs: Arbitrary keyword arguments passed to the superclass.
        """
        super().__init__(*args, **kwargs)

        self.model_names_only_supports_batchsize_of_one = {
            "LaTeX_OCR_rec",
        }
        if self.model_name in self.model_names_only_supports_batchsize_of_one:
            logging.warning(
                f"Formula Recognition Models: \"{', '.join(list(self.model_names_only_supports_batchsize_of_one))}\" only supports prediction with a batch_size of one, "
                "if you set the predictor with a batch_size larger than one, no error will occur, however, it will actually inference with a batch_size of one, "
                f"which will lead to a slower inference speed. You are now using {self.config['Global']['model_name']}."
            )

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

        infer = self.create_static_infer()

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
        elif self.model_name in (
            "PP-FormulaNet-S",
            "PP-FormulaNet-L",
            "PP-FormulaNet_plus-S",
            "PP-FormulaNet_plus-M",
            "PP-FormulaNet_plus-L",
        ):
            batch_imgs = self.pre_tfs["UniMERNetImgDecode"](imgs=batch_raw_imgs)
            batch_imgs = self.pre_tfs["UniMERNetTestTransform"](imgs=batch_imgs)
            batch_imgs = self.pre_tfs["LatexImageFormat"](imgs=batch_imgs)

        if self.model_name in self.model_names_only_supports_batchsize_of_one:
            batch_preds = []
            max_length = 0
            for batch_img in batch_imgs:
                batch_pred_ = self.infer([batch_img])[0].reshape([-1])
                max_length = max(max_length, batch_pred_.shape[0])
                batch_preds.append(batch_pred_)
            for i in range(len(batch_preds)):
                batch_preds[i] = np.pad(
                    batch_preds[i],
                    (0, max_length - batch_preds[i].shape[0]),
                    mode="constant",
                    constant_values=0,
                )
        else:
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
    def build_readimg(self, channel_first, img_mode="RGB"):
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
