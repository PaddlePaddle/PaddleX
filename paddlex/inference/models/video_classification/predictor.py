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

from typing import Union

from ....modules.video_classification.model_list import MODELS
from ....utils.func_register import FuncRegister
from ...common.batch_sampler import VideoBatchSampler
from ...common.reader import ReadVideo
from ..base import BasePredictor
from .processors import (
    CenterCrop,
    Image2Array,
    NormalizeVideo,
    Scale,
    ToBatch,
    VideoClasTopk,
)
from .result import TopkVideoResult


class VideoClasPredictor(BasePredictor):

    entities = MODELS

    _FUNC_MAP = {}
    register = FuncRegister(_FUNC_MAP)

    def __init__(self, topk: Union[int, None] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.topk = topk
        self.pre_tfs, self.infer, self.post_op = self._build()

    def _build_batch_sampler(self):
        return VideoBatchSampler()

    def _get_result_class(self):
        return TopkVideoResult

    def _build(self):
        pre_tfs = {}
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

        post_op = {}
        for key in self.config["PostProcess"]:
            func = self._FUNC_MAP.get(key)
            args = self.config["PostProcess"].get(key, {})
            name, op = func(self, **args) if args else func(self)
            post_op[name] = op

        return pre_tfs, infer, post_op

    def process(self, batch_data, topk: Union[int, None] = None):
        batch_raw_videos = self.pre_tfs["ReadVideo"](videos=batch_data)
        batch_videos = self.pre_tfs["Scale"](videos=batch_raw_videos)
        batch_videos = self.pre_tfs["CenterCrop"](videos=batch_videos)
        batch_videos = self.pre_tfs["Image2Array"](videos=batch_videos)
        batch_videos = self.pre_tfs["NormalizeVideo"](videos=batch_videos)
        x = self.pre_tfs["ToBatch"](videos=batch_videos)
        batch_preds = self.infer(x=x)

        batch_class_ids, batch_scores, batch_label_names = self.post_op["Topk"](
            batch_preds, topk=topk or self.topk
        )
        return {
            "input_path": batch_data,
            "class_ids": batch_class_ids,
            "scores": batch_scores,
            "label_names": batch_label_names,
        }

    @register("ReadVideo")
    def build_readvideo(
        self,
        num_seg=8,
        target_size=224,
        seg_len=1,
        sample_type=None,
    ):
        return "ReadVideo", ReadVideo(
            backend="decord",
            num_seg=num_seg,
            seg_len=seg_len,
            sample_type=sample_type,
        )

    @register("Scale")
    def build_scale(self, short_size=224):
        return "Scale", Scale(
            short_size=short_size,
            fixed_ratio=True,
            keep_ratio=None,
            do_round=False,
        )

    @register("CenterCrop")
    def build_center_crop(self, target_size=224):
        return "CenterCrop", CenterCrop(target_size=target_size)

    @register("Image2Array")
    def build_image2array(self, data_format="tchw"):
        return "Image2Array", Image2Array(transpose=True, data_format="tchw")

    @register("NormalizeVideo")
    def build_normalize(
        self,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ):
        return "NormalizeVideo", NormalizeVideo(mean=mean, std=std)

    @register("Topk")
    def build_topk(self, topk, label_list=None):
        if not self.topk:
            self.topk = int(topk)
        return "Topk", VideoClasTopk(class_ids=label_list)

    @register("KeepKeys")
    def foo(self, *args, **kwargs):
        return None, None
