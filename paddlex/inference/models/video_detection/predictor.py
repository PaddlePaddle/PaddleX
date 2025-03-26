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

from ....modules.video_detection.model_list import MODELS
from ....utils.func_register import FuncRegister
from ...common.batch_sampler import VideoBatchSampler
from ...common.reader import ReadVideo
from ..base import BasePredictor
from .processors import DetVideoPostProcess, Image2Array, NormalizeVideo, ResizeVideo
from .result import DetVideoResult


class VideoDetPredictor(BasePredictor):

    entities = MODELS

    _FUNC_MAP = {}
    register = FuncRegister(_FUNC_MAP)

    def __init__(
        self,
        nms_thresh: Union[float, None] = None,
        score_thresh: Union[float, None] = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh
        self.pre_tfs, self.infer, self.post_op = self._build()

    def _build_batch_sampler(self):
        return VideoBatchSampler()

    def _get_result_class(self):
        return DetVideoResult

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

        infer = self.create_static_infer()
        post_op = {}
        for cfg in self.config["PostProcess"]["transform_ops"]:
            tf_key = list(cfg.keys())[0]
            assert tf_key in self._FUNC_MAP
            func = self._FUNC_MAP[tf_key]
            args = cfg.get(tf_key, {})
            if tf_key == "DetVideoPostProcess":
                args["label_list"] = self.config["label_list"]
            name, op = func(self, **args) if args else func(self)
            if op:
                post_op[name] = op

        return pre_tfs, infer, post_op

    def process(
        self,
        batch_data,
        nms_thresh: Union[float, None] = None,
        score_thresh: Union[float, None] = None,
    ):
        batch_raw_videos = self.pre_tfs["ReadVideo"](videos=batch_data)
        batch_videos = self.pre_tfs["ResizeVideo"](videos=batch_raw_videos)
        batch_videos = self.pre_tfs["Image2Array"](videos=batch_videos)
        x = self.pre_tfs["NormalizeVideo"](videos=batch_videos)
        num_seg = len(x[0])
        pred_seg = []
        for i in range(num_seg):
            batch_preds = self.infer(x=[x[0][i]])
            pred_seg.append(batch_preds)
        batch_bboxes = self.post_op["DetVideoPostProcess"](
            preds=[pred_seg],
            nms_thresh=nms_thresh or self.nms_thresh,
            score_thresh=score_thresh or self.score_thresh,
        )
        return {
            "input_path": batch_data,
            "result": batch_bboxes,
        }

    @register("ReadVideo")
    def build_readvideo(self, num_seg=8):
        return "ReadVideo", ReadVideo(backend="opencv", num_seg=num_seg)

    @register("ResizeVideo")
    def build_resize(self, target_size=224):
        return "ResizeVideo", ResizeVideo(
            target_size=target_size,
        )

    @register("Image2Array")
    def build_image2array(self, data_format="tchw"):
        return "Image2Array", Image2Array(data_format="tchw")

    @register("NormalizeVideo")
    def build_normalize(
        self,
        scale=255.0,
    ):
        return "NormalizeVideo", NormalizeVideo(scale=scale)

    @register("DetVideoPostProcess")
    def build_postprocess(self, nms_thresh, score_thresh, label_list=[]):
        if not self.nms_thresh:
            self.nms_thresh = nms_thresh
        if not self.score_thresh:
            self.score_thresh = score_thresh
        return "DetVideoPostProcess", DetVideoPostProcess(label_list=label_list)
