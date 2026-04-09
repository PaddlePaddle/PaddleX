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

from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from ....utils.func_register import FuncRegister
from ...common.batch_sampler import ImageBatchSampler
from ...common.reader import ReadImage
from ..common import ToBatch, ToCHWImage
from ..predictors import RunnerPredictor, TransformersPredictor
from .processors import DBPostProcess, DetResizeForTest, NormalizeImage
from .result import TextDetResult

_TEXT_DET_MAX_LIMIT_MODELS = {
    "PP-OCRv5_server_det",
    "PP-OCRv5_mobile_det",
    "PP-OCRv4_server_det",
    "PP-OCRv4_mobile_det",
    "PP-OCRv3_server_det",
    "PP-OCRv3_mobile_det",
}
TEXT_DET_TRANSFORMERS_MODELS = ["PP-OCRv5_server_det", "PP-OCRv5_mobile_det"]


def _get_text_det_resize_cfg(config):
    for cfg in config.get("PreProcess", {}).get("transform_ops", []):
        resize_cfg = cfg.get("DetResizeForTest")
        if resize_cfg is not None:
            return resize_cfg
    return {}


def _get_text_det_resize_defaults(model_name: str, resize_cfg: dict) -> Tuple[int, str]:
    if model_name in _TEXT_DET_MAX_LIMIT_MODELS:
        return resize_cfg.get("resize_long", 960), resize_cfg.get("limit_type", "max")
    return resize_cfg.get("resize_long", 736), resize_cfg.get("limit_type", "min")


def _get_text_det_postprocess_defaults(config) -> Tuple[float, float, float]:
    postprocess_cfg = config.get("PostProcess", {})
    return (
        postprocess_cfg.get("thresh", 0.3),
        postprocess_cfg.get("box_thresh", 0.6),
        postprocess_cfg.get("unclip_ratio", 2.0),
    )


class TextDetRunnerPredictor(RunnerPredictor):

    _FUNC_MAP = {}
    register = FuncRegister(_FUNC_MAP)

    def __init__(
        self,
        limit_side_len: Union[int, None] = None,
        limit_type: Union[str, None] = None,
        thresh: Union[float, None] = None,
        box_thresh: Union[float, None] = None,
        unclip_ratio: Union[float, None] = None,
        input_shape=None,
        max_side_limit: int = 4000,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.limit_side_len = limit_side_len
        self.limit_type = limit_type
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.unclip_ratio = unclip_ratio
        self.input_shape = input_shape
        self.max_side_limit = max_side_limit

        self.pre_tfs, self.post_op = self._build()

    def _build_batch_sampler(self):
        return ImageBatchSampler()

    def _get_result_class(self):
        return TextDetResult

    def _build(self):
        pre_tfs = {"Read": ReadImage(format="RGB")}

        for cfg in self.config["PreProcess"]["transform_ops"]:
            tf_key = list(cfg.keys())[0]
            func = self._FUNC_MAP[tf_key]
            args = cfg.get(tf_key, {})
            name, op = func(self, **args) if args else func(self)
            if op:
                pre_tfs[name] = op
        pre_tfs["ToBatch"] = ToBatch()

        post_op = self.build_postprocess(**self.config["PostProcess"])
        return pre_tfs, post_op

    def process(
        self,
        batch_data: List[Union[str, np.ndarray]],
        limit_side_len: Union[int, None] = None,
        limit_type: Union[str, None] = None,
        thresh: Union[float, None] = None,
        box_thresh: Union[float, None] = None,
        unclip_ratio: Union[float, None] = None,
        max_side_limit: Union[int, None] = None,
    ):

        batch_raw_imgs = self.pre_tfs["Read"](imgs=batch_data.instances)
        batch_imgs, batch_shapes = self.pre_tfs["Resize"](
            imgs=batch_raw_imgs,
            limit_side_len=limit_side_len or self.limit_side_len,
            limit_type=limit_type or self.limit_type,
            max_side_limit=(
                max_side_limit if max_side_limit is not None else self.max_side_limit
            ),
        )
        batch_imgs = self.pre_tfs["Normalize"](imgs=batch_imgs)
        batch_imgs = self.pre_tfs["ToCHW"](imgs=batch_imgs)
        x = self.pre_tfs["ToBatch"](imgs=batch_imgs)

        batch_preds = self.runner(x=x)
        polys, scores = self.post_op(
            batch_preds,
            batch_shapes,
            thresh=thresh or self.thresh,
            box_thresh=box_thresh or self.box_thresh,
            unclip_ratio=unclip_ratio or self.unclip_ratio,
        )
        return {
            "input_path": batch_data.input_paths,
            "page_index": batch_data.page_indexes,
            "input_img": batch_raw_imgs,
            "dt_polys": polys,
            "dt_scores": scores,
        }

    @register("DecodeImage")
    def build_readimg(self, channel_first, img_mode):
        assert channel_first == False
        return "Read", ReadImage(format=img_mode)

    @register("DetResizeForTest")
    def build_resize(
        self,
        limit_side_len: Union[int, None] = None,
        limit_type: Union[str, None] = None,
        **kwargs,
    ):
        # TODO: align to PaddleOCR
        default_limit_side_len, default_limit_type = _get_text_det_resize_defaults(
            self.model_name, kwargs
        )
        limit_side_len = self.limit_side_len or default_limit_side_len
        limit_type = self.limit_type or default_limit_type

        return "Resize", DetResizeForTest(
            limit_side_len=limit_side_len,
            limit_type=limit_type,
            input_shape=self.input_shape,
            **kwargs,
        )

    @register("NormalizeImage")
    def build_normalize(
        self,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        scale=1 / 255,
        order="",
    ):
        return "Normalize", NormalizeImage(mean=mean, std=std, scale=scale, order=order)

    @register("ToCHWImage")
    def build_to_chw(self):
        return "ToCHW", ToCHWImage()

    def build_postprocess(self, **kwargs):
        if kwargs.get("name") == "DBPostProcess":
            default_thresh, default_box_thresh, default_unclip_ratio = (
                _get_text_det_postprocess_defaults({"PostProcess": kwargs})
            )
            return DBPostProcess(
                thresh=self.thresh or default_thresh,
                box_thresh=self.box_thresh or default_box_thresh,
                unclip_ratio=self.unclip_ratio or default_unclip_ratio,
                max_candidates=kwargs.get("max_candidates", 1000),
                use_dilation=kwargs.get("use_dilation", False),
                score_mode=kwargs.get("score_mode", "fast"),
                box_type=kwargs.get("box_type", "quad"),
            )

        else:
            raise Exception()

    @register("DetLabelEncode")
    def foo(self, *args, **kwargs):
        return None, None

    @register("KeepKeys")
    def foo(self, *args, **kwargs):
        return None, None


class TextDetTransformersPredictor(TransformersPredictor):

    def __init__(
        self,
        limit_side_len: Optional[int] = None,
        limit_type: Optional[str] = None,
        thresh: Optional[float] = None,
        box_thresh: Optional[float] = None,
        unclip_ratio: Optional[float] = None,
        max_side_limit: int = 4000,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.limit_side_len = limit_side_len
        self.limit_type = limit_type
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.unclip_ratio = unclip_ratio
        self.max_side_limit = max_side_limit

        self._load_default_settings()
        self.read_op = ReadImage(format="RGB")
        self.image_processor, self.infer = self._build()

    def _build_batch_sampler(self):
        return ImageBatchSampler()

    def _get_result_class(self):
        return TextDetResult

    def _build(self):
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        image_processor = self._load_pretrained_processor(AutoImageProcessor)
        model = self._load_pretrained_model(AutoModelForObjectDetection)
        return image_processor, model

    def _load_default_settings(self):
        resize_cfg = _get_text_det_resize_cfg(self.model_config)
        default_limit_side_len, default_limit_type = _get_text_det_resize_defaults(
            self.model_name, resize_cfg
        )
        default_thresh, default_box_thresh, default_unclip_ratio = (
            _get_text_det_postprocess_defaults(self.model_config)
        )

        if self.limit_side_len is None:
            self.limit_side_len = default_limit_side_len
        if self.limit_type is None:
            self.limit_type = default_limit_type
        if self.thresh is None:
            self.thresh = default_thresh
        if self.box_thresh is None:
            self.box_thresh = default_box_thresh
        if self.unclip_ratio is None:
            self.unclip_ratio = default_unclip_ratio
        if self.max_side_limit is None:
            self.max_side_limit = 4000

    def _normalize_dt_polys(self, boxes) -> np.ndarray:
        polys = boxes.detach().cpu().numpy().astype(np.int16, copy=False)
        if polys.size == 0:
            return np.empty((0, 4, 2), dtype=np.int16)
        return polys

    def _normalize_dt_scores(self, scores) -> np.ndarray:
        dt_scores = scores.detach().cpu().numpy().astype(np.float32, copy=False)
        if dt_scores.size == 0:
            return np.empty((0,), dtype=np.float32)
        return dt_scores

    def process(
        self,
        batch_data: List[Union[str, np.ndarray]],
        limit_side_len: Optional[int] = None,
        limit_type: Optional[str] = None,
        thresh: Optional[float] = None,
        box_thresh: Optional[float] = None,
        unclip_ratio: Optional[float] = None,
        max_side_limit: Optional[int] = None,
    ):
        batch_raw_imgs = self.read_op(imgs=batch_data.instances)
        images = [Image.fromarray(img) for img in batch_raw_imgs]

        model_inputs = self.preprocess_images(
            images=images,
            limit_side_len=(
                limit_side_len if limit_side_len is not None else self.limit_side_len
            ),
            limit_type=limit_type if limit_type is not None else self.limit_type,
            max_side_limit=(
                max_side_limit if max_side_limit is not None else self.max_side_limit
            ),
        )
        outputs = self.forward(model_inputs)
        polys, scores = self.postprocess(
            outputs,
            threshold=thresh if thresh is not None else self.thresh,
            target_sizes=model_inputs["target_sizes"],
            box_threshold=box_thresh if box_thresh is not None else self.box_thresh,
            unclip_ratio=(
                unclip_ratio if unclip_ratio is not None else self.unclip_ratio
            ),
        )

        return {
            "input_path": batch_data.input_paths,
            "page_index": batch_data.page_indexes,
            "input_img": batch_raw_imgs,
            "dt_polys": polys,
            "dt_scores": scores,
        }

    def postprocess(
        self, outputs, *, threshold, target_sizes, box_threshold, unclip_ratio, **kwargs
    ):
        predictions = self.image_processor.post_process_object_detection(
            outputs,
            threshold=threshold,
            target_sizes=target_sizes,
            box_threshold=box_threshold,
            unclip_ratio=unclip_ratio,
        )

        polys = [self._normalize_dt_polys(pred["boxes"]) for pred in predictions]
        scores = [self._normalize_dt_scores(pred["scores"]) for pred in predictions]

        return polys, scores
