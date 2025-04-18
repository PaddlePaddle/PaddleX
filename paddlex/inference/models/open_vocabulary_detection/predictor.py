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

import inspect
from typing import Any, Callable, Dict, List, Optional, Union

from ....modules.open_vocabulary_detection.model_list import MODELS
from ....utils.func_register import FuncRegister
from ...common.batch_sampler import ImageBatchSampler
from ...common.reader import ReadImage
from ..base import BasePredictor
from ..object_detection.result import DetResult
from .processors import (
    GroundingDINOPostProcessor,
    GroundingDINOProcessor,
    YOLOWorldPostProcessor,
    YOLOWorldProcessor,
)


class OVDetPredictor(BasePredictor):

    entities = MODELS

    _FUNC_MAP = {}
    register = FuncRegister(_FUNC_MAP)

    def __init__(
        self, *args, thresholds: Optional[Union[Dict, float]] = None, **kwargs
    ):
        """Initializes DetPredictor.
        Args:
            *args: Arbitrary positional arguments passed to the superclass.
            thresholds (Optional[Union[Dict, float]], optional): The thresholds for filtering out low-confidence predictions, using a dict to record multiple thresholds
                Defaults to None.
            **kwargs: Arbitrary keyword arguments passed to the superclass.
        """
        super().__init__(*args, **kwargs)
        if isinstance(thresholds, float):
            thresholds = {"threshold": thresholds}
        self.thresholds = thresholds
        self.pre_ops, self.infer, self.post_op = self._build()

    def _build_batch_sampler(self):
        return ImageBatchSampler()

    def _get_result_class(self):
        return DetResult

    def _build(self):
        # build model preprocess ops
        pre_ops = [ReadImage(format="RGB")]
        for cfg in self.config["Preprocess"]:
            tf_key = cfg["type"]
            func = self._FUNC_MAP[tf_key]
            cfg.pop("type")
            args = cfg
            op = func(self, **args) if args else func(self)
            if op:
                pre_ops.append(op)

        # build infer
        infer = self.create_static_infer()

        # build postprocess op
        post_op = self.build_postprocess(pre_ops=pre_ops)

        return pre_ops, infer, post_op

    def process(
        self, batch_data: List[Any], prompt: str, thresholds: Optional[dict] = None
    ):
        """
        Process a batch of data through the preprocessing, inference, and postprocessing.

        Args:
            batch_data (List[str]): A batch of input data (e.g., image file paths).
            prompt (str): Text prompt for open vocabulary detection.
            thresholds (Optional[dict]): thresholds used for postprocess.

        Returns:
            dict: A dictionary containing the input path, raw image, class IDs, scores, and label names
                for every instance of the batch. Keys include 'input_path', 'input_img', 'class_ids', 'scores', and 'label_names'.
        """
        image_paths = batch_data.input_paths
        src_images = self.pre_ops[0](batch_data.instances)
        datas = src_images
        # preprocess for image only
        for pre_op in self.pre_ops[1:-1]:
            datas = pre_op(datas)

        # use Model-specific preprocessor to format batch inputs
        batch_inputs = self.pre_ops[-1](datas, prompt)

        # do infer
        batch_preds = self.infer(batch_inputs)

        # postprocess
        current_thresholds = self._parse_current_thresholds(
            self.post_op, self.thresholds, thresholds
        )
        boxes = self.post_op(
            *batch_preds, prompt=prompt, src_images=src_images, **current_thresholds
        )

        return {
            "input_path": image_paths,
            "input_img": [img[..., ::-1] for img in src_images],
            "boxes": boxes,
        }

    def _parse_current_thresholds(self, func, init_thresholds, process_thresholds):
        assert isinstance(func, Callable)
        thr2val = {}
        for name, param in inspect.signature(func).parameters.items():
            if "threshold" in name:
                thr2val[name] = None
        if init_thresholds is not None:
            thr2val.update(init_thresholds)
        if process_thresholds is not None:
            thr2val.update(process_thresholds)
        return thr2val

    def build_postprocess(self, **kwargs):
        if "GroundingDINO" in self.model_name:
            pre_ops = kwargs.get("pre_ops")
            return GroundingDINOPostProcessor(
                tokenizer=pre_ops[-1].tokenizer,
                box_threshold=self.config["box_threshold"],
                text_threshold=self.config["text_threshold"],
            )
        elif "YOLO-World" in self.model_name:
            return YOLOWorldPostProcessor(
                threshold=self.config["threshold"],
            )
        else:
            raise NotImplementedError

    @register("GroundingDINOProcessor")
    def build_grounding_dino_preprocessor(
        self, text_max_words=256, target_size=(800, 1333)
    ):
        return GroundingDINOProcessor(
            model_dir=self.model_dir,
            text_max_words=text_max_words,
            target_size=target_size,
        )

    @register("YOLOWorldProcessor")
    def build_yoloworld_preprocessor(
        self,
        image_target_size=(640, 640),
        image_mean=[0.0, 0.0, 0.0],
        image_std=[1.0, 1.0, 1.0],
    ):
        return YOLOWorldProcessor(
            model_dir=self.model_dir,
            image_target_size=image_target_size,
            image_mean=image_mean,
            image_std=image_std,
        )
