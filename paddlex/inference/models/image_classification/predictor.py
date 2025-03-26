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

from typing import Any, Dict, List, Tuple, Union

import numpy as np

from ....modules.image_classification.model_list import MODELS
from ....utils.func_register import FuncRegister
from ...common.batch_sampler import ImageBatchSampler
from ...common.reader import ReadImage
from ..base import BasePredictor
from ..common import Normalize, Resize, ResizeByShort, ToBatch, ToCHWImage
from .processors import Crop, Topk
from .result import TopkResult


class ClasPredictor(BasePredictor):
    """ClasPredictor that inherits from BasePredictor."""

    entities = MODELS

    _FUNC_MAP = {}
    register = FuncRegister(_FUNC_MAP)

    def __init__(
        self, topk: Union[int, None] = None, *args: List, **kwargs: Dict
    ) -> None:
        """Initializes ClasPredictor.

        Args:
            topk (int, optional): The number of top-k predictions to return. If None, it will be depending on config of inference or predict. Defaults to None.
            *args: Arbitrary positional arguments passed to the superclass.
            **kwargs: Arbitrary keyword arguments passed to the superclass.
        """
        super().__init__(*args, **kwargs)
        self.topk = topk
        self.preprocessors, self.infer, self.postprocessors = self._build()

    def _build_batch_sampler(self) -> ImageBatchSampler:
        """Builds and returns an ImageBatchSampler instance.

        Returns:
            ImageBatchSampler: An instance of ImageBatchSampler.
        """
        return ImageBatchSampler()

    def _get_result_class(self) -> type:
        """Returns the result class, TopkResult.

        Returns:
            type: The TopkResult class.
        """
        return TopkResult

    def _build(self) -> Tuple:
        """Build the preprocessors, inference engine, and postprocessors based on the configuration.

        Returns:
            tuple: A tuple containing the preprocessors, inference engine, and postprocessors.
        """
        preprocessors = {"Read": ReadImage(format="RGB")}
        for cfg in self.config["PreProcess"]["transform_ops"]:
            tf_key = list(cfg.keys())[0]
            func = self._FUNC_MAP[tf_key]
            args = cfg.get(tf_key, {})
            name, op = func(self, **args) if args else func(self)
            preprocessors[name] = op
        preprocessors["ToBatch"] = ToBatch()

        infer = self.create_static_infer()

        postprocessors = {}
        for key in self.config["PostProcess"]:
            func = self._FUNC_MAP.get(key)
            args = self.config["PostProcess"].get(key, {})
            name, op = func(self, **args) if args else func(self)
            postprocessors[name] = op
        return preprocessors, infer, postprocessors

    def process(
        self, batch_data: List[Union[str, np.ndarray]], topk: Union[int, None] = None
    ) -> Dict[str, Any]:
        """
        Process a batch of data through the preprocessing, inference, and postprocessing.

        Args:
            batch_data (List[Union[str, np.ndarray], ...]): A batch of input data (e.g., image file paths).
            topk: The number of top predictions to keep. If None, it will be depending on `self.topk`. Defaults to None.

        Returns:
            dict: A dictionary containing the input path, raw image, class IDs, scores, and label names for every instance of the batch. Keys include 'input_path', 'input_img', 'class_ids', 'scores', and 'label_names'.
        """
        batch_raw_imgs = self.preprocessors["Read"](imgs=batch_data.instances)
        batch_imgs = self.preprocessors["Resize"](imgs=batch_raw_imgs)
        if "Crop" in self.preprocessors:
            batch_imgs = self.preprocessors["Crop"](imgs=batch_imgs)
        batch_imgs = self.preprocessors["Normalize"](imgs=batch_imgs)
        batch_imgs = self.preprocessors["ToCHW"](imgs=batch_imgs)
        x = self.preprocessors["ToBatch"](imgs=batch_imgs)
        batch_preds = self.infer(x=x)
        batch_class_ids, batch_scores, batch_label_names = self.postprocessors["Topk"](
            batch_preds, topk=topk or self.topk
        )
        return {
            "input_path": batch_data.input_paths,
            "page_index": batch_data.page_indexes,
            "input_img": batch_raw_imgs,
            "class_ids": batch_class_ids,
            "scores": batch_scores,
            "label_names": batch_label_names,
        }

    @register("ResizeImage")
    # TODO(gaotingquan): backend & interpolation
    def build_resize(
        self, resize_short=None, size=None, backend="cv2", interpolation="LINEAR"
    ):
        assert resize_short or size
        if resize_short:
            op = ResizeByShort(
                target_short_edge=resize_short,
                size_divisor=None,
                interp=interpolation,
                backend=backend,
            )
        else:
            op = Resize(
                target_size=size,
                size_divisor=None,
                interp=interpolation,
                backend=backend,
            )
        return "Resize", op

    @register("CropImage")
    def build_crop(self, size=224):
        return "Crop", Crop(crop_size=size)

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
        return "Normalize", Normalize(scale=scale, mean=mean, std=std)

    @register("ToCHWImage")
    def build_to_chw(self):
        return "ToCHW", ToCHWImage()

    @register("Topk")
    def build_topk(self, topk, label_list=None):
        if not self.topk:
            self.topk = int(topk)
        return "Topk", Topk(class_ids=label_list)
