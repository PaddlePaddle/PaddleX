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

from ....modules.anomaly_detection.model_list import MODELS
from ....utils.func_register import FuncRegister
from ...common.batch_sampler import ImageBatchSampler
from ...common.reader import ReadImage
from ..base import BasePredictor
from ..common import Normalize, Resize, ToBatch, ToCHWImage
from .processors import MapToMask
from .result import UadResult


class UadPredictor(BasePredictor):
    """UadPredictor that inherits from BasePredictor."""

    entities = MODELS

    _FUNC_MAP = {}
    register = FuncRegister(_FUNC_MAP)

    def __init__(self, *args: List, **kwargs: Dict) -> None:
        """Initializes UadPredictor.

        Args:
            *args: Arbitrary positional arguments passed to the superclass.
            **kwargs: Arbitrary keyword arguments passed to the superclass.
        """
        super().__init__(*args, **kwargs)
        self.preprocessors, self.infer, self.postprocessors = self._build()

    def _build_batch_sampler(self) -> ImageBatchSampler:
        """Builds and returns an ImageBatchSampler instance.

        Returns:
            ImageBatchSampler: An instance of ImageBatchSampler.
        """
        return ImageBatchSampler()

    def _get_result_class(self) -> type:
        """Returns the result class, UadResult.

        Returns:
            type: The UadResult class.
        """
        return UadResult

    def _build(self) -> Tuple:
        """Build the preprocessors, inference engine, and postprocessors based on the configuration.

        Returns:
            tuple: A tuple containing the preprocessors, inference engine, and postprocessors.
        """
        preprocessors = {"Read": ReadImage(format="RGB")}
        preprocessors["ToCHW"] = ToCHWImage()
        for cfg in self.config["Deploy"]["transforms"]:
            tf_key = cfg.pop("type")
            func = self._FUNC_MAP[tf_key]
            args = cfg
            name, op = func(self, **args) if args else func(self)
            preprocessors[name] = op
        preprocessors["ToBatch"] = ToBatch()

        infer = self.create_static_infer()
        postprocessors = {"Map_to_mask": MapToMask()}
        return preprocessors, infer, postprocessors

    def process(self, batch_data: List[Union[str, np.ndarray]]) -> Dict[str, Any]:
        """
        Process a batch of data through the preprocessing, inference, and postprocessing.

        Args:
            batch_data (List[Union[str, np.ndarray], ...]): A batch of input data (e.g., image file paths).

        Returns:
            dict: A dictionary containing the input path, raw image, and predicted segmentation maps for every instance of the batch. Keys include 'input_path', 'input_img', and 'pred'.
        """
        batch_raw_imgs = self.preprocessors["Read"](imgs=batch_data.instances)
        batch_imgs = self.preprocessors["Resize"](imgs=batch_raw_imgs)
        batch_imgs = self.preprocessors["Normalize"](imgs=batch_imgs)
        batch_imgs = self.preprocessors["ToCHW"](imgs=batch_imgs)
        x = self.preprocessors["ToBatch"](imgs=batch_imgs)
        batch_preds = self.infer(x=x)
        batch_preds = self.postprocessors["Map_to_mask"](preds=batch_preds)
        if len(batch_data) > 1:
            batch_preds = np.split(batch_preds[0], len(batch_data), axis=0)

        return {
            "input_path": batch_data.input_paths,
            "page_index": batch_data.page_indexes,
            "input_img": batch_raw_imgs,
            "pred": batch_preds,
        }

    @register("Resize")
    def build_resize(
        self, target_size, keep_ratio=False, size_divisor=None, interp="LINEAR"
    ):
        assert target_size
        op = Resize(
            target_size=target_size,
            keep_ratio=keep_ratio,
            size_divisor=size_divisor,
            interp=interp,
        )
        return "Resize", op

    @register("Normalize")
    def build_normalize(
        self,
        mean=0.5,
        std=0.5,
    ):
        op = Normalize(mean=mean, std=std)
        return "Normalize", op

    @register("Map_to_mask")
    def map_to_mask(self, mask_map):
        op = MapToMask()
        return "Map_to_mask", op
