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

from typing import Any, Dict, List, Union

import numpy as np

from ....modules.face_recognition.model_list import MODELS
from ..image_feature import ImageFeaturePredictor


class FaceFeaturePredictor(ImageFeaturePredictor):
    """FaceFeaturePredictor that inherits from ImageFeaturePredictor."""

    entities = MODELS

    def __init__(self, *args: List, flip: bool = False, **kwargs: Dict) -> None:
        """Initializes ClasPredictor.

        Args:
            *args: Arbitrary positional arguments passed to the superclass.
            flip: Whether to perform face flipping during inference. Default is False.
            **kwargs: Arbitrary keyword arguments passed to the superclass.
        """
        super().__init__(*args, **kwargs)
        self.flip = flip

    def process(self, batch_data: List[Union[str, np.ndarray]]) -> Dict[str, Any]:
        """
        Process a batch of data through the preprocessing, inference, and postprocessing.

        Args:
            batch_data (List[Union[str, np.ndarray], ...]): A batch of input data (e.g., image file paths).

        Returns:
            dict: A dictionary containing the input path, raw image, class IDs, scores, and label names for every instance of the batch. Keys include 'input_path', 'input_img', 'class_ids', 'scores', and 'label_names'.
        """
        batch_raw_imgs = self.preprocessors["Read"](imgs=batch_data.instances)
        batch_imgs = self.preprocessors["Resize"](imgs=batch_raw_imgs)
        batch_imgs = self.preprocessors["Normalize"](imgs=batch_imgs)
        batch_imgs = self.preprocessors["ToCHW"](imgs=batch_imgs)
        x = self.preprocessors["ToBatch"](imgs=batch_imgs)
        batch_preds = self.infer(x=x)
        if self.flip:
            batch_preds_flipped = self.infer(x=[np.flip(data, axis=3) for data in x])
            for i in range(len(batch_preds)):
                batch_preds[i] = batch_preds[i] + batch_preds_flipped[i]
        features = self.postprocessors["NormalizeFeatures"](batch_preds)

        return {
            "input_path": batch_data.input_paths,
            "page_index": batch_data.page_indexes,
            "input_img": batch_raw_imgs,
            "feature": features,
        }
