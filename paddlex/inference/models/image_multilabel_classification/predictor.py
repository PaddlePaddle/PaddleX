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

from ....modules.multilabel_classification.model_list import MODELS
from ..image_classification import ClasPredictor
from .processors import MultiLabelThreshOutput
from .result import MLClassResult


class MLClasPredictor(ClasPredictor):
    """MLClasPredictor that inherits from BasePredictor."""

    entities = MODELS

    def __init__(
        self,
        threshold: Union[float, dict, list, None] = None,
        *args: List,
        **kwargs: Dict
    ) -> None:
        """Initializes MLClasPredictor.

        Args:
            threshold (float, dict, optional): The threshold predictions to return. If None, it will be depending on config of inference or predict. Defaults to None.
            *args: Arbitrary positional arguments passed to the superclass.
            **kwargs: Arbitrary keyword arguments passed to the superclass.
        """
        self.threshold = threshold
        super().__init__(*args, **kwargs)

    def _get_result_class(self) -> type:
        """Returns the result class, MLClassResult.

        Returns:
            type: The MLClassResult class.
        """

        return MLClassResult

    def process(
        self,
        batch_data: List[Union[str, np.ndarray]],
        threshold: Union[int, dict, None] = None,
    ) -> Dict[str, Any]:
        """
        Process a batch of data through the preprocessing, inference, and postprocessing.

        Args:
            batch_data (List[Union[str, np.ndarray], ...]): A batch of input data (e.g., image file paths).
            threshold (float, dict, optional): The threshold predictions to return. If None, it will be depending on config of inference or predict. Defaults to None.

        Returns:
            dict: A dictionary containing the input path, raw image, class IDs, scores, and label names for every instance of the batch. Keys include 'input_path', 'input_img', 'class_ids', 'scores', and 'label_names'.
        """
        batch_raw_imgs = self.preprocessors["Read"](imgs=batch_data.instances)
        batch_imgs = self.preprocessors["Resize"](imgs=batch_raw_imgs)
        batch_imgs = self.preprocessors["Normalize"](imgs=batch_imgs)
        batch_imgs = self.preprocessors["ToCHW"](imgs=batch_imgs)
        x = self.preprocessors["ToBatch"](imgs=batch_imgs)
        batch_preds = self.infer(x=x)
        batch_class_ids, batch_scores, batch_label_names = self.postprocessors[
            "MultiLabelThreshOutput"
        ](
            preds=batch_preds,
            threshold=self.threshold if threshold is None else threshold,
        )
        return {
            "input_path": batch_data.input_paths,
            "page_index": batch_data.page_indexes,
            "input_img": batch_raw_imgs,
            "class_ids": batch_class_ids,
            "scores": batch_scores,
            "label_names": batch_label_names,
        }

    @ClasPredictor.register("MultiLabelThreshOutput")
    def build_threshoutput(self, threshold: Union[float, dict, list], label_list=None):
        if self.threshold is None:
            self.threshold = threshold
        return "MultiLabelThreshOutput", MultiLabelThreshOutput(class_ids=label_list)
