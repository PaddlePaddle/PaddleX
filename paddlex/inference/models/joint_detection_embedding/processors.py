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

from typing import List, Tuple, Union, Optional

import cv2
import numpy as np
from numpy import ndarray


Boxes = List[dict]
Number = Union[int, float]


class JDEPostProcess:
    """Save Result Transform

    This class is responsible for post-processing detection results, including
    thresholding, non-maximum suppression (NMS), and restructuring the boxes
    based on the input type (normal or rotated object detection).
    """

    def __init__(self, labels: Optional[List[str]] = None) -> None:
        """Initialize the DetPostProcess class.

        Args:
            threshold (float, optional): The threshold to apply to the detection scores. Defaults to 0.5.
            labels (Optional[List[str]], optional): The list of labels for the detection categories. Defaults to None.
            layout_postprocess (bool, optional): Whether to apply layout post-processing. Defaults to False.
        """
        super().__init__()
        self.labels = labels

    def apply(
        self,
        boxes: ndarray,
        img_size: Tuple[int, int],
        threshold: Union[float, dict],
    ) -> Boxes:
        """Apply post-processing to the detection boxes.

        Args:
            boxes (ndarray): The input detection boxes with scores.
            img_size (tuple): The original image size.

        Returns:
            Boxes: The post-processed detection boxes.
        """
        if isinstance(threshold, float):
            expect_boxes = (boxes[:, 1] > threshold) & (boxes[:, 0] > -1)
            boxes = boxes[expect_boxes, :]
        elif isinstance(threshold, dict):
            category_filtered_boxes = []
            for cat_id in np.unique(boxes[:, 0]):
                category_boxes = boxes[boxes[:, 0] == cat_id]
                category_threshold = threshold.get(int(cat_id), 0.5)
                selected_indices = (category_boxes[:, 1] > category_threshold) & (
                    category_boxes[:, 0] > -1
                )
                category_filtered_boxes.append(category_boxes[selected_indices])
            boxes = (
                np.vstack(category_filtered_boxes)
                if category_filtered_boxes
                else np.array([])
            )

        return boxes

    def __call__(
        self,
        batch_outputs: List[dict],
        datas: List[dict],
        threshold: Optional[Union[float, dict]] = None,
    ) -> List[Boxes]:
        """Apply the post-processing to a batch of outputs.

        Args:
            batch_outputs (List[dict]): The list of detection outputs.
            datas (List[dict]): The list of input data.

        Returns:
            List[Boxes]: The list of post-processed detection boxes.
        """
        outputs = []
        for data, output in zip(datas, batch_outputs):
            boxes = self.apply(
                output["boxes"],
                data["ori_img_size"],
                threshold,
            )
            outputs.append(boxes)
        return outputs
