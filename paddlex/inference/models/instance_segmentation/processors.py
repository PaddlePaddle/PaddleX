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

from typing import List, Optional

from ...utils.benchmark import benchmark
from ..object_detection.processors import restructured_boxes


def extract_masks_from_boxes(boxes, masks):
    """
    Extracts the portion of each mask that is within the corresponding box.
    """
    new_masks = []

    for i, box in enumerate(boxes):
        x_min, y_min, x_max, y_max = box["coordinate"]
        x_min, y_min, x_max, y_max = map(
            lambda x: int(round(x)), [x_min, y_min, x_max, y_max]
        )

        cropped_mask = masks[i][y_min:y_max, x_min:x_max]
        new_masks.append(cropped_mask)

    return new_masks


@benchmark.timeit
class InstanceSegPostProcess(object):
    """Save Result Transform"""

    def __init__(self, threshold=0.5, labels=None):
        super().__init__()
        self.threshold = threshold
        self.labels = labels

    def apply(self, masks, img_size, boxes=None, class_id=None, threshold=None):
        """apply"""
        if boxes is not None:
            expect_boxes = (boxes[:, 1] > threshold) & (boxes[:, 0] > -1)
            boxes = boxes[expect_boxes, :]
            boxes = restructured_boxes(boxes, self.labels, img_size)
            masks = masks[expect_boxes, :, :]
            masks = extract_masks_from_boxes(boxes, masks)
            result = {"boxes": boxes, "masks": masks}
        else:
            mask_info = []
            class_id = [list(item) for item in zip(class_id[0], class_id[1])]

            selected_masks = []
            for i, info in enumerate(class_id):
                label_id = int(info[0])
                if info[1] < threshold:
                    continue
                mask_info.append(
                    {
                        "label": self.labels[label_id],
                        "score": info[1],
                        "class_id": label_id,
                    }
                )
                selected_masks.append(masks[i])
            result = {"boxes": mask_info, "masks": selected_masks}

        return result

    def __call__(
        self,
        batch_outputs: List[dict],
        datas: List[dict],
        threshold: Optional[float] = None,
    ):
        """Apply the post-processing to a batch of outputs.

        Args:
            batch_outputs (List[dict]): The list of detection outputs.
            datas (List[dict]): The list of input data.
            threshold: Optional[float]: object score threshold for postprocess.

        Returns:
            List[Boxes]: The list of post-processed detection boxes.
        """
        outputs = []
        for data, output in zip(datas, batch_outputs):
            boxes_masks = self.apply(
                img_size=data["ori_img_size"],
                **output,
                threshold=threshold if threshold is not None else self.threshold
            )
            outputs.append(boxes_masks)
        return outputs
