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

import numpy as np

from ...utils.benchmark import benchmark


@benchmark.timeit
class MultiLabelThreshOutput:
    """MultiLabelThresh Transform"""

    def __init__(self, class_ids=None, delimiter=None):
        super().__init__()
        self.delimiter = delimiter if delimiter is not None else " "
        self.class_id_map = self._parse_class_id_map(class_ids)

    def _parse_class_id_map(self, class_ids):
        """parse class id to label map file"""
        if class_ids is None:
            return None
        class_id_map = {id: str(lb) for id, lb in enumerate(class_ids)}
        return class_id_map

    def __call__(self, preds, threshold: Union[float, dict, list]):
        threshold_list = []
        num_classes = preds[0].shape[-1]
        if isinstance(threshold, float):
            threshold_list = [threshold for _ in range(num_classes)]
        elif isinstance(threshold, dict):
            if threshold.get("default") is None:
                raise ValueError(
                    "If using dictionary format, please specify default threshold explicitly with key 'default'."
                )
            default_threshold = threshold.get("default")
            threshold_list = [default_threshold for _ in range(num_classes)]
            for k, v in threshold.items():
                if k == "default":
                    continue
                if isinstance(k, str):
                    assert (
                        k.isdigit()
                    ), f"Invalid key of threshold: {k}, it must be integer"
                    k = int(k)
                if not isinstance(v, float):
                    raise ValueError(
                        f"Invalid value type of threshold: {type(v)}, it must be float"
                    )
                assert (
                    k < num_classes
                ), f"Invalid key of threshold: {k}, it must be less than the number of classes({num_classes})"
                threshold_list[k] = v
        elif isinstance(threshold, list):
            assert (
                len(threshold) == num_classes
            ), f"The length of threshold({len(threshold)}) should be equal to the number of classes({num_classes})."
            threshold_list = threshold
        else:
            raise ValueError(
                "Invalid type of threshold, should be 'list', 'dict' or 'float'."
            )

        pred_indexes = [
            np.argsort(-x[x > threshold])
            for x, threshold in zip(preds[0], threshold_list)
        ]
        indexes = [
            np.where(x > threshold)[0][indices]
            for x, indices, threshold in zip(preds[0], pred_indexes, threshold_list)
        ]
        scores = [
            np.around(pred[index].astype(float), decimals=5)
            for pred, index in zip(preds[0], indexes)
        ]
        label_names = [[self.class_id_map[i] for i in index] for index in indexes]
        return indexes, scores, label_names
