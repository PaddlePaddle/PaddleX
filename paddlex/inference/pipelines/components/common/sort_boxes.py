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

from typing import List

import numpy as np

from .base_operator import BaseOperator


class SortQuadBoxes(BaseOperator):
    """SortQuadBoxes Operator."""

    entities = "SortQuadBoxes"

    def __init__(self):
        """Initializes the class."""
        super().__init__()

    def __call__(self, dt_polys: List[np.ndarray]) -> np.ndarray:
        """
        Sort quad boxes in order from top to bottom, left to right
        args:
            dt_polys(ndarray):detected quad boxes with shape [4, 2]
        return:
            sorted boxes(ndarray) with shape [4, 2]
        """
        dt_boxes = np.array(dt_polys)
        num_boxes = dt_boxes.shape[0]
        sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
        _boxes = list(sorted_boxes)

        for i in range(num_boxes - 1):
            for j in range(i, -1, -1):
                if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and (
                    _boxes[j + 1][0][0] < _boxes[j][0][0]
                ):
                    tmp = _boxes[j]
                    _boxes[j] = _boxes[j + 1]
                    _boxes[j + 1] = tmp
                else:
                    break
        return _boxes


class SortPolyBoxes(BaseOperator):
    """SortPolyBoxes Operator."""

    entities = "SortPolyBoxes"

    def __init__(self):
        """Initializes the class."""
        super().__init__()

    def __call__(self, dt_polys: List[np.ndarray]) -> np.ndarray:
        """
        Sort poly boxes in order from top to bottom, left to right
        args:
            dt_polys(ndarray):detected poly boxes with a [N, 2] np.ndarray list
        return:
            sorted boxes(ndarray) with [N, 2] np.ndarray list
        """
        num_boxes = len(dt_polys)
        if num_boxes == 0:
            return dt_polys
        else:
            y_min_list = []
            for bno in range(num_boxes):
                y_min_list.append(min(dt_polys[bno][:, 1]))
            rank = np.argsort(np.array(y_min_list))
            dt_polys_rank = []
            for no in range(num_boxes):
                dt_polys_rank.append(dt_polys[rank[no]])
            return dt_polys_rank
