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

import os

import numpy as np

from ...common.result import BaseResult
from .visualizer_3d import Visualizer3D


class BEV3DDetResult(BaseResult):
    """Base class for computer vision results."""

    def __init__(self, data: dict) -> None:
        """
        Initialize the BaseCVResult.

        Args:
            data (dict): The initial data.

        Raises:
            AssertionError: If the required key (`BaseCVResult.INPUT_IMG_KEY`) are not found in the data.
        """

        super().__init__(data)

    def visualize(self, save_path: str, show: bool) -> None:
        # input point cloud
        assert "input_path" in self.keys(), "input_path is not found in the data"
        points = np.fromfile(self["input_path"], dtype=np.float32)
        points = points.reshape(-1, 5)
        points = points[:, :4]

        # detection result
        result = dict()
        assert "boxes_3d" in self.keys(), "boxes_3d is not found in the data"
        result["bbox3d"] = self["boxes_3d"]
        assert "scores_3d" in self.keys(), "scores_3d is not found in the data"
        result["scores"] = self["scores_3d"]
        assert "labels_3d" in self.keys(), "labels_3d is not found in the data"
        result["labels"] = self["labels_3d"]

        if save_path is not None:
            # save result for local visualization
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            np.save(os.path.join(save_path, "results.npy"), result)
            np.save(os.path.join(save_path, "points.npy"), points)

        if show:
            # visualize
            score_threshold = 0.25
            vis = Visualizer3D()
            vis.draw_results(points, result, score_threshold)

        return
