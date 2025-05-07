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

import copy
import math

import numpy as np
from PIL import Image

from ....utils.deps import function_requires_deps, is_dep_available
from ...common.result import BaseCVResult, JsonMixin

if is_dep_available("opencv-contrib-python"):
    import cv2
if is_dep_available("matplotlib"):
    import matplotlib.pyplot as plt


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color


@function_requires_deps("matplotlib", "opencv-contrib-python")
def draw_keypoints(img, results, visual_thresh=0.1, ids=None):
    plt.switch_backend("agg")
    skeletons = results["keypoints"]
    skeletons = np.array(skeletons)
    if len(skeletons) > 0:
        kpt_nums = skeletons.shape[1]
    if kpt_nums == 17:  # plot coco keypoint
        EDGES = [
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 4),
            (3, 5),
            (4, 6),
            (5, 7),
            (6, 8),
            (7, 9),
            (8, 10),
            (5, 11),
            (6, 12),
            (11, 13),
            (12, 14),
            (13, 15),
            (14, 16),
            (11, 12),
        ]
    else:  # plot mpii keypoint
        EDGES = [
            (0, 1),
            (1, 2),
            (3, 4),
            (4, 5),
            (2, 6),
            (3, 6),
            (6, 7),
            (7, 8),
            (8, 9),
            (10, 11),
            (11, 12),
            (13, 14),
            (14, 15),
            (8, 12),
            (8, 13),
        ]
    NUM_EDGES = len(EDGES)

    colors = [
        [255, 0, 0],
        [255, 85, 0],
        [255, 170, 0],
        [255, 255, 0],
        [170, 255, 0],
        [85, 255, 0],
        [0, 255, 0],
        [0, 255, 85],
        [0, 255, 170],
        [0, 255, 255],
        [0, 170, 255],
        [0, 85, 255],
        [0, 0, 255],
        [85, 0, 255],
        [170, 0, 255],
        [255, 0, 255],
        [255, 0, 170],
        [255, 0, 85],
    ]
    plt.figure()
    color_set = results["colors"] if "colors" in results else None

    if "bbox" in results and ids is None:
        bboxs = results["bbox"]
        for j, rect in enumerate(bboxs):
            xmin, ymin, xmax, ymax = rect
            color = (
                colors[0] if color_set is None else colors[color_set[j] % len(colors)]
            )
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 1)

    canvas = img.copy()
    for i in range(kpt_nums):
        for j in range(len(skeletons)):
            if skeletons[j][i, 2] < visual_thresh:
                continue
            if ids is None:
                color = (
                    colors[i]
                    if color_set is None
                    else colors[color_set[j] % len(colors)]
                )
            else:
                color = get_color(ids[j])

            cv2.circle(
                canvas,
                tuple(skeletons[j][i, 0:2].astype("int32")),
                2,
                color,
                thickness=-1,
            )

    stickwidth = 1

    for i in range(NUM_EDGES):
        for j in range(len(skeletons)):
            edge = EDGES[i]
            if (
                skeletons[j][edge[0], 2] < visual_thresh
                or skeletons[j][edge[1], 2] < visual_thresh
            ):
                continue

            cur_canvas = canvas.copy()
            X = [skeletons[j][edge[0], 1], skeletons[j][edge[1], 1]]
            Y = [skeletons[j][edge[0], 0], skeletons[j][edge[1], 0]]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly(
                (int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1
            )
            if ids is None:
                color = (
                    colors[i]
                    if color_set is None
                    else colors[color_set[j] % len(colors)]
                )
            else:
                color = get_color(ids[j])
            cv2.fillConvexPoly(cur_canvas, polygon, color)
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    plt.close()
    return canvas


class KptResult(BaseCVResult):
    """Save Result Transform"""

    def _to_img(self):
        """apply"""
        if "kpts" in self:  # for single module result
            keypoints = [kpt["keypoints"] for kpt in self["kpts"]]
        else:
            keypoints = [
                obj["keypoints"] for obj in self["boxes"]
            ]  # for top-down pipeline result
        image = self["input_img"]
        if keypoints:
            image = draw_keypoints(image, dict(keypoints=np.stack(keypoints)))
        image = Image.fromarray(image[..., ::-1])
        return {"res": image}

    def _to_str(self, *args, **kwargs):
        data = copy.deepcopy(self)
        data.pop("input_img")
        return JsonMixin._to_str(data, *args, **kwargs)

    def _to_json(self, *args, **kwargs):
        data = copy.deepcopy(self)
        data.pop("input_img")
        return JsonMixin._to_json(data, *args, **kwargs)
