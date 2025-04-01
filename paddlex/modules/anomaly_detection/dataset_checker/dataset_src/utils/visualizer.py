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


import numpy as np

from ......utils.deps import function_requires_deps, is_dep_available

if is_dep_available("opencv-contrib-python"):
    import cv2


def get_color_map_list(length):
    """Returns the color map for visualizing the segmentation mask"""
    length += 1
    color_map = length * [0, 0, 0]
    for i in range(0, length):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= ((lab >> 0) & 1) << (7 - j)
            color_map[i * 3 + 1] |= ((lab >> 1) & 1) << (7 - j)
            color_map[i * 3 + 2] |= ((lab >> 2) & 1) << (7 - j)
            j += 1
            lab >>= 3
    color_map = color_map[3:]
    return color_map


@function_requires_deps("opencv-contrib-python")
def visualize(image, result, weight=0.6, use_multilabel=False):
    """Convert predict result to color image, and save added image."""
    color_map = get_color_map_list(256)
    color_map = [color_map[i : i + 3] for i in range(0, len(color_map), 3)]
    color_map = np.array(color_map).astype("uint8")

    if not use_multilabel:
        # Use OpenCV LUT for color mapping
        c1 = cv2.LUT(result, color_map[:, 0])
        c2 = cv2.LUT(result, color_map[:, 1])
        c3 = cv2.LUT(result, color_map[:, 2])
        pseudo_img = np.dstack((c3, c2, c1))

        vis_result = cv2.addWeighted(image, weight, pseudo_img, 1 - weight, 0)
    else:
        vis_result = image.copy()
        for i in range(result.shape[0]):
            mask = result[i]
            c1 = np.where(mask, color_map[i, 0], vis_result[..., 0])
            c2 = np.where(mask, color_map[i, 1], vis_result[..., 1])
            c3 = np.where(mask, color_map[i, 2], vis_result[..., 2])
            pseudo_img = np.dstack((c3, c2, c1)).astype("uint8")

            contour, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            vis_result = cv2.addWeighted(vis_result, weight, pseudo_img, 1 - weight, 0)
            contour_color = (
                int(color_map[i, 0]),
                int(color_map[i, 1]),
                int(color_map[i, 2]),
            )
            vis_result = cv2.drawContours(vis_result, contour, -1, contour_color, 1)

    return vis_result
