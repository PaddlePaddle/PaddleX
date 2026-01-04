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
from PIL import ImageDraw

from ......utils.deps import function_requires_deps, is_dep_available

if is_dep_available("pycocotools"):
    from pycocotools.coco import COCO


@function_requires_deps("pycocotools")
def draw_keypoint(image, coco_info: "COCO", img_id):
    """
    Draw keypoints on image for the COCO human keypoint dataset with 17 keypoints.

    Args:
        image: PIL.Image object
        coco_info: COCO object (from pycocotools.coco)
        img_id: Image ID

    Returns:
        image: PIL.Image object with keypoints and skeleton drawn
    """

    # Initialize the drawing context
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    image_size = image.size
    width = int(max(image_size) * 0.005)  # Line thickness for drawing

    # Define the skeleton connections based on COCO keypoint indexes
    skeleton = [
        [15, 13],
        [13, 11],
        [16, 14],
        [14, 12],
        [11, 12],
        [5, 11],
        [6, 12],
        [5, 6],
        [5, 7],
        [6, 8],
        [7, 9],
        [8, 10],
        [1, 2],
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 4],
        [3, 5],
        [4, 6],
    ]

    # Define colors for each keypoint (you can customize these colors)
    keypoint_colors = [
        (255, 0, 0),  # Nose
        (255, 85, 0),  # Left Eye
        (255, 170, 0),  # Right Eye
        (255, 255, 0),  # Left Ear
        (170, 255, 0),  # Right Ear
        (85, 255, 0),  # Left Shoulder
        (0, 255, 0),  # Right Shoulder
        (0, 255, 85),  # Left Elbow
        (0, 255, 170),  # Right Elbow
        (0, 255, 255),  # Left Wrist
        (0, 170, 255),  # Right Wrist
        (0, 85, 255),  # Left Hip
        (0, 0, 255),  # Right Hip
        (85, 0, 255),  # Left Knee
        (170, 0, 255),  # Right Knee
        (255, 0, 255),  # Left Ankle
        (255, 0, 170),  # Right Ankle
    ]

    # Get annotations for the image
    annotations = coco_info.loadAnns(coco_info.getAnnIds(imgIds=img_id))

    # Loop over each person annotation
    for ann in annotations:
        keypoints = ann.get("keypoints", [])
        if not keypoints:
            continue  # Skip if no keypoints are present

        # Reshape keypoints into (num_keypoints, 3)
        keypoints = np.array(keypoints).reshape(-1, 3)

        # Draw keypoints
        for idx, (x, y, v) in enumerate(keypoints):
            if v == 2:  # v=2 means the keypoint is labeled and visible
                radius = max(1, int(width / 2))
                x, y = float(x), float(y)
                color = keypoint_colors[idx % len(keypoint_colors)]
                draw.ellipse(
                    (x - radius, y - radius, x + radius, y + radius), fill=color
                )

        # Draw skeleton by connecting keypoints
        for sk in skeleton:
            kp1_idx, kp2_idx = sk[0], sk[1]
            x1, y1, v1 = keypoints[kp1_idx]
            x2, y2, v2 = keypoints[kp2_idx]
            if v1 == 2 and v2 == 2:
                # Both keypoints are visible
                x1, y1 = float(x1), float(y1)
                x2, y2 = float(x2), float(y2)
                draw.line(
                    (x1, y1, x2, y2),
                    fill=(0, 255, 0),  # Line color (you can customize)
                    width=width,
                )

    return image
