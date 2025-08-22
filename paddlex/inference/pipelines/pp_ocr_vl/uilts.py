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
from copy import deepcopy
from typing import Dict, List

import numpy as np
from PIL import Image

from ..layout_parsing.utils import (
    calculate_bbox_area,
    calculate_overlap_ratio,
    calculate_projection_overlap_ratio,
)


def filter_overlap_boxes(
    layout_det_res: Dict[str, List[Dict]]
) -> Dict[str, List[Dict]]:
    """
    Filter out overlapping boxes from layout detection results based on overlap ratio.

    Args:
        layout_det_res (Dict[str, List[Dict]]): Dictionary containing detection results with 'boxes' key.

    Returns:
        Dict[str, List[Dict]]: Filtered layout detection results with overlapping boxes removed.
    """
    layout_det_res_filted = deepcopy(layout_det_res)
    boxes = layout_det_res_filted["boxes"]
    dropped_indexes = set()

    # Iterate over each pair of boxes to find overlaps
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            # Skip boxes that are already marked for removal
            if i in dropped_indexes or j in dropped_indexes:
                continue

            # Calculate the overlap ratio
            overlap_ratio = calculate_overlap_ratio(
                boxes[i]["coordinate"], boxes[j]["coordinate"], "small"
            )

            # If overlap ratio is significant, mark one of the boxes for removal
            if (
                overlap_ratio == 1
            ):  # Assuming 1 is the threshold for significant overlap
                # Here we are assuming higher score is preferable, you might want to adjust this logic
                box_area_i = calculate_bbox_area(boxes[i]["coordinate"])
                box_area_j = calculate_bbox_area(boxes[j]["coordinate"])
                if (
                    boxes[i]["label"] == "image" or boxes[j]["label"] == "image"
                ) and boxes[i]["label"] != boxes[j]["label"]:
                    continue
                if box_area_i >= box_area_j:
                    dropped_indexes.add(j)
                else:
                    dropped_indexes.add(i)

    # Remove marked boxes
    layout_det_res_filted["boxes"] = [
        box for idx, box in enumerate(boxes) if idx not in dropped_indexes
    ]

    return layout_det_res_filted


def merge_images(images):
    """
    Merge a list of images (np.array) into a single image (PIL.Image).
    """
    if not images:
        return None

    # Calculate total height and max width
    total_height = sum(
        image.shape[0] for image in images
    )  # image.shape[0] is the height
    max_width = max(image.shape[1] for image in images)  # image.shape[1] is the width

    # Create a new blank image with white background
    new_image = Image.new("RGB", (max_width, total_height), (255, 255, 255))

    current_height = 0
    for image in images:
        pil_image = Image.fromarray(image)  # Convert np.array to PIL.Image
        x_offset = (max_width - pil_image.width) // 2
        new_image.paste(pil_image, (x_offset, current_height))
        current_height += pil_image.height

    return np.array(new_image)


def merge_blocks(blocks, non_merge_labels):
    current_group_images = []
    current_group_label = None
    crossing = False
    group_index = 0

    for i, block in enumerate(blocks):
        block_img = block["img"]
        block_bbox = block["box"]
        block_label = block["label"]

        if block_label in non_merge_labels:
            # If the current block's label is in the non-merge list, reset grouping
            if current_group_images:
                merged_image = merge_images(current_group_images)
                for j in range(group_index, i):
                    if j == group_index:
                        blocks[j]["img"] = merged_image
                    else:
                        blocks[j]["img"] = None
            # Reset for the non-merge block
            blocks[i]["img"] = block_img
            current_group_images = []
            current_group_label = None
            crossing = False
            group_index = i + 1
            continue

        if not current_group_images:
            current_group_images = [block_img]
            current_group_label = block_label
            crossing = False
            continue

        prev_block = blocks[i - 1]
        iou = calculate_projection_overlap_ratio(
            block_bbox, prev_block["box"], "horizontal"
        )

        if iou == 0 and block_label == current_group_label:
            current_group_images.append(block_img)
            crossing = True
        else:
            if crossing:
                merged_image = merge_images(current_group_images)
                for j in range(group_index, i):
                    if j == group_index:
                        blocks[j]["img"] = merged_image
                    else:
                        blocks[j]["img"] = None
                group_index = i
                current_group_images = [block_img]
                current_group_label = block_label
                crossing = False
            else:
                if iou > 0 and block_label == current_group_label:
                    current_group_images.append(block_img)
                else:
                    merged_image = merge_images(current_group_images)
                    for j in range(group_index, i):
                        if j == group_index:
                            blocks[j]["img"] = merged_image
                        else:
                            blocks[j]["img"] = None
                    group_index = i
                    current_group_images = [block_img]
                    current_group_label = block_label
                    crossing = False

    if current_group_images:
        merged_image = merge_images(current_group_images)
        for j in range(group_index, len(blocks)):
            if j == group_index:
                blocks[j]["img"] = merged_image
            else:
                blocks[j]["img"] = None

    return blocks
