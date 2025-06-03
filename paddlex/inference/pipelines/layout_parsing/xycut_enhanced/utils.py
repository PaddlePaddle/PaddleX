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

from typing import List, Tuple

import numpy as np

from ..result_v2 import LayoutParsingBlock, LayoutParsingRegion
from ..setting import BLOCK_LABEL_MAP, XYCUT_SETTINGS
from ..utils import calculate_projection_overlap_ratio


def get_nearest_edge_distance(
    bbox1: List[int],
    bbox2: List[int],
    weight: List[float] = [1.0, 1.0, 1.0, 1.0],
) -> Tuple[float]:
    """
    Calculate the nearest edge distance between two bounding boxes, considering directional weights.

    Args:
        bbox1 (list): The bounding box coordinates [x1, y1, x2, y2] of the input object.
        bbox2 (list): The bounding box coordinates [x1', y1', x2', y2'] of the object to match against.
        weight (list, optional): directional weights for the edge distances [left, right, up, down]. Defaults to [1, 1, 1, 1].

    Returns:
        float: The calculated minimum edge distance between the bounding boxes.
    """
    x1, y1, x2, y2 = bbox1
    x1_prime, y1_prime, x2_prime, y2_prime = bbox2
    min_x_distance, min_y_distance = 0, 0
    horizontal_iou = calculate_projection_overlap_ratio(bbox1, bbox2, "horizontal")
    vertical_iou = calculate_projection_overlap_ratio(bbox1, bbox2, "vertical")
    if horizontal_iou > 0 and vertical_iou > 0:
        return 0.0
    if horizontal_iou == 0:
        min_x_distance = min(abs(x1 - x2_prime), abs(x2 - x1_prime)) * (
            weight[0] if x2 < x1_prime else weight[1]
        )
    if vertical_iou == 0:
        min_y_distance = min(abs(y1 - y2_prime), abs(y2 - y1_prime)) * (
            weight[2] if y2 < y1_prime else weight[3]
        )

    return min_x_distance + min_y_distance


def projection_by_bboxes(boxes: np.ndarray, axis: int) -> np.ndarray:
    """
    Generate a 1D projection histogram from bounding boxes along a specified axis.

    Args:
        boxes: A (N, 4) array of bounding boxes defined by [x_min, y_min, x_max, y_max].
        axis: Axis for projection; 0 for horizontal (x-axis), 1 for vertical (y-axis).

    Returns:
        A 1D numpy array representing the projection histogram based on bounding box intervals.
    """
    assert axis in [0, 1]

    if np.min(boxes[:, axis::2]) < 0:
        max_length = abs(np.min(boxes[:, axis::2]))
    else:
        max_length = np.max(boxes[:, axis::2])

    projection = np.zeros(max_length, dtype=int)

    # Increment projection histogram over the interval defined by each bounding box
    for start, end in boxes[:, axis::2]:
        start = abs(start)
        end = abs(end)
        projection[start:end] += 1

    return projection


def split_projection_profile(arr_values: np.ndarray, min_value: float, min_gap: float):
    """
    Split the projection profile into segments based on specified thresholds.

    Args:
        arr_values: 1D array representing the projection profile.
        min_value: Minimum value threshold to consider a profile segment significant.
        min_gap: Minimum gap width to consider a separation between segments.

    Returns:
        A tuple of start and end indices for each segment that meets the criteria.
    """
    # Identify indices where the projection exceeds the minimum value
    significant_indices = np.where(arr_values > min_value)[0]
    if not len(significant_indices):
        return

    # Calculate gaps between significant indices
    index_diffs = significant_indices[1:] - significant_indices[:-1]
    gap_indices = np.where(index_diffs > min_gap)[0]

    # Determine start and end indices of segments
    segment_starts = np.insert(
        significant_indices[gap_indices + 1],
        0,
        significant_indices[0],
    )
    segment_ends = np.append(
        significant_indices[gap_indices],
        significant_indices[-1] + 1,
    )

    return segment_starts, segment_ends


def recursive_yx_cut(
    boxes: np.ndarray, indices: List[int], res: List[int], min_gap: int = 1
):
    """
    Recursively project and segment bounding boxes, starting with Y-axis and followed by X-axis.

    Args:
        boxes: A (N, 4) array representing bounding boxes.
        indices: List of indices indicating the original position of boxes.
        res: List to store indices of the final segmented bounding boxes.
        min_gap (int): Minimum gap width to consider a separation between segments on the X-axis. Defaults to 1.

    Returns:
        None: This function modifies the `res` list in place.
    """
    assert len(boxes) == len(
        indices
    ), "The length of boxes and indices must be the same."

    # Sort by y_min for Y-axis projection
    y_sorted_indices = boxes[:, 1].argsort()
    y_sorted_boxes = boxes[y_sorted_indices]
    y_sorted_indices = np.array(indices)[y_sorted_indices]

    # Perform Y-axis projection
    y_projection = projection_by_bboxes(boxes=y_sorted_boxes, axis=1)
    y_intervals = split_projection_profile(y_projection, 0, 1)

    if not y_intervals:
        return

    # Process each segment defined by Y-axis projection
    for y_start, y_end in zip(*y_intervals):
        # Select boxes within the current y interval
        y_interval_indices = (y_start <= y_sorted_boxes[:, 1]) & (
            y_sorted_boxes[:, 1] < y_end
        )
        y_boxes_chunk = y_sorted_boxes[y_interval_indices]
        y_indices_chunk = y_sorted_indices[y_interval_indices]

        # Sort by x_min for X-axis projection
        x_sorted_indices = y_boxes_chunk[:, 0].argsort()
        x_sorted_boxes_chunk = y_boxes_chunk[x_sorted_indices]
        x_sorted_indices_chunk = y_indices_chunk[x_sorted_indices]

        # Perform X-axis projection
        x_projection = projection_by_bboxes(boxes=x_sorted_boxes_chunk, axis=0)
        x_intervals = split_projection_profile(x_projection, 0, min_gap)

        if not x_intervals:
            continue

        # If X-axis cannot be further segmented, add current indices to results
        if len(x_intervals[0]) == 1:
            res.extend(x_sorted_indices_chunk)
            continue

        if np.min(x_sorted_boxes_chunk[:, 0]) < 0:
            x_intervals = np.flip(x_intervals, axis=1)
        # Recursively process each segment defined by X-axis projection
        for x_start, x_end in zip(*x_intervals):
            x_interval_indices = (x_start <= abs(x_sorted_boxes_chunk[:, 0])) & (
                abs(x_sorted_boxes_chunk[:, 0]) < x_end
            )
            recursive_yx_cut(
                x_sorted_boxes_chunk[x_interval_indices],
                x_sorted_indices_chunk[x_interval_indices],
                res,
            )


def recursive_xy_cut(
    boxes: np.ndarray, indices: List[int], res: List[int], min_gap: int = 1
):
    """
    Recursively performs X-axis projection followed by Y-axis projection to segment bounding boxes.

    Args:
        boxes: A (N, 4) array representing bounding boxes with [x_min, y_min, x_max, y_max].
        indices: A list of indices representing the position of boxes in the original data.
        res: A list to store indices of bounding boxes that meet the criteria.
        min_gap (int): Minimum gap width to consider a separation between segments on the X-axis. Defaults to 1.

    Returns:
        None: This function modifies the `res` list in place.
    """
    # Ensure boxes and indices have the same length
    assert len(boxes) == len(
        indices
    ), "The length of boxes and indices must be the same."

    # Sort by x_min to prepare for X-axis projection
    x_sorted_indices = boxes[:, 0].argsort()
    x_sorted_boxes = boxes[x_sorted_indices]
    x_sorted_indices = np.array(indices)[x_sorted_indices]

    # Perform X-axis projection
    x_projection = projection_by_bboxes(boxes=x_sorted_boxes, axis=0)
    x_intervals = split_projection_profile(x_projection, 0, 1)

    if not x_intervals:
        return

    if np.min(x_sorted_boxes[:, 0]) < 0:
        x_intervals = np.flip(x_intervals, axis=1)
    # Process each segment defined by X-axis projection
    for x_start, x_end in zip(*x_intervals):
        # Select boxes within the current x interval
        x_interval_indices = (x_start <= abs(x_sorted_boxes[:, 0])) & (
            abs(x_sorted_boxes[:, 0]) < x_end
        )
        x_boxes_chunk = x_sorted_boxes[x_interval_indices]
        x_indices_chunk = x_sorted_indices[x_interval_indices]

        # Sort selected boxes by y_min to prepare for Y-axis projection
        y_sorted_indices = x_boxes_chunk[:, 1].argsort()
        y_sorted_boxes_chunk = x_boxes_chunk[y_sorted_indices]
        y_sorted_indices_chunk = x_indices_chunk[y_sorted_indices]

        # Perform Y-axis projection
        y_projection = projection_by_bboxes(boxes=y_sorted_boxes_chunk, axis=1)
        y_intervals = split_projection_profile(y_projection, 0, min_gap)

        if not y_intervals:
            continue

        # If Y-axis cannot be further segmented, add current indices to results
        if len(y_intervals[0]) == 1:
            res.extend(y_sorted_indices_chunk)
            continue

        # Recursively process each segment defined by Y-axis projection
        for y_start, y_end in zip(*y_intervals):
            y_interval_indices = (y_start <= y_sorted_boxes_chunk[:, 1]) & (
                y_sorted_boxes_chunk[:, 1] < y_end
            )
            recursive_xy_cut(
                y_sorted_boxes_chunk[y_interval_indices],
                y_sorted_indices_chunk[y_interval_indices],
                res,
            )


def reference_insert(
    block: LayoutParsingBlock,
    sorted_blocks: List[LayoutParsingBlock],
    **kwargs,
):
    """
    Insert reference block into sorted blocks based on the distance between the block and the nearest sorted block.

    Args:
        block: The block to insert into the sorted blocks.
        sorted_blocks: The sorted blocks where the new block will be inserted.
        config: Configuration dictionary containing parameters related to the layout parsing.
        median_width: Median width of the document. Defaults to 0.0.

    Returns:
        sorted_blocks: The updated sorted blocks after insertion.
    """
    min_distance = float("inf")
    nearest_sorted_block_index = 0
    for sorted_block_idx, sorted_block in enumerate(sorted_blocks):
        if sorted_block.bbox[3] <= block.bbox[1]:
            distance = -(sorted_block.bbox[2] * 10 + sorted_block.bbox[3])
        if distance < min_distance:
            min_distance = distance
            nearest_sorted_block_index = sorted_block_idx

    sorted_blocks.insert(nearest_sorted_block_index + 1, block)
    return sorted_blocks


def manhattan_insert(
    block: LayoutParsingBlock,
    sorted_blocks: List[LayoutParsingBlock],
    **kwargs,
):
    """
    Insert a block into a sorted list of blocks based on the Manhattan distance between the block and the nearest sorted block.

    Args:
        block: The block to insert into the sorted blocks.
        sorted_blocks: The sorted blocks where the new block will be inserted.
        config: Configuration dictionary containing parameters related to the layout parsing.
        median_width: Median width of the document. Defaults to 0.0.

    Returns:
        sorted_blocks: The updated sorted blocks after insertion.
    """
    min_distance = float("inf")
    nearest_sorted_block_index = 0
    for sorted_block_idx, sorted_block in enumerate(sorted_blocks):
        distance = _manhattan_distance(block.bbox, sorted_block.bbox)
        if distance < min_distance:
            min_distance = distance
            nearest_sorted_block_index = sorted_block_idx

    sorted_blocks.insert(nearest_sorted_block_index + 1, block)
    return sorted_blocks


def weighted_distance_insert(
    block: LayoutParsingBlock,
    sorted_blocks: List[LayoutParsingBlock],
    region: LayoutParsingRegion,
):
    """
    Insert a block into a sorted list of blocks based on the weighted distance between the block and the nearest sorted block.

    Args:
        block: The block to insert into the sorted blocks.
        sorted_blocks: The sorted blocks where the new block will be inserted.
        config: Configuration dictionary containing parameters related to the layout parsing.
        median_width: Median width of the document. Defaults to 0.0.

    Returns:
        sorted_blocks: The updated sorted blocks after insertion.
    """

    tolerance_len = XYCUT_SETTINGS["edge_distance_compare_tolerance_len"]
    x1, y1, x2, y2 = block.bbox
    min_weighted_distance, min_edge_distance, min_up_edge_distance = (
        float("inf"),
        float("inf"),
        float("inf"),
    )
    nearest_sorted_block_index = 0
    for sorted_block_idx, sorted_block in enumerate(sorted_blocks):

        x1_prime, y1_prime, x2_prime, y2_prime = sorted_block.bbox

        # Calculate edge distance
        weight = _get_weights(block.order_label, block.direction)
        edge_distance = get_nearest_edge_distance(block.bbox, sorted_block.bbox, weight)

        if block.label in BLOCK_LABEL_MAP["doc_title_labels"]:
            disperse = max(1, region.text_line_width)
            tolerance_len = max(tolerance_len, disperse)
        if block.label == "abstract":
            tolerance_len *= 2
            edge_distance = max(0.1, edge_distance) * 10

        # Calculate up edge distances
        up_edge_distance = y1_prime if region.direction == "horizontal" else -x2_prime
        left_edge_distance = x1_prime if region.direction == "horizontal" else y1_prime
        is_below_sorted_block = (
            y2_prime < y1 if region.direction == "horizontal" else x1_prime > x2
        )

        if (
            block.label not in BLOCK_LABEL_MAP["unordered_labels"]
            or block.label in BLOCK_LABEL_MAP["doc_title_labels"]
            or block.label in BLOCK_LABEL_MAP["paragraph_title_labels"]
            or block.label in BLOCK_LABEL_MAP["vision_labels"]
        ) and is_below_sorted_block:
            up_edge_distance = -up_edge_distance
            left_edge_distance = -left_edge_distance

        if abs(min_up_edge_distance - up_edge_distance) <= tolerance_len:
            up_edge_distance = min_up_edge_distance

        # Calculate weighted distance
        weighted_distance = (
            +edge_distance
            * XYCUT_SETTINGS["distance_weight_map"].get("edge_weight", 10**4)
            + up_edge_distance
            * XYCUT_SETTINGS["distance_weight_map"].get("up_edge_weight", 1)
            + left_edge_distance
            * XYCUT_SETTINGS["distance_weight_map"].get("left_edge_weight", 0.0001)
        )

        min_edge_distance = min(edge_distance, min_edge_distance)
        min_up_edge_distance = min(up_edge_distance, min_up_edge_distance)

        if weighted_distance < min_weighted_distance:
            nearest_sorted_block_index = sorted_block_idx
            min_weighted_distance = weighted_distance
            if y1 > y1_prime or (y1 == y1_prime and x1 > x1_prime):
                nearest_sorted_block_index = sorted_block_idx + 1

    sorted_blocks.insert(nearest_sorted_block_index, block)
    return sorted_blocks


def insert_child_blocks(
    block: LayoutParsingBlock,
    block_idx: int,
    sorted_blocks: List[LayoutParsingBlock],
) -> List[LayoutParsingBlock]:
    """
    Insert child blocks of a block into the sorted blocks list.

    Args:
        block: The parent block whose child blocks need to be inserted.
        block_idx: Index at which the parent block exists in the sorted blocks list.
        sorted_blocks: Sorted blocks list where the child blocks are to be inserted.

    Returns:
        sorted_blocks: Updated sorted blocks list after inserting child blocks.
    """
    if block.child_blocks:
        sub_blocks = block.get_child_blocks()
        sub_blocks.append(block)
        sub_blocks = sort_child_blocks(sub_blocks, sub_blocks[0].direction)
        sorted_blocks[block_idx] = sub_blocks[0]
        for block in sub_blocks[1:]:
            block_idx += 1
            sorted_blocks.insert(block_idx, block)
    return sorted_blocks


def sort_child_blocks(blocks, direction="horizontal") -> List[LayoutParsingBlock]:
    """
    Sort child blocks based on their bounding box coordinates.

    Args:
        blocks: A list of LayoutParsingBlock objects representing the child blocks.
        direction: direction of the blocks ('horizontal' or 'vertical'). Default is 'horizontal'.
    Returns:
        sorted_blocks: A sorted list of LayoutParsingBlock objects.
    """
    if direction == "horizontal":
        # from top to bottom
        blocks.sort(
            key=lambda x: (
                x.bbox[1],  # y_min
                x.bbox[0],  # x_min
                x.bbox[1] ** 2 + x.bbox[0] ** 2,  # distance with (0,0)
            ),
        )
    else:
        # from right to left
        blocks.sort(
            key=lambda x: (
                -x.bbox[0],  # x_min
                x.bbox[1],  # y_min
                x.bbox[1] ** 2 - x.bbox[0] ** 2,  # distance with (max,0)
            ),
        )
    return blocks


def _get_weights(label, direction="horizontal"):
    """Define weights based on the label and direction."""
    if label == "doc_title":
        return (
            [1, 0.1, 0.1, 1] if direction == "horizontal" else [0.2, 0.1, 1, 1]
        )  # left-down ,  right-left
    elif label in [
        "paragraph_title",
        "table_title",
        "abstract",
        "image",
        "seal",
        "chart",
        "figure",
    ]:
        return [1, 1, 0.1, 1]  # down
    else:
        return [1, 1, 1, 0.1]  # up


def _manhattan_distance(
    point1: Tuple[float, float],
    point2: Tuple[float, float],
    weight_x: float = 1.0,
    weight_y: float = 1.0,
) -> float:
    """
    Calculate the weighted Manhattan distance between two points.

    Args:
        point1 (Tuple[float, float]): The first point as (x, y).
        point2 (Tuple[float, float]): The second point as (x, y).
        weight_x (float): The weight for the x-axis distance. Default is 1.0.
        weight_y (float): The weight for the y-axis distance. Default is 1.0.

    Returns:
        float: The weighted Manhattan distance between the two points.
    """
    return weight_x * abs(point1[0] - point2[0]) + weight_y * abs(point1[1] - point2[1])


def sort_normal_blocks(blocks, text_line_height, text_line_width, region_direction):
    if region_direction == "horizontal":
        blocks.sort(
            key=lambda x: (
                x.bbox[1] // text_line_height,
                x.bbox[0] // text_line_width,
                x.bbox[1] ** 2 + x.bbox[0] ** 2,
            ),
        )
    else:
        blocks.sort(
            key=lambda x: (
                -x.bbox[0] // text_line_width,
                x.bbox[1] // text_line_height,
                x.bbox[1] ** 2 - x.bbox[2] ** 2,  # distance with (max,0)
            ),
        )
    return blocks


def sort_normal_blocks(blocks, text_line_height, text_line_width, region_direction):
    if region_direction == "horizontal":
        blocks.sort(
            key=lambda x: (
                x.bbox[1] // text_line_height,
                x.bbox[0] // text_line_width,
                x.bbox[1] ** 2 + x.bbox[0] ** 2,
            ),
        )
    else:
        blocks.sort(
            key=lambda x: (
                -x.bbox[0] // text_line_width,
                x.bbox[1] // text_line_height,
                -(x.bbox[2] ** 2 + x.bbox[1] ** 2),
            ),
        )
    return blocks


def get_cut_blocks(blocks, cut_direction, cut_coordinates, mask_labels=[]):
    """
    Cut blocks based on the given cut direction and coordinates.

    Args:
        blocks (list): list of blocks to be cut.
        cut_direction (str): cut direction, either "horizontal" or "vertical".
        cut_coordinates (list): list of cut coordinates.

    Returns:
        list: a list of tuples containing the cutted blocks and their corresponding mean width。
    """
    cuted_list = []
    # filter out mask blocks,including header, footer, unordered and child_blocks

    # 0: horizontal, 1: vertical
    cut_aixis = 0 if cut_direction == "horizontal" else 1
    blocks.sort(key=lambda x: x.bbox[cut_aixis + 2])
    cut_coordinates.append(float("inf"))

    cut_coordinates = list(set(cut_coordinates))
    cut_coordinates.sort()

    cut_idx = 0
    for cut_coordinate in cut_coordinates:
        group_blocks = []
        block_idx = cut_idx
        while block_idx < len(blocks):
            block = blocks[block_idx]
            if block.bbox[cut_aixis + 2] > cut_coordinate:
                break
            elif block.order_label not in mask_labels:
                group_blocks.append(block)
            block_idx += 1
        cut_idx = block_idx
        if group_blocks:
            cuted_list.append(group_blocks)

    return cuted_list


def add_split_block(
    blocks: List[LayoutParsingBlock], region_bbox: List[int]
) -> List[LayoutParsingBlock]:
    block_bboxes = np.array([block.bbox for block in blocks])
    discontinuous = calculate_discontinuous_projection(
        block_bboxes, direction="vertical"
    )
    current_interval = discontinuous[0]
    for interval in discontinuous[1:]:
        gap_len = interval[0] - current_interval[1]
        if gap_len > 40:
            x1, _, x2, __ = region_bbox
            y1 = current_interval[1] + 5
            y2 = interval[0] - 5
            bbox = [x1, y1, x2, y2]
            split_block = LayoutParsingBlock(label="split", bbox=bbox)
            blocks.append(split_block)
        current_interval = interval


def get_nearest_blocks(
    block: LayoutParsingBlock,
    ref_blocks: List[LayoutParsingBlock],
    overlap_threshold,
    direction="horizontal",
) -> List:
    """
    Get the adjacent blocks with the same direction as the current block.
    Args:
        block (LayoutParsingBlock): The current block.
        blocks (List[LayoutParsingBlock]): A list of all blocks.
        ref_block_idxes (List[int]): A list of indices of reference blocks.
        iou_threshold (float): The IOU threshold to determine if two blocks are considered adjacent.
    Returns:
        Int: The index of the previous block with same direction.
        Int: The index of the following block with same direction.
    """
    prev_blocks: List[LayoutParsingBlock] = []
    post_blocks: List[LayoutParsingBlock] = []
    sort_index = 1 if direction == "horizontal" else 0
    for ref_block in ref_blocks:
        if ref_block.index == block.index:
            continue
        overlap_ratio = calculate_projection_overlap_ratio(
            block.bbox, ref_block.bbox, direction, mode="small"
        )
        if overlap_ratio > overlap_threshold:
            if ref_block.bbox[sort_index] <= block.bbox[sort_index]:
                prev_blocks.append(ref_block)
            else:
                post_blocks.append(ref_block)

    if prev_blocks:
        prev_blocks.sort(key=lambda x: x.bbox[sort_index], reverse=True)
    if post_blocks:
        post_blocks.sort(key=lambda x: x.bbox[sort_index])

    return prev_blocks, post_blocks


def get_adjacent_blocks_by_direction(
    blocks: List[LayoutParsingBlock],
    block_idx: int,
    ref_block_idxes: List[int],
    iou_threshold,
) -> List:
    """
    Get the adjacent blocks with the same direction as the current block.
    Args:
        block (LayoutParsingBlock): The current block.
        blocks (List[LayoutParsingBlock]): A list of all blocks.
        ref_block_idxes (List[int]): A list of indices of reference blocks.
        iou_threshold (float): The IOU threshold to determine if two blocks are considered adjacent.
    Returns:
        Int: The index of the previous block with same direction.
        Int: The index of the following block with same direction.
    """
    min_prev_block_distance = float("inf")
    prev_block_index = None
    min_post_block_distance = float("inf")
    post_block_index = None
    block = blocks[block_idx]
    child_labels = [
        "vision_footnote",
        "sub_paragraph_title",
        "doc_title_text",
        "vision_title",
    ]

    # find the nearest text block with same direction to the current block
    for ref_block_idx in ref_block_idxes:
        ref_block = blocks[ref_block_idx]
        ref_block_direction = ref_block.direction
        if ref_block.order_label in child_labels:
            continue
        match_block_iou = calculate_projection_overlap_ratio(
            block.bbox,
            ref_block.bbox,
            ref_block_direction,
        )

        child_match_distance_tolerance_len = block.short_side_length / 10

        if block.order_label == "vision":
            if ref_block.num_of_lines == 1:
                gap_tolerance_len = ref_block.short_side_length * 2
            else:
                gap_tolerance_len = block.short_side_length / 10
        else:
            gap_tolerance_len = block.short_side_length * 2

        if match_block_iou >= iou_threshold:
            prev_distance = (
                block.secondary_direction_start_coordinate
                - ref_block.secondary_direction_end_coordinate
                + child_match_distance_tolerance_len
            ) // 5 + ref_block.start_coordinate / 5000
            next_distance = (
                ref_block.secondary_direction_start_coordinate
                - block.secondary_direction_end_coordinate
                + child_match_distance_tolerance_len
            ) // 5 + ref_block.start_coordinate / 5000
            if (
                ref_block.secondary_direction_end_coordinate
                <= block.secondary_direction_start_coordinate
                + child_match_distance_tolerance_len
                and prev_distance < min_prev_block_distance
            ):
                min_prev_block_distance = prev_distance
                if (
                    block.secondary_direction_start_coordinate
                    - ref_block.secondary_direction_end_coordinate
                    < gap_tolerance_len
                ):
                    prev_block_index = ref_block_idx
            elif (
                ref_block.secondary_direction_start_coordinate
                > block.secondary_direction_end_coordinate
                - child_match_distance_tolerance_len
                and next_distance < min_post_block_distance
            ):
                min_post_block_distance = next_distance
                if (
                    ref_block.secondary_direction_start_coordinate
                    - block.secondary_direction_end_coordinate
                    < gap_tolerance_len
                ):
                    post_block_index = ref_block_idx

    diff_dist = abs(min_prev_block_distance - min_post_block_distance)

    # if the difference in distance is too large, only consider the nearest one
    if diff_dist * 5 > block.short_side_length:
        if min_prev_block_distance < min_post_block_distance:
            post_block_index = None
        else:
            prev_block_index = None

    return prev_block_index, post_block_index


def update_doc_title_child_blocks(
    block: LayoutParsingBlock,
    region: LayoutParsingRegion,
) -> None:
    """
    Update the child blocks of a document title block.

    The child blocks need to meet the following conditions:
        1. They must be adjacent
        2. They must have the same direction as the parent block.
        3. Their short side length should be less than 80% of the parent's short side length.
        4. Their long side length should be less than 150% of the parent's long side length.
        5. The child block must be text block.
        6. The nearest edge distance should be less than 2 times of the text line height.

    Args:
        blocks (List[LayoutParsingBlock]): overall blocks.
        block (LayoutParsingBlock): document title block.
        prev_idx (int): previous block index, None if not exist.
        post_idx (int): post block index, None if not exist.
        config (dict): configurations.

    Returns:
        None

    """
    ref_blocks = [region.block_map[idx] for idx in region.normal_text_block_idxes]
    overlap_threshold = XYCUT_SETTINGS["child_block_overlap_ratio_threshold"]
    prev_blocks, post_blocks = get_nearest_blocks(
        block, ref_blocks, overlap_threshold, block.direction
    )
    prev_block = None
    post_block = None

    if prev_blocks:
        prev_block = prev_blocks[0]
    if post_blocks:
        post_block = post_blocks[0]

    for ref_block in [prev_block, post_block]:
        if ref_block is None:
            continue
        with_seem_direction = ref_block.direction == block.direction

        short_side_length_condition = (
            ref_block.short_side_length < block.short_side_length * 0.8
        )

        long_side_length_condition = (
            ref_block.long_side_length < block.long_side_length
            or ref_block.long_side_length > 1.5 * block.long_side_length
        )

        nearest_edge_distance = get_nearest_edge_distance(block.bbox, ref_block.bbox)

        if (
            with_seem_direction
            and ref_block.label in BLOCK_LABEL_MAP["text_labels"]
            and short_side_length_condition
            and long_side_length_condition
            and ref_block.num_of_lines < 3
            and nearest_edge_distance < ref_block.text_line_height * 2
        ):
            ref_block.order_label = "doc_title_text"
            block.append_child_block(ref_block)
            region.normal_text_block_idxes.remove(ref_block.index)


def update_paragraph_title_child_blocks(
    block: LayoutParsingBlock,
    region: LayoutParsingRegion,
) -> None:
    """
    Update the child blocks of a paragraph title block.

    The child blocks need to meet the following conditions:
        1. They must be adjacent
        2. They must have the same direction as the parent block.
        3. The child block must be paragraph title block.

    Args:
        blocks (List[LayoutParsingBlock]): overall blocks.
        block (LayoutParsingBlock): document title block.
        prev_idx (int): previous block index, None if not exist.
        post_idx (int): post block index, None if not exist.
        config (dict): configurations.

    Returns:
        None

    """
    if block.order_label == "sub_paragraph_title":
        return
    ref_blocks = [
        region.block_map[idx]
        for idx in region.paragraph_title_block_idxes + region.normal_text_block_idxes
    ]
    overlap_threshold = XYCUT_SETTINGS["child_block_overlap_ratio_threshold"]
    prev_blocks, post_blocks = get_nearest_blocks(
        block, ref_blocks, overlap_threshold, block.direction
    )
    for ref_blocks in [prev_blocks, post_blocks]:
        for ref_block in ref_blocks:
            if ref_block.label not in BLOCK_LABEL_MAP["paragraph_title_labels"]:
                break
            min_text_line_height = min(
                block.text_line_height, ref_block.text_line_height
            )
            nearest_edge_distance = get_nearest_edge_distance(
                block.bbox, ref_block.bbox
            )
            with_seem_direction = ref_block.direction == block.direction
            if (
                with_seem_direction
                and nearest_edge_distance <= min_text_line_height * 1.5
            ):
                ref_block.order_label = "sub_paragraph_title"
                block.append_child_block(ref_block)
                region.paragraph_title_block_idxes.remove(ref_block.index)


def update_vision_child_blocks(
    block: LayoutParsingBlock,
    region: LayoutParsingRegion,
) -> None:
    """
    Update the child blocks of a paragraph title block.

    The child blocks need to meet the following conditions:
    - For Both:
        1. They must be adjacent
        2. The child block must be vision_title or text block.
    - For vision_title:
        1. The distance between the child block and the parent block should be less than 1/2 of the parent's height.
    - For text block:
        1. The distance between the child block and the parent block should be less than 15.
        2. The child short_side_length should be less than the parent's short side length.
        3. The child long_side_length should be less than 50% of the parent's long side length.
        4. The difference between their centers is very small.

    Args:
        blocks (List[LayoutParsingBlock]): overall blocks.
        block (LayoutParsingBlock): document title block.
        ref_block_idxes (List[int]): A list of indices of reference blocks.
        prev_idx (int): previous block index, None if not exist.
        post_idx (int): post block index, None if not exist.
        config (dict): configurations.

    Returns:
        None

    """
    ref_blocks = [
        region.block_map[idx]
        for idx in region.normal_text_block_idxes + region.vision_title_block_idxes
    ]
    overlap_threshold = XYCUT_SETTINGS["child_block_overlap_ratio_threshold"]
    has_vision_footnote = False
    has_vision_title = False
    for direction in [block.direction, block.secondary_direction]:
        prev_blocks, post_blocks = get_nearest_blocks(
            block, ref_blocks, overlap_threshold, direction
        )
        for ref_block in prev_blocks:
            if (
                ref_block.label
                not in BLOCK_LABEL_MAP["text_labels"]
                + BLOCK_LABEL_MAP["vision_title_labels"]
            ):
                break
            nearest_edge_distance = get_nearest_edge_distance(
                block.bbox, ref_block.bbox
            )
            block_center = block.get_centroid()
            ref_block_center = ref_block.get_centroid()
            if (
                ref_block.label in BLOCK_LABEL_MAP["vision_title_labels"]
                and nearest_edge_distance <= ref_block.text_line_height * 2
            ):
                has_vision_title = True
                ref_block.order_label = "vision_title"
                block.append_child_block(ref_block)
                region.vision_title_block_idxes.remove(ref_block.index)
            if ref_block.label in BLOCK_LABEL_MAP["text_labels"]:
                if (
                    not has_vision_footnote
                    and ref_block.direction == block.direction
                    and ref_block.long_side_length < block.long_side_length
                ):
                    if (
                        (
                            nearest_edge_distance <= block.text_line_height * 2
                            and ref_block.short_side_length < block.short_side_length
                            and ref_block.long_side_length
                            < 0.5 * block.long_side_length
                            and abs(block_center[0] - ref_block_center[0]) < 10
                        )
                        or (
                            block.bbox[0] - ref_block.bbox[0] < 10
                            and ref_block.num_of_lines == 1
                        )
                        or (
                            block.bbox[2] - ref_block.bbox[2] < 10
                            and ref_block.num_of_lines == 1
                        )
                    ):
                        has_vision_footnote = True
                        ref_block.order_label = "vision_footnote"
                        block.append_child_block(ref_block)
                        region.normal_text_block_idxes.remove(ref_block.index)
                break
        for ref_block in post_blocks:
            if (
                has_vision_footnote
                and ref_block.label in BLOCK_LABEL_MAP["text_labels"]
            ):
                break
            nearest_edge_distance = get_nearest_edge_distance(
                block.bbox, ref_block.bbox
            )
            block_center = block.get_centroid()
            ref_block_center = ref_block.get_centroid()
            if (
                ref_block.label in BLOCK_LABEL_MAP["vision_title_labels"]
                and nearest_edge_distance <= ref_block.text_line_height * 2
            ):
                has_vision_title = True
                ref_block.order_label = "vision_title"
                block.append_child_block(ref_block)
                region.vision_title_block_idxes.remove(ref_block.index)
            if ref_block.label in BLOCK_LABEL_MAP["text_labels"]:
                if (
                    not has_vision_footnote
                    and nearest_edge_distance <= block.text_line_height * 2
                    and ref_block.short_side_length < block.short_side_length
                    and ref_block.long_side_length < 0.5 * block.long_side_length
                    and ref_block.direction == block.direction
                    and (
                        abs(block_center[0] - ref_block_center[0]) < 10
                        or (
                            block.bbox[0] - ref_block.bbox[0] < 10
                            and ref_block.num_of_lines == 1
                        )
                        or (
                            block.bbox[2] - ref_block.bbox[2] < 10
                            and ref_block.num_of_lines == 1
                        )
                    )
                ):
                    has_vision_footnote = True
                    ref_block.order_label = "vision_footnote"
                    block.append_child_block(ref_block)
                    region.normal_text_block_idxes.remove(ref_block.index)
                break
        if has_vision_title:
            break


def calculate_discontinuous_projection(
    boxes, direction="horizontal", return_num=False
) -> List:
    """
    Calculate the discontinuous projection of boxes along the specified direction.

    Args:
        boxes (ndarray): Array of bounding boxes represented by [[x_min, y_min, x_max, y_max]].
        direction (str): direction along which to perform the projection ('horizontal' or 'vertical').

    Returns:
        list: List of tuples representing the merged intervals.
    """
    boxes = np.array(boxes)
    if direction == "horizontal":
        intervals = boxes[:, [0, 2]]
    elif direction == "vertical":
        intervals = boxes[:, [1, 3]]
    else:
        raise ValueError("direction must be 'horizontal' or 'vertical'")

    intervals = intervals[np.argsort(intervals[:, 0])]

    merged_intervals = []
    num = 1
    current_start, current_end = intervals[0]
    num_list = []

    for start, end in intervals[1:]:
        if start <= current_end:
            num += 1
            current_end = max(current_end, end)
        else:
            num_list.append(num)
            merged_intervals.append((current_start, current_end))
            num = 1
            current_start, current_end = start, end

    num_list.append(num)
    merged_intervals.append((current_start, current_end))
    if return_num:
        return merged_intervals, num_list
    return merged_intervals


def is_projection_consistent(blocks, intervals, direction="horizontal"):

    for interval in intervals:
        if direction == "horizontal":
            start_index, stop_index = 0, 2
            interval_box = [interval[0], 0, interval[1], 1]
        else:
            start_index, stop_index = 1, 3
            interval_box = [0, interval[0], 1, interval[1]]
        same_interval_bboxes = []
        for block in blocks:
            overlap_ratio = calculate_projection_overlap_ratio(
                interval_box, block.bbox, direction=direction
            )
            if overlap_ratio > 0 and block.label in BLOCK_LABEL_MAP["text_labels"]:
                same_interval_bboxes.append(block.bbox)
        start_coordinates = [bbox[start_index] for bbox in same_interval_bboxes]
        if start_coordinates:
            min_start_coordinate = min(start_coordinates)
            max_start_coordinate = max(start_coordinates)
            is_start_consistent = (
                False
                if max_start_coordinate - min_start_coordinate
                >= abs(interval[0] - interval[1]) * 0.05
                else True
            )
            stop_coordinates = [bbox[stop_index] for bbox in same_interval_bboxes]
            min_stop_coordinate = min(stop_coordinates)
            max_stop_coordinate = max(stop_coordinates)
            if (
                max_stop_coordinate - min_stop_coordinate
                >= abs(interval[0] - interval[1]) * 0.05
                and is_start_consistent
            ):
                return False
    return True


def shrink_overlapping_boxes(
    boxes, direction="horizontal", min_threshold=0, max_threshold=0.1
) -> List:
    """
    Shrink overlapping boxes along the specified direction.

    Args:
        boxes (ndarray): Array of bounding boxes represented by [[x_min, y_min, x_max, y_max]].
        direction (str): direction along which to perform the shrinking ('horizontal' or 'vertical').
        min_threshold (float): Minimum threshold for shrinking. Default is 0.
        max_threshold (float): Maximum threshold for shrinking. Default is 0.2.

    Returns:
        list: List of tuples representing the merged intervals.
    """
    current_block = boxes[0]
    for block in boxes[1:]:
        x1, y1, x2, y2 = current_block.bbox
        x1_prime, y1_prime, x2_prime, y2_prime = block.bbox
        cut_iou = calculate_projection_overlap_ratio(
            current_block.bbox, block.bbox, direction=direction
        )
        match_iou = calculate_projection_overlap_ratio(
            current_block.bbox,
            block.bbox,
            direction="horizontal" if direction == "vertical" else "vertical",
        )
        if direction == "vertical":
            if (
                (match_iou > 0 and cut_iou > min_threshold and cut_iou < max_threshold)
                or y2 == y1_prime
                or abs(y2 - y1_prime) <= 3
            ):
                overlap_y_min = max(y1, y1_prime)
                overlap_y_max = min(y2, y2_prime)
                split_y = int((overlap_y_min + overlap_y_max) / 2)
                overlap_y_min = split_y - 1
                overlap_y_max = split_y + 1
                current_block.bbox = [x1, y1, x2, overlap_y_min]
                block.bbox = [x1_prime, overlap_y_max, x2_prime, y2_prime]
        else:
            if (
                (match_iou > 0 and cut_iou > min_threshold and cut_iou < max_threshold)
                or x2 == x1_prime
                or abs(x2 - x1_prime) <= 3
            ):
                overlap_x_min = max(x1, x1_prime)
                overlap_x_max = min(x2, x2_prime)
                split_x = int((overlap_x_min + overlap_x_max) / 2)
                overlap_x_min = split_x - 1
                overlap_x_max = split_x + 1
                current_block.bbox = [x1, y1, overlap_x_min, y2]
                block.bbox = [overlap_x_max, y1_prime, x2_prime, y2_prime]
        current_block = block
    return boxes
