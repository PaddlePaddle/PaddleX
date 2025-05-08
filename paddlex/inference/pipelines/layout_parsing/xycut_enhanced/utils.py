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

from typing import Dict, List, Tuple

import numpy as np

from ..result_v2 import LayoutParsingBlock
from ..utils import calculate_projection_overlap_ratio


def get_nearest_edge_distance(
    bbox1: List[int],
    bbox2: List[int],
    weight: List[float] = [1.0, 1.0, 1.0, 1.0],
) -> Tuple[float]:
    """
    Calculate the nearest edge distance between two bounding boxes, considering orientational weights.

    Args:
        bbox1 (list): The bounding box coordinates [x1, y1, x2, y2] of the input object.
        bbox2 (list): The bounding box coordinates [x1', y1', x2', y2'] of the object to match against.
        weight (list, optional): orientational weights for the edge distances [left, right, up, down]. Defaults to [1, 1, 1, 1].

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


def _projection_by_bboxes(boxes: np.ndarray, axis: int) -> np.ndarray:
    """
    Generate a 1D projection histogram from bounding boxes along a specified axis.

    Args:
        boxes: A (N, 4) array of bounding boxes defined by [x_min, y_min, x_max, y_max].
        axis: Axis for projection; 0 for horizontal (x-axis), 1 for vertical (y-axis).

    Returns:
        A 1D numpy array representing the projection histogram based on bounding box intervals.
    """
    assert axis in [0, 1]
    max_length = np.max(boxes[:, axis::2])
    projection = np.zeros(max_length, dtype=int)

    # Increment projection histogram over the interval defined by each bounding box
    for start, end in boxes[:, axis::2]:
        projection[start:end] += 1

    return projection


def _split_projection_profile(arr_values: np.ndarray, min_value: float, min_gap: float):
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
    y_projection = _projection_by_bboxes(boxes=y_sorted_boxes, axis=1)
    y_intervals = _split_projection_profile(y_projection, 0, 1)

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
        x_projection = _projection_by_bboxes(boxes=x_sorted_boxes_chunk, axis=0)
        x_intervals = _split_projection_profile(x_projection, 0, min_gap)

        if not x_intervals:
            continue

        # If X-axis cannot be further segmented, add current indices to results
        if len(x_intervals[0]) == 1:
            res.extend(x_sorted_indices_chunk)
            continue

        # Recursively process each segment defined by X-axis projection
        for x_start, x_end in zip(*x_intervals):
            x_interval_indices = (x_start <= x_sorted_boxes_chunk[:, 0]) & (
                x_sorted_boxes_chunk[:, 0] < x_end
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
    x_projection = _projection_by_bboxes(boxes=x_sorted_boxes, axis=0)
    x_intervals = _split_projection_profile(x_projection, 0, 1)

    if not x_intervals:
        return

    # Process each segment defined by X-axis projection
    for x_start, x_end in zip(*x_intervals):
        # Select boxes within the current x interval
        x_interval_indices = (x_start <= x_sorted_boxes[:, 0]) & (
            x_sorted_boxes[:, 0] < x_end
        )
        x_boxes_chunk = x_sorted_boxes[x_interval_indices]
        x_indices_chunk = x_sorted_indices[x_interval_indices]

        # Sort selected boxes by y_min to prepare for Y-axis projection
        y_sorted_indices = x_boxes_chunk[:, 1].argsort()
        y_sorted_boxes_chunk = x_boxes_chunk[y_sorted_indices]
        y_sorted_indices_chunk = x_indices_chunk[y_sorted_indices]

        # Perform Y-axis projection
        y_projection = _projection_by_bboxes(boxes=y_sorted_boxes_chunk, axis=1)
        y_intervals = _split_projection_profile(y_projection, 0, min_gap)

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
    config: Dict,
    median_width: float = 0.0,
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
    config: Dict,
    median_width: float = 0.0,
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
    config: Dict,
    median_width: float = 0.0,
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
    doc_title_labels = config.get("doc_title_labels", [])
    paragraph_title_labels = config.get("paragraph_title_labels", [])
    vision_labels = config.get("vision_labels", [])
    xy_cut_block_labels = config.get("xy_cut_block_labels", [])
    tolerance_len = config.get("tolerance_len", 2)
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
        weight = _get_weights(block.order_label, block.orientation)
        edge_distance = get_nearest_edge_distance(block.bbox, sorted_block.bbox, weight)

        if block.label in doc_title_labels:
            disperse = max(1, median_width)
            tolerance_len = max(tolerance_len, disperse)
        if block.label == "abstract":
            tolerance_len *= 2
            edge_distance = max(0.1, edge_distance) * 10

        # Calculate up edge distances
        up_edge_distance = y1_prime
        left_edge_distance = x1_prime
        if (
            block.label in xy_cut_block_labels
            or block.label in doc_title_labels
            or block.label in paragraph_title_labels
            or block.label in vision_labels
        ) and y1 > y2_prime:
            up_edge_distance = -y2_prime
            left_edge_distance = -x2_prime

        if abs(min_up_edge_distance - up_edge_distance) <= tolerance_len:
            up_edge_distance = min_up_edge_distance

        # Calculate weighted distance
        weighted_distance = (
            +edge_distance * config.get("edge_weight", 10**4)
            + up_edge_distance * config.get("up_edge_weight", 1)
            + left_edge_distance * config.get("left_edge_weight", 0.0001)
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
        sub_blocks = sort_child_blocks(sub_blocks, block.orientation)
        sorted_blocks[block_idx] = sub_blocks[0]
        for block in sub_blocks[1:]:
            block_idx += 1
            sorted_blocks.insert(block_idx, block)
    return sorted_blocks


def sort_child_blocks(blocks, orientation="horizontal") -> List[LayoutParsingBlock]:
    """
    Sort child blocks based on their bounding box coordinates.

    Args:
        blocks: A list of LayoutParsingBlock objects representing the child blocks.
        orientation: Orientation of the blocks ('horizontal' or 'vertical'). Default is 'horizontal'.
    Returns:
        sorted_blocks: A sorted list of LayoutParsingBlock objects.
    """
    if orientation == "horizontal":
        # from top to bottom
        blocks.sort(
            key=lambda x: (
                x.bbox[1],  # y_min
                x.bbox[0],  # x_min
                x.bbox[1] ** 2 + x.bbox[0] ** 2,  # distance with (0,0)
            ),
            reverse=False,
        )
    else:
        # from right to left
        blocks.sort(
            key=lambda x: (
                x.bbox[0],  # x_min
                x.bbox[1],  # y_min
                x.bbox[1] ** 2 + x.bbox[0] ** 2,  # distance with (0,0)
            ),
            reverse=True,
        )
    return blocks


def _get_weights(label, dircetion="horizontal"):
    """Define weights based on the label and orientation."""
    if label == "doc_title":
        return (
            [1, 0.1, 0.1, 1] if dircetion == "horizontal" else [0.2, 0.1, 1, 1]
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


def sort_blocks(blocks, median_width=None, reverse=False):
    """
    Sort blocks based on their y_min, x_min and distance with (0,0).

    Args:
        blocks (list): list of blocks to be sorted.
        median_width (int): the median width of the text blocks.
        reverse (bool, optional): whether to sort in descending order. Default is False.

    Returns:
        list: a list of sorted blocks.
    """
    if median_width is None:
        median_width = 1
    blocks.sort(
        key=lambda x: (
            x.bbox[1] // 10,  # y_min
            x.bbox[0] // median_width,  # x_min
            x.bbox[1] ** 2 + x.bbox[0] ** 2,  # distance with (0,0)
        ),
        reverse=reverse,
    )
    return blocks


def get_cut_blocks(
    blocks, cut_orientation, cut_coordinates, overall_region_box, mask_labels=[]
):
    """
    Cut blocks based on the given cut orientation and coordinates.

    Args:
        blocks (list): list of blocks to be cut.
        cut_orientation (str): cut orientation, either "horizontal" or "vertical".
        cut_coordinates (list): list of cut coordinates.
        overall_region_box (list): the overall region box that contains all blocks.

    Returns:
        list: a list of tuples containing the cutted blocks and their corresponding mean width。
    """
    cuted_list = []
    # filter out mask blocks,including header, footer, unordered and child_blocks

    # 0: horizontal, 1: vertical
    cut_aixis = 0 if cut_orientation == "horizontal" else 1
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
        block_bboxes, orientation="vertical"
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


def get_adjacent_blocks_by_orientation(
    blocks: List[LayoutParsingBlock],
    block_idx: int,
    ref_block_idxes: List[int],
    iou_threshold,
) -> List:
    """
    Get the adjacent blocks with the same orientation as the current block.
    Args:
        block (LayoutParsingBlock): The current block.
        blocks (List[LayoutParsingBlock]): A list of all blocks.
        ref_block_idxes (List[int]): A list of indices of reference blocks.
        iou_threshold (float): The IOU threshold to determine if two blocks are considered adjacent.
    Returns:
        Int: The index of the previous block with same orientation.
        Int: The index of the following block with same orientation.
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

    # find the nearest text block with same orientation to the current block
    for ref_block_idx in ref_block_idxes:
        ref_block = blocks[ref_block_idx]
        ref_block_orientation = ref_block.orientation
        if ref_block.order_label in child_labels:
            continue
        match_block_iou = calculate_projection_overlap_ratio(
            block.bbox,
            ref_block.bbox,
            ref_block_orientation,
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
                block.secondary_orientation_start_coordinate
                - ref_block.secondary_orientation_end_coordinate
                + child_match_distance_tolerance_len
            ) // 5 + ref_block.start_coordinate / 5000
            next_distance = (
                ref_block.secondary_orientation_start_coordinate
                - block.secondary_orientation_end_coordinate
                + child_match_distance_tolerance_len
            ) // 5 + ref_block.start_coordinate / 5000
            if (
                ref_block.secondary_orientation_end_coordinate
                <= block.secondary_orientation_start_coordinate
                + child_match_distance_tolerance_len
                and prev_distance < min_prev_block_distance
            ):
                min_prev_block_distance = prev_distance
                if (
                    block.secondary_orientation_start_coordinate
                    - ref_block.secondary_orientation_end_coordinate
                    < gap_tolerance_len
                ):
                    prev_block_index = ref_block_idx
            elif (
                ref_block.secondary_orientation_start_coordinate
                > block.secondary_orientation_end_coordinate
                - child_match_distance_tolerance_len
                and next_distance < min_post_block_distance
            ):
                min_post_block_distance = next_distance
                if (
                    ref_block.secondary_orientation_start_coordinate
                    - block.secondary_orientation_end_coordinate
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
    blocks: List[LayoutParsingBlock],
    block: LayoutParsingBlock,
    prev_idx: int,
    post_idx: int,
    config: dict,
) -> None:
    """
    Update the child blocks of a document title block.

    The child blocks need to meet the following conditions:
        1. They must be adjacent
        2. They must have the same orientation as the parent block.
        3. Their short side length should be less than 80% of the parent's short side length.
        4. Their long side length should be less than 150% of the parent's long side length.
        5. The child block must be text block.

    Args:
        blocks (List[LayoutParsingBlock]): overall blocks.
        block (LayoutParsingBlock): document title block.
        prev_idx (int): previous block index, None if not exist.
        post_idx (int): post block index, None if not exist.
        config (dict): configurations.

    Returns:
        None

    """
    for idx in [prev_idx, post_idx]:
        if idx is None:
            continue
        ref_block = blocks[idx]
        with_seem_orientation = ref_block.orientation == block.orientation

        short_side_length_condition = (
            ref_block.short_side_length < block.short_side_length * 0.8
        )

        long_side_length_condition = (
            ref_block.long_side_length < block.long_side_length
            or ref_block.long_side_length > 1.5 * block.long_side_length
        )

        if (
            with_seem_orientation
            and short_side_length_condition
            and long_side_length_condition
            and ref_block.num_of_lines < 3
        ):
            ref_block.order_label = "doc_title_text"
            block.append_child_block(ref_block)
            config["text_block_idxes"].remove(idx)


def update_paragraph_title_child_blocks(
    blocks: List[LayoutParsingBlock],
    block: LayoutParsingBlock,
    prev_idx: int,
    post_idx: int,
    config: dict,
) -> None:
    """
    Update the child blocks of a paragraph title block.

    The child blocks need to meet the following conditions:
        1. They must be adjacent
        2. They must have the same orientation as the parent block.
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
    paragraph_title_labels = config.get("paragraph_title_labels", [])
    for idx in [prev_idx, post_idx]:
        if idx is None:
            continue
        ref_block = blocks[idx]
        min_height = min(block.height, ref_block.height)
        nearest_edge_distance = get_nearest_edge_distance(block.bbox, ref_block.bbox)
        with_seem_orientation = ref_block.orientation == block.orientation
        if (
            with_seem_orientation
            and ref_block.label in paragraph_title_labels
            and nearest_edge_distance <= min_height * 2
        ):
            ref_block.order_label = "sub_paragraph_title"
            block.append_child_block(ref_block)
            config["paragraph_title_block_idxes"].remove(idx)


def update_vision_child_blocks(
    blocks: List[LayoutParsingBlock],
    block: LayoutParsingBlock,
    ref_block_idxes: List[int],
    prev_idx: int,
    post_idx: int,
    config: dict,
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
    vision_title_labels = config.get("vision_title_labels", [])
    text_labels = config.get("text_labels", [])
    for idx in [prev_idx, post_idx]:
        if idx is None:
            continue
        ref_block = blocks[idx]
        nearest_edge_distance = get_nearest_edge_distance(block.bbox, ref_block.bbox)
        block_center = block.get_centroid()
        ref_block_center = ref_block.get_centroid()
        if ref_block.label in vision_title_labels and nearest_edge_distance <= min(
            block.height * 0.5, ref_block.height * 2
        ):
            ref_block.order_label = "vision_title"
            block.append_child_block(ref_block)
            config["vision_title_block_idxes"].remove(idx)
        elif (
            nearest_edge_distance <= 15
            and ref_block.short_side_length < block.short_side_length
            and ref_block.long_side_length < 0.5 * block.long_side_length
            and ref_block.orientation == block.orientation
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
            has_vision_footnote = False
            if len(block.child_blocks) > 0:
                for child_block in block.child_blocks:
                    if child_block.label in text_labels:
                        has_vision_footnote = True
            if not has_vision_footnote:
                ref_block.order_label = "vision_footnote"
                block.append_child_block(ref_block)
                config["text_block_idxes"].remove(idx)


def calculate_discontinuous_projection(
    boxes, orientation="horizontal", return_num=False
) -> List:
    """
    Calculate the discontinuous projection of boxes along the specified orientation.

    Args:
        boxes (ndarray): Array of bounding boxes represented by [[x_min, y_min, x_max, y_max]].
        orientation (str): orientation along which to perform the projection ('horizontal' or 'vertical').

    Returns:
        list: List of tuples representing the merged intervals.
    """
    boxes = np.array(boxes)
    if orientation == "horizontal":
        intervals = boxes[:, [0, 2]]
    elif orientation == "vertical":
        intervals = boxes[:, [1, 3]]
    else:
        raise ValueError("orientation must be 'horizontal' or 'vertical'")

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


def shrink_overlapping_boxes(
    boxes, orientation="horizontal", min_threshold=0, max_threshold=0.1
) -> List:
    """
    Shrink overlapping boxes along the specified orientation.

    Args:
        boxes (ndarray): Array of bounding boxes represented by [[x_min, y_min, x_max, y_max]].
        orientation (str): orientation along which to perform the shrinking ('horizontal' or 'vertical').
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
            current_block.bbox, block.bbox, orientation=orientation
        )
        match_iou = calculate_projection_overlap_ratio(
            current_block.bbox,
            block.bbox,
            orientation="horizontal" if orientation == "vertical" else "vertical",
        )
        if orientation == "vertical":
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
