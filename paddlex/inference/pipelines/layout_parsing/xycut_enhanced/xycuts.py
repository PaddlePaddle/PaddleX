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
from typing import Dict, List, Tuple

import numpy as np

from ..layout_objects import LayoutBlock, LayoutRegion
from ..setting import BLOCK_LABEL_MAP, XYCUT_SETTINGS
from ..utils import calculate_overlap_ratio, calculate_projection_overlap_ratio
from .utils import (
    calculate_discontinuous_projection,
    euclidean_insert,
    find_local_minima_flat_regions,
    get_blocks_by_direction_interval,
    get_cut_blocks,
    insert_child_blocks,
    manhattan_insert,
    projection_by_bboxes,
    recursive_xy_cut,
    recursive_yx_cut,
    reference_insert,
    shrink_overlapping_boxes,
    sort_normal_blocks,
    update_doc_title_child_blocks,
    update_paragraph_title_child_blocks,
    update_region_child_blocks,
    update_vision_child_blocks,
    weighted_distance_insert,
)


def pre_process(
    region: LayoutRegion,
) -> List:
    """
    Preprocess the layout for sorting purposes.

    This function performs two main tasks:
    1. Pre-cuts the layout to ensure the document is correctly partitioned and sorted.
    2. Match the blocks with their children.

    Args:
        region: LayoutParsingRegion, the layout region to be pre-processed.

    Returns:
        List: A list of pre-cutted layout blocks list.
    """
    mask_labels = [
        "header",
        "unordered",
        "footer",
        "vision_footnote",
        "sub_paragraph_title",
        "doc_title_text",
        "vision_title",
        "sub_region",
    ]
    pre_cut_block_idxes = []
    block_map = region.block_map
    blocks: List[LayoutBlock] = list(block_map.values())
    for block in blocks:
        if block.order_label not in mask_labels:
            update_region_label(block, region)

        block_direction = block.direction
        if block_direction == "horizontal":
            tolerance_len = block.long_side_length // 5
        else:
            tolerance_len = block.short_side_length // 10

        block_center = (
            block.bbox[region.direction_start_index]
            + block.bbox[region.direction_end_index]
        ) / 2
        center_offset = abs(block_center - region.direction_center_coordinate)
        is_centered = center_offset <= tolerance_len
        if is_centered:
            pre_cut_block_idxes.append(block.index)

    pre_cut_list = []
    cut_direction = region.secondary_direction
    cut_coordinates = []
    discontinuous = []
    all_boxes = np.array(
        [block.bbox for block in blocks if block.order_label not in mask_labels]
    )
    if len(all_boxes) == 0:
        return pre_cut_list
    if pre_cut_block_idxes:
        discontinuous, num_list = calculate_discontinuous_projection(
            all_boxes, direction=cut_direction, return_num=True
        )
        for idx in pre_cut_block_idxes:
            block = block_map[idx]
            if (
                block.order_label not in mask_labels
                and block.secondary_direction == cut_direction
            ):
                if (
                    block.secondary_direction_start_coordinate,
                    block.secondary_direction_end_coordinate,
                ) in discontinuous:
                    idx = discontinuous.index(
                        (
                            block.secondary_direction_start_coordinate,
                            block.secondary_direction_end_coordinate,
                        )
                    )
                    if num_list[idx] == 1:
                        cut_coordinates.append(
                            block.secondary_direction_start_coordinate
                        )
                        cut_coordinates.append(block.secondary_direction_end_coordinate)
    secondary_check_bboxes = np.array(
        [
            block.bbox
            for block in blocks
            if block.order_label not in mask_labels + ["vision"]
        ]
    )
    if len(secondary_check_bboxes) > 0 or blocks[0].label == "region":
        secondary_discontinuous = calculate_discontinuous_projection(
            secondary_check_bboxes, direction=region.direction
        )
        if len(secondary_discontinuous) == 1 or blocks[0].label == "region":
            if not discontinuous:
                discontinuous = calculate_discontinuous_projection(
                    all_boxes, direction=cut_direction
                )
            current_interval = discontinuous[0]
            pre_cut_coordinates = [
                cood for cood in cut_coordinates if cood < current_interval[1]
            ]
            if not pre_cut_coordinates:
                pre_cut_coordinate = 0
            else:
                pre_cut_coordinate = max(pre_cut_coordinates)
            pre_cut_coordinate = max(current_interval[0], pre_cut_coordinate)
            for interval in discontinuous[1:]:
                gap_len = interval[0] - current_interval[1]
                if (
                    gap_len >= region.text_line_height * 3
                    or blocks[0].label == "region"
                ):
                    cut_coordinates.append(current_interval[1])
                elif gap_len > region.text_line_height * 1.2:
                    pre_blocks = get_blocks_by_direction_interval(
                        list(block_map.values()),
                        pre_cut_coordinate,
                        current_interval[1],
                        cut_direction,
                    )
                    post_blocks = get_blocks_by_direction_interval(
                        list(block_map.values()),
                        current_interval[1],
                        interval[1],
                        cut_direction,
                    )
                    pre_bboxes = np.array([block.bbox for block in pre_blocks])
                    post_bboxes = np.array([block.bbox for block in post_blocks])
                    projection_index = 1 if cut_direction == "horizontal" else 0
                    pre_projection = projection_by_bboxes(pre_bboxes, projection_index)
                    post_projection = projection_by_bboxes(
                        post_bboxes, projection_index
                    )
                    pre_intervals = find_local_minima_flat_regions(pre_projection)
                    post_intervals = find_local_minima_flat_regions(post_projection)
                    pre_gap_boxes = []
                    if pre_intervals is not None:
                        for start, end in pre_intervals:
                            bbox = [0] * 4
                            bbox[projection_index] = start
                            bbox[projection_index + 2] = end
                            pre_gap_boxes.append(bbox)
                    post_gap_boxes = []
                    if post_intervals is not None:
                        for start, end in post_intervals:
                            bbox = [0] * 4
                            bbox[projection_index] = start
                            bbox[projection_index + 2] = end
                            post_gap_boxes.append(bbox)
                    max_gap_boxes_num = max(len(pre_gap_boxes), len(post_gap_boxes))
                    if max_gap_boxes_num > 0:
                        discontinuous_intervals = calculate_discontinuous_projection(
                            pre_gap_boxes + post_gap_boxes, direction=region.direction
                        )
                        if len(discontinuous_intervals) != max_gap_boxes_num:
                            pre_cut_coordinate = current_interval[1]
                            cut_coordinates.append(current_interval[1])
                current_interval = interval
    cut_list = get_cut_blocks(blocks, cut_direction, cut_coordinates, mask_labels)
    pre_cut_list.extend(cut_list)
    if region.direction == "vertical":
        pre_cut_list = pre_cut_list[::-1]

    return pre_cut_list


def update_region_label(
    block: LayoutBlock,
    region: LayoutRegion,
) -> None:
    """
    Update the region label of a block based on its label and match the block with its children.

    Args:
        blocks (List[LayoutBlock]): The list of blocks to process.
        config (Dict[str, Any]): The configuration dictionary containing the necessary information.
        block_idx (int): The index of the current block being processed.

    Returns:
        None
    """
    if block.label in BLOCK_LABEL_MAP["header_labels"]:
        block.order_label = "header"
    elif block.label in BLOCK_LABEL_MAP["doc_title_labels"]:
        block.order_label = "doc_title"
    elif (
        block.label in BLOCK_LABEL_MAP["paragraph_title_labels"]
        and block.order_label is None
    ):
        block.order_label = "paragraph_title"
    elif block.label in BLOCK_LABEL_MAP["vision_labels"]:
        block.order_label = "vision"
        block.num_of_lines = 1
        block.update_direction(region.direction)
    elif block.label in BLOCK_LABEL_MAP["footer_labels"]:
        block.order_label = "footer"
    elif block.label in BLOCK_LABEL_MAP["unordered_labels"]:
        block.order_label = "unordered"
    elif block.label == "region":
        block.order_label = "region"
    else:
        block.order_label = "normal_text"

    # only vision and doc title block can have child block
    if block.order_label not in ["vision", "doc_title", "paragraph_title", "region"]:
        return

    # match doc title text block
    if block.order_label == "doc_title":
        update_doc_title_child_blocks(block, region)
    # match sub title block
    elif block.order_label == "paragraph_title":
        update_paragraph_title_child_blocks(block, region)
    # match vision title block and vision footnote block
    elif block.order_label == "vision":
        update_vision_child_blocks(block, region)
    elif block.order_label == "region":
        update_region_child_blocks(block, region)


def get_layout_structure(
    blocks: List[LayoutBlock],
    region_direction: str,
    region_secondary_direction: str,
) -> Tuple[List[Dict[str, any]], bool]:
    """
    Determine the layout cross column of blocks.

    Args:
        blocks (List[Dict[str, any]]): List of block dictionaries containing 'label' and 'block_bbox'.

    Returns:
        Tuple[List[Dict[str, any]], bool]: Updated list of blocks with layout information and a boolean
        indicating if the cross layout area is greater than the single layout area.
    """
    blocks.sort(
        key=lambda x: (x.bbox[0], x.width),
    )

    mask_labels = ["doc_title", "cross_layout", "cross_reference"]
    for block_idx, block in enumerate(blocks):
        if block.order_label in mask_labels:
            continue

        for ref_idx, ref_block in enumerate(blocks):
            if block_idx == ref_idx or ref_block.order_label in mask_labels:
                continue

            bbox_iou = calculate_overlap_ratio(block.bbox, ref_block.bbox)
            if bbox_iou:
                if ref_block.order_label == "vision":
                    ref_block.order_label = "cross_layout"
                    break
                if bbox_iou > 0.1 and block.area < ref_block.area:
                    block.order_label = "cross_layout"
                    break

            match_projection_iou = calculate_projection_overlap_ratio(
                block.bbox,
                ref_block.bbox,
                region_direction,
            )
            if match_projection_iou > 0:
                for second_ref_idx, second_ref_block in enumerate(blocks):
                    if (
                        second_ref_idx in [block_idx, ref_idx]
                        or second_ref_block.order_label in mask_labels
                    ):
                        continue

                    bbox_iou = calculate_overlap_ratio(
                        block.bbox, second_ref_block.bbox
                    )
                    if bbox_iou > 0.1:
                        if second_ref_block.order_label == "vision":
                            second_ref_block.order_label = "cross_layout"
                            break
                        if (
                            block.order_label == "vision"
                            or block.area < second_ref_block.area
                        ):
                            block.order_label = "cross_layout"
                            break

                    second_match_projection_iou = calculate_projection_overlap_ratio(
                        block.bbox,
                        second_ref_block.bbox,
                        region_direction,
                    )
                    ref_match_projection_iou = calculate_projection_overlap_ratio(
                        ref_block.bbox,
                        second_ref_block.bbox,
                        region_direction,
                    )
                    secondary_direction_ref_match_projection_overlap_ratio = (
                        calculate_projection_overlap_ratio(
                            ref_block.bbox,
                            second_ref_block.bbox,
                            region_secondary_direction,
                        )
                    )
                    if (
                        second_match_projection_iou > 0
                        and ref_match_projection_iou == 0
                        and secondary_direction_ref_match_projection_overlap_ratio > 0
                    ):
                        if block.order_label in ["vision", "region"] or (
                            ref_block.order_label == "normal_text"
                            and second_ref_block.order_label == "normal_text"
                            and ref_block.long_side_length
                            > ref_block.text_line_height
                            * XYCUT_SETTINGS.get(
                                "cross_layout_ref_text_block_words_num_threshold", 8
                            )
                            and second_ref_block.long_side_length
                            > second_ref_block.text_line_height
                            * XYCUT_SETTINGS.get(
                                "cross_layout_ref_text_block_words_num_threshold", 8
                            )
                        ):
                            block.order_label = (
                                "cross_reference"
                                if block.label == "reference"
                                else "cross_layout"
                            )


def sort_by_xycut(
    block_bboxes: List,
    direction: str = "vertical",
    min_gap: int = 1,
) -> List[int]:
    """
    Sort bounding boxes using recursive XY cut method based on the specified direction.

    Args:
        block_bboxes (Union[np.ndarray, List[List[int]]]): An array or list of bounding boxes,
                                                           where each box is represented as
                                                           [x_min, y_min, x_max, y_max].
        direction (int): direction for the initial cut. Use 1 for Y-axis first and 0 for X-axis first.
                         Defaults to 0.
        min_gap (int): Minimum gap width to consider a separation between segments. Defaults to 1.

    Returns:
        List[int]: A list of indices representing the order of sorted bounding boxes.
    """
    block_bboxes = np.asarray(block_bboxes).astype(int)
    res = []
    if direction == "vertical":
        recursive_yx_cut(
            block_bboxes,
            np.arange(len(block_bboxes)).tolist(),
            res,
            min_gap,
        )
    else:
        recursive_xy_cut(
            block_bboxes,
            np.arange(len(block_bboxes)).tolist(),
            res,
            min_gap,
        )
    return res


def match_unsorted_blocks(
    sorted_blocks: List[LayoutBlock],
    unsorted_blocks: List[LayoutBlock],
    region: LayoutRegion,
) -> List[LayoutBlock]:
    """
    Match special blocks with the sorted blocks based on their region labels.
    Args:
        sorted_blocks (List[LayoutBlock]): Sorted blocks to be matched.
        unsorted_blocks (List[LayoutBlock]): Unsorted blocks to be matched.
        config (Dict): Configuration dictionary containing various parameters.
        median_width (int): Median width value used for calculations.

    Returns:
        List[LayoutBlock]: The updated sorted blocks after matching special blocks.
    """
    distance_type_map = {
        "cross_layout": weighted_distance_insert,
        "paragraph_title": weighted_distance_insert,
        "doc_title": weighted_distance_insert,
        "vision_title": weighted_distance_insert,
        "vision": weighted_distance_insert,
        "cross_reference": reference_insert,
        "unordered": manhattan_insert,
        "other": manhattan_insert,
        "region": euclidean_insert,
    }

    unsorted_blocks = sort_normal_blocks(
        unsorted_blocks,
        region.text_line_height,
        region.text_line_width,
        region.direction,
    )
    for idx, block in enumerate(unsorted_blocks):
        order_label = block.order_label if block.label != "region" else "region"
        if idx == 0 and order_label == "doc_title":
            sorted_blocks.insert(0, block)
            continue
        sorted_blocks = distance_type_map[order_label](
            block=block, sorted_blocks=sorted_blocks, region=region
        )
    return sorted_blocks


def xycut_enhanced(
    region: LayoutRegion,
) -> LayoutRegion:
    """
    xycut_enhance function performs the following steps:
        1. Preprocess the input blocks by extracting headers, footers, and pre-cut blocks.
        2. Mask blocks that are crossing different blocks.
        3. Perform xycut_enhanced algorithm on the remaining blocks.
        4. Match unsorted blocks with the sorted blocks based on their order labels.
        5. Update child blocks of the sorted blocks based on their parent blocks.
        6. Return the ordered result list.

    Args:
        blocks (List[LayoutBlock]): Input blocks to be processed.

    Returns:
        List[LayoutBlock]: Ordered result list after processing.
    """
    if len(region.block_map) == 0:
        return []

    pre_cut_list: List[List[LayoutBlock]] = pre_process(region)
    final_order_res_list: List[LayoutBlock] = []

    header_blocks: List[LayoutBlock] = [
        region.block_map[idx] for idx in region.header_block_idxes
    ]
    unordered_blocks: List[LayoutBlock] = [
        region.block_map[idx] for idx in region.unordered_block_idxes
    ]
    footer_blocks: List[LayoutBlock] = [
        region.block_map[idx] for idx in region.footer_block_idxes
    ]

    header_blocks: List[LayoutBlock] = sort_normal_blocks(
        header_blocks, region.text_line_height, region.text_line_width, region.direction
    )
    footer_blocks: List[LayoutBlock] = sort_normal_blocks(
        footer_blocks, region.text_line_height, region.text_line_width, region.direction
    )
    unordered_blocks: List[LayoutBlock] = sort_normal_blocks(
        unordered_blocks,
        region.text_line_height,
        region.text_line_width,
        region.direction,
    )
    final_order_res_list.extend(header_blocks)

    unsorted_blocks: List[LayoutBlock] = []
    sorted_blocks_by_pre_cuts: List[LayoutBlock] = []
    for pre_cut_blocks in pre_cut_list:
        sorted_blocks: List[LayoutBlock] = []
        doc_title_blocks: List[LayoutBlock] = []
        xy_cut_blocks: List[LayoutBlock] = []

        if pre_cut_blocks and pre_cut_blocks[0].label == "region":
            block_bboxes = np.array([block.bbox for block in pre_cut_blocks])
            discontinuous = calculate_discontinuous_projection(
                block_bboxes, direction=region.direction
            )
            if len(discontinuous) == 1:
                get_layout_structure(
                    pre_cut_blocks, region.direction, region.secondary_direction
                )
        else:
            get_layout_structure(
                pre_cut_blocks, region.direction, region.secondary_direction
            )

        # Get xy cut blocks and add other blocks in special_block_map
        for block in pre_cut_blocks:
            if block.order_label not in [
                "cross_layout",
                "cross_reference",
                "doc_title",
                "unordered",
            ]:
                xy_cut_blocks.append(block)
            elif block.label == "doc_title":
                doc_title_blocks.append(block)
            else:
                unsorted_blocks.append(block)

        if len(xy_cut_blocks) > 0:
            block_bboxes = np.array([block.bbox for block in xy_cut_blocks])
            block_text_lines = [block.num_of_lines for block in xy_cut_blocks]
            discontinuous = calculate_discontinuous_projection(
                block_bboxes, direction=region.direction
            )
            blocks_to_sort = deepcopy(xy_cut_blocks)
            if region.direction == "vertical":
                for block in blocks_to_sort:
                    block.bbox = np.array(
                        [-block.bbox[0], block.bbox[1], -block.bbox[2], block.bbox[3]]
                    )
            if len(discontinuous) == 1 or max(block_text_lines) == 1:
                blocks_to_sort.sort(
                    key=lambda x: (
                        x.bbox[region.secondary_direction_start_index]
                        // (region.text_line_height // 2),
                        x.bbox[region.direction_start_index],
                    )
                )
                blocks_to_sort = shrink_overlapping_boxes(
                    blocks_to_sort, region.secondary_direction
                )
                block_bboxes = np.array([block.bbox for block in blocks_to_sort])
                sorted_indexes = sort_by_xycut(
                    block_bboxes, direction=region.secondary_direction, min_gap=1
                )
            else:
                blocks_to_sort.sort(
                    key=lambda x: (
                        x.bbox[region.direction_start_index]
                        // (region.text_line_width // 2),
                        x.bbox[region.secondary_direction_start_index],
                    )
                )
                blocks_to_sort = shrink_overlapping_boxes(
                    blocks_to_sort, region.secondary_direction
                )
                block_bboxes = np.array([block.bbox for block in blocks_to_sort])
                sorted_indexes = sort_by_xycut(
                    block_bboxes, direction=region.direction, min_gap=1
                )

            sorted_blocks = [
                region.block_map[blocks_to_sort[i].index] for i in sorted_indexes
            ]
        sorted_blocks = match_unsorted_blocks(
            sorted_blocks,
            doc_title_blocks,
            region=region,
        )

        if unsorted_blocks and unsorted_blocks[0].label == "region":
            sorted_blocks = match_unsorted_blocks(
                sorted_blocks,
                unsorted_blocks,
                region=region,
            )
            unsorted_blocks = []
        sorted_blocks_by_pre_cuts.extend(sorted_blocks)

    final_sorted_blocks = match_unsorted_blocks(
        sorted_blocks_by_pre_cuts,
        unsorted_blocks,
        region=region,
    )

    final_order_res_list.extend(final_sorted_blocks)
    final_order_res_list.extend(footer_blocks)
    final_order_res_list.extend(unordered_blocks)

    for block_idx, block in enumerate(final_order_res_list):
        final_order_res_list = insert_child_blocks(
            block, block_idx, final_order_res_list
        )
        block = final_order_res_list[block_idx]
    return final_order_res_list
