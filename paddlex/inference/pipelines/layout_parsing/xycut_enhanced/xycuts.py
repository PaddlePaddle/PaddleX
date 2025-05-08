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

from typing import Any, Dict, List, Tuple

import numpy as np

from ..result_v2 import LayoutParsingBlock
from ..utils import calculate_overlap_ratio, calculate_projection_overlap_ratio
from .utils import (
    calculate_discontinuous_projection,
    get_adjacent_blocks_by_orientation,
    get_cut_blocks,
    get_nearest_edge_distance,
    insert_child_blocks,
    manhattan_insert,
    recursive_xy_cut,
    recursive_yx_cut,
    reference_insert,
    shrink_overlapping_boxes,
    sort_blocks,
    update_doc_title_child_blocks,
    update_paragraph_title_child_blocks,
    update_vision_child_blocks,
    weighted_distance_insert,
)


def pre_process(
    blocks: List[LayoutParsingBlock],
    config: Dict,
) -> List:
    """
    Preprocess the layout for sorting purposes.

    This function performs two main tasks:
    1. Pre-cuts the layout to ensure the document is correctly partitioned and sorted.
    2. Match the blocks with their children.

    Args:
        blocks (List[LayoutParsingBlock]): A list of LayoutParsingBlock objects representing the layout.
        config (Dict): Configuration parameters that include settings for pre-cutting and sorting.

    Returns:
        List: A list of pre-cutted layout blocks list.
    """
    region_bbox = config.get("region_bbox", None)
    region_x_center = (region_bbox[0] + region_bbox[2]) / 2
    region_y_center = (region_bbox[1] + region_bbox[3]) / 2

    header_block_idxes = config.get("header_block_idxes", [])
    header_blocks = []
    for idx in header_block_idxes:
        blocks[idx].order_label = "header"
        header_blocks.append(blocks[idx])

    unordered_block_idxes = config.get("unordered_block_idxes", [])
    unordered_blocks = []
    for idx in unordered_block_idxes:
        blocks[idx].order_label = "unordered"
        unordered_blocks.append(blocks[idx])

    footer_block_idxes = config.get("footer_block_idxes", [])
    footer_blocks = []
    for idx in footer_block_idxes:
        blocks[idx].order_label = "footer"
        footer_blocks.append(blocks[idx])

    mask_labels = ["header", "unordered", "footer"]
    child_labels = [
        "vision_footnote",
        "sub_paragraph_title",
        "doc_title_text",
        "vision_title",
    ]
    pre_cut_block_idxes = []
    for block_idx, block in enumerate(blocks):
        if block.label in mask_labels:
            continue

        if block.order_label not in child_labels:
            update_region_label(blocks, config, block_idx)

        block_orientation = block.orientation
        if block_orientation == "horizontal":
            region_bbox_center = region_x_center
            tolerance_len = block.long_side_length // 5
        else:
            region_bbox_center = region_y_center
            tolerance_len = block.short_side_length // 10

        block_center = (block.start_coordinate + block.end_coordinate) / 2
        center_offset = abs(block_center - region_bbox_center)
        is_centered = center_offset <= tolerance_len

        if is_centered:
            pre_cut_block_idxes.append(block_idx)

    pre_cut_list = []
    cut_orientation = "vertical"
    cut_coordinates = []
    discontinuous = []
    mask_labels = child_labels + mask_labels
    all_boxes = np.array(
        [block.bbox for block in blocks if block.order_label not in mask_labels]
    )
    if len(all_boxes) == 0:
        return header_blocks, pre_cut_list, footer_blocks, unordered_blocks
    if pre_cut_block_idxes:
        horizontal_cut_num = 0
        for block_idx in pre_cut_block_idxes:
            block = blocks[block_idx]
            horizontal_cut_num += (
                1 if block.secondary_orientation == "horizontal" else 0
            )
        cut_orientation = (
            "horizontal"
            if horizontal_cut_num > len(pre_cut_block_idxes) * 0.5
            else "vertical"
        )
        discontinuous, num_list = calculate_discontinuous_projection(
            all_boxes, orientation=cut_orientation, return_num=True
        )
        for idx in pre_cut_block_idxes:
            block = blocks[idx]
            if (
                block.order_label not in mask_labels
                and block.secondary_orientation == cut_orientation
            ):
                if (
                    block.secondary_orientation_start_coordinate,
                    block.secondary_orientation_end_coordinate,
                ) in discontinuous:
                    idx = discontinuous.index(
                        (
                            block.secondary_orientation_start_coordinate,
                            block.secondary_orientation_end_coordinate,
                        )
                    )
                    if num_list[idx] == 1:
                        cut_coordinates.append(
                            block.secondary_orientation_start_coordinate
                        )
                        cut_coordinates.append(
                            block.secondary_orientation_end_coordinate
                        )
    if not discontinuous:
        discontinuous = calculate_discontinuous_projection(
            all_boxes, orientation=cut_orientation
        )
    current_interval = discontinuous[0]
    for interval in discontinuous[1:]:
        gap_len = interval[0] - current_interval[1]
        if gap_len >= 60:
            cut_coordinates.append(current_interval[1])
        elif gap_len > 40:
            x1, _, x2, __ = region_bbox
            y1 = current_interval[1]
            y2 = interval[0]
            bbox = [x1, y1, x2, y2]
            ref_interval = interval[0] - current_interval[1]
            ref_bboxes = []
            for block in blocks:
                if get_nearest_edge_distance(bbox, block.bbox) < ref_interval * 2:
                    ref_bboxes.append(block.bbox)
            discontinuous = calculate_discontinuous_projection(
                ref_bboxes, orientation="horizontal"
            )
            if len(discontinuous) != 2:
                cut_coordinates.append(current_interval[1])
        current_interval = interval
    cut_list = get_cut_blocks(
        blocks, cut_orientation, cut_coordinates, region_bbox, mask_labels
    )
    pre_cut_list.extend(cut_list)

    return header_blocks, pre_cut_list, footer_blocks, unordered_blocks


def update_region_label(
    blocks: List[LayoutParsingBlock], config: Dict[str, Any], block_idx: int
) -> None:
    """
    Update the region label of a block based on its label and match the block with its children.

    Args:
        blocks (List[LayoutParsingBlock]): The list of blocks to process.
        config (Dict[str, Any]): The configuration dictionary containing the necessary information.
        block_idx (int): The index of the current block being processed.

    Returns:
        None
    """

    # special title block labels
    doc_title_labels = config.get("doc_title_labels", [])
    paragraph_title_labels = config.get("paragraph_title_labels", [])
    vision_labels = config.get("vision_labels", [])

    block = blocks[block_idx]
    if block.label in doc_title_labels:
        block.order_label = "doc_title"
    # Force the orientation of vision type to be horizontal
    if block.label in vision_labels:
        block.order_label = "vision"
        block.num_of_lines = 1
        block.update_orientation_info()
    # some paragraph title block may be labeled as sub_title, so we need to check if block.order_label is "other"(default).
    if block.label in paragraph_title_labels and block.order_label == "other":
        block.order_label = "paragraph_title"

    # only vision and doc title block can have child block
    if block.order_label not in ["vision", "doc_title", "paragraph_title"]:
        return

    iou_threshold = config.get("child_block_match_iou_threshold", 0.1)
    # match doc title text block
    if block.order_label == "doc_title":
        text_block_idxes = config.get("text_block_idxes", [])
        prev_idx, post_idx = get_adjacent_blocks_by_orientation(
            blocks, block_idx, text_block_idxes, iou_threshold
        )
        update_doc_title_child_blocks(blocks, block, prev_idx, post_idx, config)
    # match sub title block
    elif block.order_label == "paragraph_title":
        iou_threshold = config.get("sub_title_match_iou_threshold", 0.1)
        paragraph_title_block_idxes = config.get("paragraph_title_block_idxes", [])
        text_block_idxes = config.get("text_block_idxes", [])
        megred_block_idxes = text_block_idxes + paragraph_title_block_idxes
        prev_idx, post_idx = get_adjacent_blocks_by_orientation(
            blocks, block_idx, megred_block_idxes, iou_threshold
        )
        update_paragraph_title_child_blocks(blocks, block, prev_idx, post_idx, config)
    # match vision title block
    elif block.order_label == "vision":
        # for matching vision title block
        vision_title_block_idxes = config.get("vision_title_block_idxes", [])
        # for matching vision footnote block
        text_block_idxes = config.get("text_block_idxes", [])
        megred_block_idxes = text_block_idxes + vision_title_block_idxes
        # Some vision title block may be matched with multiple vision title block, so we need to try multiple times
        for i in range(3):
            prev_idx, post_idx = get_adjacent_blocks_by_orientation(
                blocks, block_idx, megred_block_idxes, iou_threshold
            )
            update_vision_child_blocks(
                blocks, block, megred_block_idxes, prev_idx, post_idx, config
            )


def get_layout_structure(
    blocks: List[LayoutParsingBlock],
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

    mask_labels = ["doc_title", "cross_text", "cross_reference"]
    for block_idx, block in enumerate(blocks):
        if block.order_label in mask_labels:
            continue

        for ref_idx, ref_block in enumerate(blocks):
            if block_idx == ref_idx or ref_block.order_label in mask_labels:
                continue

            bbox_iou = calculate_overlap_ratio(block.bbox, ref_block.bbox)
            if bbox_iou > 0:
                if ref_block.order_label == "vision":
                    ref_block.order_label = "cross_text"
                    break
                if block.order_label == "vision" or block.area < ref_block.area:
                    block.order_label = "cross_text"
                    break

            match_projection_iou = calculate_projection_overlap_ratio(
                block.bbox,
                ref_block.bbox,
                "horizontal",
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
                            second_ref_block.order_label = "cross_text"
                            break
                        if (
                            block.order_label == "vision"
                            or block.area < second_ref_block.area
                        ):
                            block.order_label = "cross_text"
                            break

                    second_match_projection_iou = calculate_projection_overlap_ratio(
                        block.bbox,
                        second_ref_block.bbox,
                        "horizontal",
                    )
                    ref_match_projection_iou = calculate_projection_overlap_ratio(
                        ref_block.bbox,
                        second_ref_block.bbox,
                        "horizontal",
                    )
                    ref_match_projection_iou_ = calculate_projection_overlap_ratio(
                        ref_block.bbox,
                        second_ref_block.bbox,
                        "vertical",
                    )
                    if (
                        second_match_projection_iou > 0
                        and ref_match_projection_iou == 0
                        and ref_match_projection_iou_ > 0
                        and "vision"
                        not in [ref_block.order_label, second_ref_block.order_label]
                    ):
                        block.order_label = (
                            "cross_reference"
                            if block.label == "reference"
                            else "cross_text"
                        )


def sort_by_xycut(
    block_bboxes: List,
    orientation: int = 0,
    min_gap: int = 1,
) -> List[int]:
    """
    Sort bounding boxes using recursive XY cut method based on the specified orientation.

    Args:
        block_bboxes (Union[np.ndarray, List[List[int]]]): An array or list of bounding boxes,
                                                           where each box is represented as
                                                           [x_min, y_min, x_max, y_max].
        orientation (int): orientation for the initial cut. Use 1 for Y-axis first and 0 for X-axis first.
                         Defaults to 0.
        min_gap (int): Minimum gap width to consider a separation between segments. Defaults to 1.

    Returns:
        List[int]: A list of indices representing the order of sorted bounding boxes.
    """
    block_bboxes = np.asarray(block_bboxes).astype(int)
    res = []
    if orientation == 1:
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
    sorted_blocks: List[LayoutParsingBlock],
    unsorted_blocks: List[LayoutParsingBlock],
    config: Dict,
    median_width: int,
) -> List[LayoutParsingBlock]:
    """
    Match special blocks with the sorted blocks based on their region labels.
    Args:
        sorted_blocks (List[LayoutParsingBlock]): Sorted blocks to be matched.
        unsorted_blocks (List[LayoutParsingBlock]): Unsorted blocks to be matched.
        config (Dict): Configuration dictionary containing various parameters.
        median_width (int): Median width value used for calculations.

    Returns:
        List[LayoutParsingBlock]: The updated sorted blocks after matching special blocks.
    """
    distance_type_map = {
        "cross_text": weighted_distance_insert,
        "paragraph_title": weighted_distance_insert,
        "doc_title": weighted_distance_insert,
        "vision_title": weighted_distance_insert,
        "vision": weighted_distance_insert,
        "cross_reference": reference_insert,
        "unordered": manhattan_insert,
        "other": manhattan_insert,
    }

    unsorted_blocks = sort_blocks(unsorted_blocks, median_width, reverse=False)
    for idx, block in enumerate(unsorted_blocks):
        order_label = block.order_label
        if idx == 0 and order_label == "doc_title":
            sorted_blocks.insert(0, block)
            continue
        sorted_blocks = distance_type_map[order_label](
            block, sorted_blocks, config, median_width
        )
    return sorted_blocks


def xycut_enhanced(
    blocks: List[LayoutParsingBlock], config: Dict
) -> List[LayoutParsingBlock]:
    """
    xycut_enhance function performs the following steps:
        1. Preprocess the input blocks by extracting headers, footers, and pre-cut blocks.
        2. Mask blocks that are crossing different blocks.
        3. Perform xycut_enhanced algorithm on the remaining blocks.
        4. Match unsorted blocks with the sorted blocks based on their order labels.
        5. Update child blocks of the sorted blocks based on their parent blocks.
        6. Return the ordered result list.

    Args:
        blocks (List[LayoutParsingBlock]): Input blocks to be processed.

    Returns:
        List[LayoutParsingBlock]: Ordered result list after processing.
    """
    if len(blocks) == 0:
        return blocks

    text_labels = config.get("text_labels", [])
    header_blocks, pre_cut_list, footer_blocks, unordered_blocks = pre_process(
        blocks, config
    )
    final_order_res_list: List[LayoutParsingBlock] = []

    header_blocks = sort_blocks(header_blocks)
    footer_blocks = sort_blocks(footer_blocks)
    unordered_blocks = sort_blocks(unordered_blocks)
    final_order_res_list.extend(header_blocks)

    unsorted_blocks: List[LayoutParsingBlock] = []
    sorted_blocks_by_pre_cuts = []
    for pre_cut_blocks in pre_cut_list:
        sorted_blocks: List[LayoutParsingBlock] = []
        doc_title_blocks: List[LayoutParsingBlock] = []
        xy_cut_blocks: List[LayoutParsingBlock] = []
        pre_cut_blocks: List[LayoutParsingBlock]
        median_width = 1
        text_block_width = [
            block.width for block in pre_cut_blocks if block.label in text_labels
        ]
        if len(text_block_width) > 0:
            median_width = int(np.median(text_block_width))

        get_layout_structure(
            pre_cut_blocks,
        )

        # Get xy cut blocks and add other blocks in special_block_map
        for block in pre_cut_blocks:
            if block.order_label not in [
                "cross_text",
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
                block_bboxes, orientation="horizontal"
            )
            if len(discontinuous) > 1:
                xy_cut_blocks = [block for block in xy_cut_blocks]
            if len(discontinuous) == 1 or max(block_text_lines) == 1:
                xy_cut_blocks.sort(key=lambda x: (x.bbox[1] // 5, x.bbox[0]))
                xy_cut_blocks = shrink_overlapping_boxes(xy_cut_blocks, "vertical")
                block_bboxes = np.array([block.bbox for block in xy_cut_blocks])
                sorted_indexes = sort_by_xycut(block_bboxes, orientation=1, min_gap=1)
            else:
                xy_cut_blocks.sort(key=lambda x: (x.bbox[0] // 20, x.bbox[1]))
                xy_cut_blocks = shrink_overlapping_boxes(xy_cut_blocks, "horizontal")
                block_bboxes = np.array([block.bbox for block in xy_cut_blocks])
                sorted_indexes = sort_by_xycut(block_bboxes, orientation=0, min_gap=20)

            sorted_blocks = [xy_cut_blocks[i] for i in sorted_indexes]

        sorted_blocks = match_unsorted_blocks(
            sorted_blocks,
            doc_title_blocks,
            config,
            median_width,
        )

        sorted_blocks_by_pre_cuts.extend(sorted_blocks)

    median_width = 1
    text_block_width = [block.width for block in blocks if block.label in text_labels]
    if len(text_block_width) > 0:
        median_width = int(np.median(text_block_width))
    final_order_res_list = match_unsorted_blocks(
        sorted_blocks_by_pre_cuts,
        unsorted_blocks,
        config,
        median_width,
    )

    final_order_res_list.extend(footer_blocks)
    final_order_res_list.extend(unordered_blocks)

    for block_idx, block in enumerate(final_order_res_list):
        final_order_res_list = insert_child_blocks(
            block, block_idx, final_order_res_list
        )
        block = final_order_res_list[block_idx]
    return final_order_res_list
