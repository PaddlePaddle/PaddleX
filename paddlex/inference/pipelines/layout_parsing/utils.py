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

__all__ = [
    "get_sub_regions_ocr_res",
    "get_show_color",
    "sorted_layout_boxes",
]

import re
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from ..components import convert_points_to_boxes
from ..ocr.result import OCRResult
from .setting import BLOCK_LABEL_MAP, REGION_SETTINGS


def get_overlap_boxes_idx(src_boxes: np.ndarray, ref_boxes: np.ndarray) -> List:
    """
    Get the indices of source boxes that overlap with reference boxes based on a specified threshold.

    Args:
        src_boxes (np.ndarray): A 2D numpy array of source bounding boxes.
        ref_boxes (np.ndarray): A 2D numpy array of reference bounding boxes.
    Returns:
        match_idx_list (list): A list of indices of source boxes that overlap with reference boxes.
    """
    match_idx_list = []
    src_boxes_num = len(src_boxes)
    if src_boxes_num > 0 and len(ref_boxes) > 0:
        for rno in range(len(ref_boxes)):
            ref_box = ref_boxes[rno]
            x1 = np.maximum(ref_box[0], src_boxes[:, 0])
            y1 = np.maximum(ref_box[1], src_boxes[:, 1])
            x2 = np.minimum(ref_box[2], src_boxes[:, 2])
            y2 = np.minimum(ref_box[3], src_boxes[:, 3])
            pub_w = x2 - x1
            pub_h = y2 - y1
            match_idx = np.where((pub_w > 3) & (pub_h > 3))[0]
            match_idx_list.extend(match_idx)
    return match_idx_list


def get_sub_regions_ocr_res(
    overall_ocr_res: OCRResult,
    object_boxes: List,
    flag_within: bool = True,
    return_match_idx: bool = False,
) -> OCRResult:
    """
    Filters OCR results to only include text boxes within specified object boxes based on a flag.

    Args:
        overall_ocr_res (OCRResult): The original OCR result containing all text boxes.
        object_boxes (list): A list of bounding boxes for the objects of interest.
        flag_within (bool): If True, only include text boxes within the object boxes. If False, exclude text boxes within the object boxes.
        return_match_idx (bool): If True, return the list of matching indices.

    Returns:
        OCRResult: A filtered OCR result containing only the relevant text boxes.
    """
    sub_regions_ocr_res = {}
    sub_regions_ocr_res["rec_polys"] = []
    sub_regions_ocr_res["rec_texts"] = []
    sub_regions_ocr_res["rec_scores"] = []
    sub_regions_ocr_res["rec_boxes"] = []

    overall_text_boxes = overall_ocr_res["rec_boxes"]
    match_idx_list = get_overlap_boxes_idx(overall_text_boxes, object_boxes)
    match_idx_list = list(set(match_idx_list))
    for box_no in range(len(overall_text_boxes)):
        if flag_within:
            if box_no in match_idx_list:
                flag_match = True
            else:
                flag_match = False
        else:
            if box_no not in match_idx_list:
                flag_match = True
            else:
                flag_match = False
        if flag_match:
            sub_regions_ocr_res["rec_polys"].append(
                overall_ocr_res["rec_polys"][box_no]
            )
            sub_regions_ocr_res["rec_texts"].append(
                overall_ocr_res["rec_texts"][box_no]
            )
            sub_regions_ocr_res["rec_scores"].append(
                overall_ocr_res["rec_scores"][box_no]
            )
            sub_regions_ocr_res["rec_boxes"].append(
                overall_ocr_res["rec_boxes"][box_no]
            )
    for key in ["rec_polys", "rec_scores", "rec_boxes"]:
        sub_regions_ocr_res[key] = np.array(sub_regions_ocr_res[key])
    return (
        (sub_regions_ocr_res, match_idx_list)
        if return_match_idx
        else sub_regions_ocr_res
    )


def sorted_layout_boxes(res, w):
    """
    Sort text boxes in order from top to bottom, left to right
    Args:
        res: List of dictionaries containing layout information.
        w: Width of image.

    Returns:
        List of dictionaries containing sorted layout information.
    """
    num_boxes = len(res)
    if num_boxes == 1:
        return res

    # Sort on the y axis first or sort it on the x axis
    sorted_boxes = sorted(res, key=lambda x: (x["block_bbox"][1], x["block_bbox"][0]))
    _boxes = list(sorted_boxes)

    new_res = []
    res_left = []
    res_right = []
    i = 0

    while True:
        if i >= num_boxes:
            break
        # Check that the bbox is on the left
        elif (
            _boxes[i]["block_bbox"][0] < w / 4
            and _boxes[i]["block_bbox"][2] < 3 * w / 5
        ):
            res_left.append(_boxes[i])
            i += 1
        elif _boxes[i]["block_bbox"][0] > 2 * w / 5:
            res_right.append(_boxes[i])
            i += 1
        else:
            new_res += res_left
            new_res += res_right
            new_res.append(_boxes[i])
            res_left = []
            res_right = []
            i += 1

    res_left = sorted(res_left, key=lambda x: (x["block_bbox"][1]))
    res_right = sorted(res_right, key=lambda x: (x["block_bbox"][1]))

    if res_left:
        new_res += res_left
    if res_right:
        new_res += res_right

    return new_res


def calculate_projection_overlap_ratio(
    bbox1: List[float],
    bbox2: List[float],
    direction: str = "horizontal",
    mode="union",
) -> float:
    """
    Calculate the IoU of lines between two bounding boxes.

    Args:
        bbox1 (List[float]): First bounding box [x_min, y_min, x_max, y_max].
        bbox2 (List[float]): Second bounding box [x_min, y_min, x_max, y_max].
        direction (str): direction of the projection, "horizontal" or "vertical".

    Returns:
        float: Line overlap ratio. Returns 0 if there is no overlap.
    """
    start_index, end_index = 1, 3
    if direction == "horizontal":
        start_index, end_index = 0, 2

    intersection_start = max(bbox1[start_index], bbox2[start_index])
    intersection_end = min(bbox1[end_index], bbox2[end_index])
    overlap = intersection_end - intersection_start
    if overlap <= 0:
        return 0

    if mode == "union":
        ref_width = max(bbox1[end_index], bbox2[end_index]) - min(
            bbox1[start_index], bbox2[start_index]
        )
    elif mode == "small":
        ref_width = min(
            bbox1[end_index] - bbox1[start_index], bbox2[end_index] - bbox2[start_index]
        )
    elif mode == "large":
        ref_width = max(
            bbox1[end_index] - bbox1[start_index], bbox2[end_index] - bbox2[start_index]
        )
    else:
        raise ValueError(
            f"Invalid mode {mode}, must be one of ['union', 'small', 'large']."
        )

    return overlap / ref_width if ref_width > 0 else 0.0


def calculate_overlap_ratio(
    bbox1: Union[list, tuple], bbox2: Union[list, tuple], mode="union"
) -> float:
    """
    Calculate the overlap ratio between two bounding boxes.

    Args:
        bbox1 (list or tuple): The first bounding box, format [x_min, y_min, x_max, y_max]
        bbox2 (list or tuple): The second bounding box, format [x_min, y_min, x_max, y_max]
        mode (str): The mode of calculation, either 'union', 'small', or 'large'.

    Returns:
        float: The overlap ratio value between the two bounding boxes
    """
    x_min_inter = max(bbox1[0], bbox2[0])
    y_min_inter = max(bbox1[1], bbox2[1])
    x_max_inter = min(bbox1[2], bbox2[2])
    y_max_inter = min(bbox1[3], bbox2[3])

    inter_width = max(0, x_max_inter - x_min_inter)
    inter_height = max(0, y_max_inter - y_min_inter)

    inter_area = float(inter_width) * float(inter_height)

    bbox1_area = caculate_bbox_area(bbox1)
    bbox2_area = caculate_bbox_area(bbox2)

    if mode == "union":
        ref_area = bbox1_area + bbox2_area - inter_area
    elif mode == "small":
        ref_area = min(bbox1_area, bbox2_area)
    elif mode == "large":
        ref_area = max(bbox1_area, bbox2_area)
    else:
        raise ValueError(
            f"Invalid mode {mode}, must be one of ['union', 'small', 'large']."
        )

    if ref_area == 0:
        return 0.0

    return inter_area / ref_area


def calculate_minimum_enclosing_bbox(bboxes):
    """
    Calculate the minimum enclosing bounding box for a list of bounding boxes.

    Args:
        bboxes (list): A list of bounding boxes represented as lists of four integers [x1, y1, x2, y2].

    Returns:
        list: The minimum enclosing bounding box represented as a list of four integers [x1, y1, x2, y2].
    """
    if not bboxes:
        raise ValueError("The list of bounding boxes is empty.")

    # Convert the list of bounding boxes to a NumPy array
    bboxes_array = np.array(bboxes)

    # Compute the minimum and maximum values along the respective axes
    min_x = np.min(bboxes_array[:, 0])
    min_y = np.min(bboxes_array[:, 1])
    max_x = np.max(bboxes_array[:, 2])
    max_y = np.max(bboxes_array[:, 3])

    # Return the minimum enclosing bounding box
    return np.array([min_x, min_y, max_x, max_y])


def is_english_letter(char):
    """check if the char is english letter"""
    return bool(re.match(r"^[A-Za-z]$", char))


def is_numeric(char):
    """check if the char is numeric"""
    return bool(re.match(r"^[\d]+$", char))


def is_non_breaking_punctuation(char):
    """
    check if the char is non-breaking punctuation

    Args:
        char (str): character to check

    Returns:
        bool: True if the char is non-breaking punctuation
    """
    non_breaking_punctuations = {
        ",",
        "，",
        "、",
        ";",
        "；",
        ":",
        "：",
        "-",
        "'",
        '"',
        "“",
    }

    return char in non_breaking_punctuations


def gather_imgs(original_img, layout_det_objs):
    imgs_in_doc = []
    for det_obj in layout_det_objs:
        if det_obj["label"] in BLOCK_LABEL_MAP["image_labels"]:
            label = det_obj["label"]
            x_min, y_min, x_max, y_max = list(map(int, det_obj["coordinate"]))
            img_path = f"imgs/img_in_{label}_box_{x_min}_{y_min}_{x_max}_{y_max}.jpg"
            img = Image.fromarray(original_img[y_min:y_max, x_min:x_max, ::-1])
            imgs_in_doc.append(
                {
                    "path": img_path,
                    "img": img,
                    "coordinate": (x_min, y_min, x_max, y_max),
                    "score": det_obj["score"],
                }
            )
    return imgs_in_doc


def _get_minbox_if_overlap_by_ratio(
    bbox1: Union[List[int], Tuple[int, int, int, int]],
    bbox2: Union[List[int], Tuple[int, int, int, int]],
    ratio: float,
    smaller: bool = True,
) -> Optional[Union[List[int], Tuple[int, int, int, int]]]:
    """
    Determine if the overlap area between two bounding boxes exceeds a given ratio
    and return the smaller (or larger) bounding box based on the `smaller` flag.

    Args:
        bbox1 (Union[List[int], Tuple[int, int, int, int]]): Coordinates of the first bounding box [x_min, y_min, x_max, y_max].
        bbox2 (Union[List[int], Tuple[int, int, int, int]]): Coordinates of the second bounding box [x_min, y_min, x_max, y_max].
        ratio (float): The overlap ratio threshold.
        smaller (bool): If True, return the smaller bounding box; otherwise, return the larger one.

    Returns:
        Optional[Union[List[int], Tuple[int, int, int, int]]]:
            The selected bounding box or None if the overlap ratio is not exceeded.
    """
    # Calculate the areas of both bounding boxes
    area1 = caculate_bbox_area(bbox1)
    area2 = caculate_bbox_area(bbox2)
    # Calculate the overlap ratio using a helper function
    overlap_ratio = calculate_overlap_ratio(bbox1, bbox2, mode="small")
    # Check if the overlap ratio exceeds the threshold
    if overlap_ratio > ratio:
        if (area1 <= area2 and smaller) or (area1 >= area2 and not smaller):
            return 1
        else:
            return 2
    return None


def remove_overlap_blocks(
    blocks: List[Dict[str, List[int]]], threshold: float = 0.65, smaller: bool = True
) -> Tuple[List[Dict[str, List[int]]], List[Dict[str, List[int]]]]:
    """
    Remove overlapping blocks based on a specified overlap ratio threshold.

    Args:
        blocks (List[Dict[str, List[int]]]): List of block dictionaries, each containing a 'block_bbox' key.
        threshold (float): Ratio threshold to determine significant overlap.
        smaller (bool): If True, the smaller block in overlap is removed.

    Returns:
        Tuple[List[Dict[str, List[int]]], List[Dict[str, List[int]]]]:
            A tuple containing the updated list of blocks and a list of dropped blocks.
    """
    dropped_indexes = set()
    blocks = deepcopy(blocks)
    overlap_image_blocks = []
    # Iterate over each pair of blocks to find overlaps
    for i, block1 in enumerate(blocks["boxes"]):
        for j in range(i + 1, len(blocks["boxes"])):
            block2 = blocks["boxes"][j]
            # Skip blocks that are already marked for removal
            if i in dropped_indexes or j in dropped_indexes:
                continue
            # Check for overlap and determine which block to remove
            overlap_box_index = _get_minbox_if_overlap_by_ratio(
                block1["coordinate"],
                block2["coordinate"],
                threshold,
                smaller=smaller,
            )
            if overlap_box_index is not None:
                is_block1_image = block1["label"] == "image"
                is_block2_image = block2["label"] == "image"

                if is_block1_image != is_block2_image:
                    # 如果只有一个块在视觉标签中，删除在视觉标签中的那个块
                    drop_index = i if is_block1_image else j
                    overlap_image_blocks.append(blocks["boxes"][drop_index])
                else:
                    # 如果两个块都在或都不在视觉标签中，根据 overlap_box_index 决定删除哪个块
                    drop_index = i if overlap_box_index == 1 else j

                dropped_indexes.add(drop_index)

    # Remove marked blocks from the original list
    for index in sorted(dropped_indexes, reverse=True):
        del blocks["boxes"][index]

    return blocks


def get_bbox_intersection(bbox1, bbox2, return_format="bbox"):
    """
    Compute the intersection of two bounding boxes, supporting both 4-coordinate and 8-coordinate formats.

    Args:
        bbox1 (tuple): The first bounding box, either in 4-coordinate format (x_min, y_min, x_max, y_max)
                       or 8-coordinate format (x1, y1, x2, y2, x3, y3, x4, y4).
        bbox2 (tuple): The second bounding box in the same format as bbox1.
        return_format (str): The format of the output intersection, either 'bbox' or 'poly'.

    Returns:
        tuple or None: The intersection bounding box in the specified format, or None if there is no intersection.
    """
    bbox1 = np.array(bbox1)
    bbox2 = np.array(bbox2)
    # Convert both bounding boxes to rectangles
    rect1 = bbox1 if len(bbox1.shape) == 1 else convert_points_to_boxes([bbox1])[0]
    rect2 = bbox2 if len(bbox2.shape) == 1 else convert_points_to_boxes([bbox2])[0]

    # Calculate the intersection rectangle

    x_min_inter = max(rect1[0], rect2[0])
    y_min_inter = max(rect1[1], rect2[1])
    x_max_inter = min(rect1[2], rect2[2])
    y_max_inter = min(rect1[3], rect2[3])

    # Check if there is an intersection
    if x_min_inter >= x_max_inter or y_min_inter >= y_max_inter:
        return None

    if return_format == "bbox":
        return np.array([x_min_inter, y_min_inter, x_max_inter, y_max_inter])
    elif return_format == "poly":
        return np.array(
            [
                [x_min_inter, y_min_inter],
                [x_max_inter, y_min_inter],
                [x_max_inter, y_max_inter],
                [x_min_inter, y_max_inter],
            ],
            dtype=np.int16,
        )
    else:
        raise ValueError("return_format must be either 'bbox' or 'poly'.")


def shrink_supplement_region_bbox(
    supplement_region_bbox,
    ref_region_bbox,
    image_width,
    image_height,
    block_idxes_set,
    block_bboxes,
) -> List:
    """
    Shrink the supplement region bbox according to the reference region bbox and match the block bboxes.

    Args:
        supplement_region_bbox (list): The supplement region bbox.
        ref_region_bbox (list): The reference region bbox.
        image_width (int): The width of the image.
        image_height (int): The height of the image.
        block_idxes_set (set): The indexes of the blocks that intersect with the region bbox.
        block_bboxes (dict): The dictionary of block bboxes.

    Returns:
        list: The new region bbox and the matched block idxes.
    """
    x1, y1, x2, y2 = supplement_region_bbox
    x1_prime, y1_prime, x2_prime, y2_prime = ref_region_bbox
    index_conversion_map = {0: 2, 1: 3, 2: 0, 3: 1}
    edge_distance_list = [
        (x1_prime - x1) / image_width,
        (y1_prime - y1) / image_height,
        (x2 - x2_prime) / image_width,
        (y2 - y2_prime) / image_height,
    ]
    edge_distance_list_tmp = deepcopy(edge_distance_list)
    min_distance = min(edge_distance_list)
    src_index = index_conversion_map[edge_distance_list.index(min_distance)]
    if len(block_idxes_set) == 0:
        return supplement_region_bbox, []
    for _ in range(3):
        dst_index = index_conversion_map[src_index]
        tmp_region_bbox = supplement_region_bbox[:]
        tmp_region_bbox[dst_index] = ref_region_bbox[src_index]
        iner_block_idxes, split_block_idxes = [], []
        for block_idx in block_idxes_set:
            overlap_ratio = calculate_overlap_ratio(
                tmp_region_bbox, block_bboxes[block_idx], mode="small"
            )
            if overlap_ratio > REGION_SETTINGS.get(
                "match_block_overlap_ratio_threshold", 0.8
            ):
                iner_block_idxes.append(block_idx)
            elif overlap_ratio > REGION_SETTINGS.get(
                "split_block_overlap_ratio_threshold", 0.4
            ):
                split_block_idxes.append(block_idx)

        if len(iner_block_idxes) > 0:
            if len(split_block_idxes) > 0:
                for split_block_idx in split_block_idxes:
                    split_block_bbox = block_bboxes[split_block_idx]
                    x1, y1, x2, y2 = tmp_region_bbox
                    x1_prime, y1_prime, x2_prime, y2_prime = split_block_bbox
                    edge_distance_list = [
                        (x1_prime - x1) / image_width,
                        (y1_prime - y1) / image_height,
                        (x2 - x2_prime) / image_width,
                        (y2 - y2_prime) / image_height,
                    ]
                    max_distance = max(edge_distance_list)
                    src_index = edge_distance_list.index(max_distance)
                    dst_index = index_conversion_map[src_index]
                    tmp_region_bbox[dst_index] = split_block_bbox[src_index]
                    tmp_region_bbox, iner_idxes = shrink_supplement_region_bbox(
                        tmp_region_bbox,
                        ref_region_bbox,
                        image_width,
                        image_height,
                        iner_block_idxes,
                        block_bboxes,
                    )
                    if len(iner_idxes) == 0:
                        continue
            matched_bboxes = [block_bboxes[idx] for idx in iner_block_idxes]
            supplement_region_bbox = calculate_minimum_enclosing_bbox(matched_bboxes)
            break
        else:
            edge_distance_list_tmp.remove(min_distance)
            min_distance = min(edge_distance_list_tmp)
            src_index = index_conversion_map[edge_distance_list.index(min_distance)]
    return supplement_region_bbox, iner_block_idxes


def update_region_box(bbox, region_box):
    """Update region box with bbox"""
    if region_box is None:
        return bbox

    x1, y1, x2, y2 = bbox
    x1_region, y1_region, x2_region, y2_region = region_box

    x1_region = int(min(x1, x1_region))
    y1_region = int(min(y1, y1_region))
    x2_region = int(max(x2, x2_region))
    y2_region = int(max(y2, y2_region))

    region_box = [x1_region, y1_region, x2_region, y2_region]

    return region_box


def convert_formula_res_to_ocr_format(formula_res_list: List, ocr_res: dict):
    """Convert formula result to OCR result format

    Args:
        formula_res_list (List): Formula results
        ocr_res (dict): OCR result
    Returns:
        ocr_res (dict): Updated OCR result
    """
    for formula_res in formula_res_list:
        x_min, y_min, x_max, y_max = list(map(int, formula_res["dt_polys"]))
        poly_points = [
            (x_min, y_min),
            (x_max, y_min),
            (x_max, y_max),
            (x_min, y_max),
        ]
        ocr_res["dt_polys"].append(poly_points)
        formula_res_text: str = formula_res["rec_formula"]
        ocr_res["rec_texts"].append(formula_res_text)
        if ocr_res["rec_boxes"].size == 0:
            ocr_res["rec_boxes"] = np.array(formula_res["dt_polys"])
        else:
            ocr_res["rec_boxes"] = np.vstack(
                (ocr_res["rec_boxes"], [formula_res["dt_polys"]])
            )
        ocr_res["rec_labels"].append("formula")
        ocr_res["rec_polys"].append(poly_points)
        ocr_res["rec_scores"].append(1)


def caculate_bbox_area(bbox):
    """Calculate bounding box area"""
    x1, y1, x2, y2 = map(float, bbox)
    area = abs((x2 - x1) * (y2 - y1))
    return area


def caculate_euclidean_dist(point1, point2):
    """Calculate euclidean distance between two points"""
    x1, y1 = point1
    x2, y2 = point2
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def get_seg_flag(block, prev_block):
    """Get segment start flag and end flag based on previous block

    Args:
        block (Block): Current block
        prev_block (Block): Previous block

    Returns:
        seg_start_flag (bool): Segment start flag
        seg_end_flag (bool): Segment end flag
    """

    seg_start_flag = True
    seg_end_flag = True

    context_left_coordinate = block.start_coordinate
    context_right_coordinate = block.end_coordinate
    seg_start_coordinate = block.seg_start_coordinate
    seg_end_coordinate = block.seg_end_coordinate

    if prev_block is not None:
        num_of_prev_lines = prev_block.num_of_lines
        pre_block_seg_end_coordinate = prev_block.seg_end_coordinate
        prev_end_space_small = (
            abs(prev_block.end_coordinate - pre_block_seg_end_coordinate) < 10
        )
        prev_lines_more_than_one = num_of_prev_lines > 1

        overlap_blocks = (
            context_left_coordinate < prev_block.end_coordinate
            and context_right_coordinate > prev_block.start_coordinate
        )

        # update context_left_coordinate and context_right_coordinate
        if overlap_blocks:
            context_left_coordinate = min(
                prev_block.start_coordinate, context_left_coordinate
            )
            context_right_coordinate = max(
                prev_block.end_coordinate, context_right_coordinate
            )
            prev_end_space_small = (
                abs(context_right_coordinate - pre_block_seg_end_coordinate) < 10
            )
            edge_distance = 0
        else:
            edge_distance = abs(block.start_coordinate - prev_block.end_coordinate)

        current_start_space_small = seg_start_coordinate - context_left_coordinate < 10

        if (
            prev_end_space_small
            and current_start_space_small
            and prev_lines_more_than_one
            and edge_distance < max(prev_block.width, block.width)
        ):
            seg_start_flag = False
    else:
        if seg_start_coordinate - context_left_coordinate < 10:
            seg_start_flag = False

    if context_right_coordinate - seg_end_coordinate < 10:
        seg_end_flag = False

    return seg_start_flag, seg_end_flag


def get_show_color(label: str, order_label=False) -> Tuple:
    if order_label:
        label_colors = {
            "doc_title": (255, 248, 220, 100),  # Cornsilk
            "doc_title_text": (255, 239, 213, 100),
            "paragraph_title": (102, 102, 255, 100),
            "sub_paragraph_title": (102, 178, 255, 100),
            "vision": (153, 255, 51, 100),
            "vision_title": (144, 238, 144, 100),  # Light Green
            "vision_footnote": (144, 238, 144, 100),  # Light Green
            "normal_text": (153, 0, 76, 100),
            "cross_layout": (53, 218, 207, 100),  # Thistle
            "cross_reference": (221, 160, 221, 100),  # Floral White
        }
    else:
        label_colors = {
            # Medium Blue (from 'titles_list')
            "paragraph_title": (102, 102, 255, 100),
            "doc_title": (255, 248, 220, 100),  # Cornsilk
            # Light Yellow (from 'tables_caption_list')
            "table_title": (255, 255, 102, 100),
            # Sky Blue (from 'imgs_caption_list')
            "figure_title": (102, 178, 255, 100),
            "chart_title": (221, 160, 221, 100),  # Plum
            "vision_footnote": (144, 238, 144, 100),  # Light Green
            # Deep Purple (from 'texts_list')
            "text": (153, 0, 76, 100),
            # Bright Green (from 'interequations_list')
            "formula": (0, 255, 0, 100),
            "abstract": (255, 239, 213, 100),  # Papaya Whip
            # Medium Green (from 'lists_list' and 'indexs_list')
            "content": (40, 169, 92, 100),
            # Neutral Gray (from 'dropped_bbox_list')
            "seal": (158, 158, 158, 100),
            # Olive Yellow (from 'tables_body_list')
            "table": (204, 204, 0, 100),
            # Bright Green (from 'imgs_body_list')
            "image": (153, 255, 51, 100),
            # Bright Green (from 'imgs_body_list')
            "figure": (153, 255, 51, 100),
            "chart": (216, 191, 216, 100),  # Thistle
            # Pale Yellow-Green (from 'tables_footnote_list')
            "reference": (229, 255, 204, 100),
            # "reference_content": (229, 255, 204, 100),
            "algorithm": (255, 250, 240, 100),  # Floral White
        }
    default_color = (158, 158, 158, 100)
    return label_colors.get(label, default_color)
