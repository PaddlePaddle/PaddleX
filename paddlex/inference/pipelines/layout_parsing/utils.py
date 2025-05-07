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
    "update_layout_order_config_block_index",
]

import re
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from ..components import convert_points_to_boxes
from ..ocr.result import OCRResult
from .xycut_enhanced import calculate_projection_iou


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


def _calculate_overlap_area_div_minbox_area_ratio(
    bbox1: Union[list, tuple],
    bbox2: Union[list, tuple],
) -> float:
    """
    Calculate the ratio of the overlap area between bbox1 and bbox2
    to the area of the smaller bounding box.

    Args:
        bbox1 (list or tuple): Coordinates of the first bounding box [x_min, y_min, x_max, y_max].
        bbox2 (list or tuple): Coordinates of the second bounding box [x_min, y_min, x_max, y_max].

    Returns:
        float: The ratio of the overlap area to the area of the smaller bounding box.
    """
    bbox1 = list(map(int, bbox1))
    bbox2 = list(map(int, bbox2))

    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    area_bbox1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area_bbox2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    min_box_area = min(area_bbox1, area_bbox2)

    if min_box_area <= 0:
        return 0.0

    return intersection_area / min_box_area


def group_boxes_into_lines(ocr_rec_res, block_info, line_height_iou_threshold):
    rec_boxes = ocr_rec_res["boxes"]
    rec_texts = ocr_rec_res["rec_texts"]
    rec_labels = ocr_rec_res["rec_labels"]

    spans = list(zip(rec_boxes, rec_texts, rec_labels))

    spans.sort(key=lambda span: span[0][1])
    spans = [list(span) for span in spans]

    lines = []
    line = [spans[0]]
    line_region_box = spans[0][0][:]
    block_info.seg_start_coordinate = spans[0][0][0]
    block_info.seg_end_coordinate = spans[-1][0][2]

    # merge line
    for span in spans[1:]:
        rec_bbox = span[0]
        if (
            calculate_projection_iou(line_region_box, rec_bbox, "vertical")
            >= line_height_iou_threshold
        ):
            line.append(span)
            line_region_box[1] = min(line_region_box[1], rec_bbox[1])
            line_region_box[3] = max(line_region_box[3], rec_bbox[3])
        else:
            lines.append(line)
            line = [span]
            line_region_box = rec_bbox[:]

    lines.append(line)
    return lines


def calculate_text_orientation(
    bboxes: List[List[int]], orientation_ratio: float = 1.5
) -> bool:
    """
    Calculate the orientation of the text based on the bounding boxes.

    Args:
        bboxes (list): A list of bounding boxes.
        orientation_ratio (float): Ratio for determining orientation. Default is 1.5.

    Returns:
        str: "horizontal" or "vertical".
    """

    bboxes = np.array(bboxes)
    x_min = np.min(bboxes[:, 0])
    x_max = np.max(bboxes[:, 2])
    width = x_max - x_min
    y_min = np.min(bboxes[:, 1])
    y_max = np.max(bboxes[:, 3])
    height = y_max - y_min
    return "horizontal" if width * orientation_ratio >= height else "vertical"


def format_line(
    line: List[List[Union[List[int], str]]],
    block_left_coordinate: int,
    block_right_coordinate: int,
    first_line_span_limit: int = 10,
    last_line_span_limit: int = 10,
    block_label: str = "text",
    delimiter_map: Dict = {},
) -> None:
    """
    Format a line of text spans based on layout constraints.

    Args:
        line (list): A list of spans, where each span is a list containing a bounding box and text.
        block_left_coordinate (int): The minimum x-coordinate of the layout bounding box.
        block_right_coordinate (int): The maximum x-coordinate of the layout bounding box.
        first_line_span_limit (int): The limit for the number of pixels before the first span that should be considered part of the first line. Default is 10.
        last_line_span_limit (int): The limit for the number of pixels after the last span that should be considered part of the last line. Default is 10.
        block_label (str): The label associated with the entire block. Default is 'text'.
    Returns:
        None: The function modifies the line in place.
    """
    first_span = line[0]
    last_span = line[-1]

    if first_span[0][0] - block_left_coordinate > first_line_span_limit:
        first_span[1] = "\n" + first_span[1]
    if block_right_coordinate - last_span[0][2] > last_line_span_limit:
        last_span[1] = last_span[1] + "\n"

    line[0] = first_span
    line[-1] = last_span

    delim = delimiter_map.get(block_label, " ")
    line_text = delim.join([span[1] for span in line])

    if block_label != "reference":
        line_text = remove_extra_space(line_text)

    if line_text.endswith("-"):
        line_text = line_text[:-1]
    return line_text


def split_boxes_if_x_contained(boxes, offset=1e-5):
    """
    Check if there is any complete containment in the x-direction
    between the bounding boxes and split the containing box accordingly.

    Args:
        boxes (list of lists): Each element is a list containing an ndarray of length 4, a description, and a label.
        offset (float): A small offset value to ensure that the split boxes are not too close to the original boxes.
    Returns:
        A new list of boxes, including split boxes, with the same `rec_text` and `label` attributes.
    """

    def is_x_contained(box_a, box_b):
        """Check if box_a completely contains box_b in the x-direction."""
        return box_a[0][0] <= box_b[0][0] and box_a[0][2] >= box_b[0][2]

    new_boxes = []

    for i in range(len(boxes)):
        box_a = boxes[i]
        is_split = False
        for j in range(len(boxes)):
            if i == j:
                continue
            box_b = boxes[j]
            if is_x_contained(box_a, box_b):
                is_split = True
                # Split box_a based on the x-coordinates of box_b
                if box_a[0][0] < box_b[0][0]:
                    w = box_b[0][0] - offset - box_a[0][0]
                    if w > 1:
                        new_boxes.append(
                            [
                                np.array(
                                    [
                                        box_a[0][0],
                                        box_a[0][1],
                                        box_b[0][0] - offset,
                                        box_a[0][3],
                                    ]
                                ),
                                box_a[1],
                                box_a[2],
                            ]
                        )
                if box_a[0][2] > box_b[0][2]:
                    w = box_a[0][2] - box_b[0][2] + offset
                    if w > 1:
                        box_a = [
                            np.array(
                                [
                                    box_b[0][2] + offset,
                                    box_a[0][1],
                                    box_a[0][2],
                                    box_a[0][3],
                                ]
                            ),
                            box_a[1],
                            box_a[2],
                        ]
            if j == len(boxes) - 1 and is_split:
                new_boxes.append(box_a)
        if not is_split:
            new_boxes.append(box_a)

    return new_boxes


def remove_extra_space(input_text: str) -> str:
    """
    Process the input text to handle spaces.

    The function removes multiple consecutive spaces between Chinese characters and ensures that
    only a single space is retained between Chinese and non-Chinese characters.

    Args:
        input_text (str): The text to be processed.

    Returns:
        str: The processed text with properly formatted spaces.
    """

    # Remove spaces between Chinese characters
    text_without_spaces = re.sub(
        r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", input_text
    )

    # Ensure single space between Chinese and non-Chinese characters
    text_with_single_spaces = re.sub(
        r"(?<=[\u4e00-\u9fff])\s+(?=[^\u4e00-\u9fff])|(?<=[^\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])",
        " ",
        text_without_spaces,
    )

    # Reduce any remaining consecutive spaces to a single space
    final_text = re.sub(r"\s+", " ", text_with_single_spaces).strip()

    return final_text


def gather_imgs(original_img, layout_det_objs):
    imgs_in_doc = []
    for det_obj in layout_det_objs:
        if det_obj["label"] in ("image", "chart"):
            x_min, y_min, x_max, y_max = list(map(int, det_obj["coordinate"]))
            img_path = f"imgs/img_in_table_box_{x_min}_{y_min}_{x_max}_{y_max}.jpg"
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
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    # Calculate the overlap ratio using a helper function
    overlap_ratio = _calculate_overlap_area_div_minbox_area_ratio(bbox1, bbox2)
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
                # Determine which block to remove based on overlap_box_index
                if overlap_box_index == 1:
                    drop_index = i
                else:
                    drop_index = j
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


def update_layout_order_config_block_index(
    config: dict, block_label: str, block_idx: int
) -> None:

    doc_title_labels = config["doc_title_labels"]
    paragraph_title_labels = config["paragraph_title_labels"]
    vision_labels = config["vision_labels"]
    vision_title_labels = config["vision_title_labels"]
    header_labels = config["header_labels"]
    unordered_labels = config["unordered_labels"]
    footer_labels = config["footer_labels"]
    text_labels = config["text_labels"]
    text_title_labels = doc_title_labels + paragraph_title_labels
    config["text_title_labels"] = text_title_labels

    if block_label in doc_title_labels:
        config["doc_title_block_idxes"].append(block_idx)
    if block_label in paragraph_title_labels:
        config["paragraph_title_block_idxes"].append(block_idx)
    if block_label in vision_labels:
        config["vision_block_idxes"].append(block_idx)
    if block_label in vision_title_labels:
        config["vision_title_block_idxes"].append(block_idx)
    if block_label in unordered_labels:
        config["unordered_block_idxes"].append(block_idx)
    if block_label in text_title_labels:
        config["text_title_block_idxes"].append(block_idx)
    if block_label in text_labels:
        config["text_block_idxes"].append(block_idx)
    if block_label in header_labels:
        config["header_block_idxes"].append(block_idx)
    if block_label in footer_labels:
        config["footer_block_idxes"].append(block_idx)


def update_region_box(bbox, region_box):
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
    for formula_res in formula_res_list:
        x_min, y_min, x_max, y_max = list(map(int, formula_res["dt_polys"]))
        poly_points = [
            (x_min, y_min),
            (x_max, y_min),
            (x_max, y_max),
            (x_min, y_max),
        ]
        ocr_res["dt_polys"].append(poly_points)
        ocr_res["rec_texts"].append(f"${formula_res['rec_formula']}$")
        ocr_res["rec_boxes"] = np.vstack(
            (ocr_res["rec_boxes"], [formula_res["dt_polys"]])
        )
        ocr_res["rec_labels"].append("formula")
        ocr_res["rec_polys"].append(poly_points)
        ocr_res["rec_scores"].append(1)


def caculate_bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    area = abs((x2 - x1) * (y2 - y1))
    return area


def get_show_color(label: str) -> Tuple:
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
        "algorithm": (255, 250, 240, 100),  # Floral White
    }
    default_color = (158, 158, 158, 100)
    return label_colors.get(label, default_color)
