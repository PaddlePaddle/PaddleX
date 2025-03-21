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
from typing import Dict

import numpy as np

from ..components import convert_points_to_boxes
from ..layout_parsing.utils import get_sub_regions_ocr_res
from ..ocr.result import OCRResult
from .result import SingleTableRecognitionResult


def get_ori_image_coordinate(x: int, y: int, box_list: list) -> list:
    """
    get the original coordinate from Cropped image to Original image.
    Args:
        x (int): x coordinate of cropped image
        y (int): y coordinate of cropped image
        box_list (list): list of table bounding boxes, eg. [[x1, y1, x2, y2, x3, y3, x4, y4]]
    Returns:
        list: list of original coordinates, eg. [[x1, y1, x2, y2, x3, y3, x4, y4]]
    """
    if not box_list:
        return box_list
    offset = np.array([x, y] * 4)
    box_list = np.array(box_list)
    if box_list.shape[-1] == 2:
        offset = offset.reshape(4, 2)
    ori_box_list = offset + box_list
    return ori_box_list


def convert_table_structure_pred_bbox(
    table_structure_pred: Dict, crop_start_point: list, img_shape: tuple
) -> None:
    """
    Convert the predicted table structure bounding boxes to the original image coordinate system.

    Args:
        table_structure_pred (Dict): A dictionary containing the predicted table structure, including bounding boxes ('bbox').
        crop_start_point (list): A list of two integers representing the starting point (x, y) of the cropped image region.
        img_shape (tuple): A tuple of two integers representing the shape (height, width) of the original image.

    Returns:
        None: The function modifies the 'table_structure_pred' dictionary in place by adding the 'cell_box_list' key.
    """

    cell_points_list = table_structure_pred["bbox"]
    ori_cell_points_list = get_ori_image_coordinate(
        crop_start_point[0], crop_start_point[1], cell_points_list
    )
    ori_cell_points_list = np.reshape(ori_cell_points_list, (-1, 4, 2))
    cell_box_list = convert_points_to_boxes(ori_cell_points_list)

    img_height, img_width = img_shape
    cell_box_list = np.clip(
        cell_box_list, 0, [img_width, img_height, img_width, img_height]
    )
    table_structure_pred["cell_box_list"] = cell_box_list
    return


def distance(box_1: list, box_2: list) -> float:
    """
    compute the distance between two boxes

    Args:
        box_1 (list): first rectangle box,eg.(x1, y1, x2, y2)
        box_2 (list): second rectangle box,eg.(x1, y1, x2, y2)

    Returns:
        float: the distance between two boxes
    """
    x1, y1, x2, y2 = box_1
    x3, y3, x4, y4 = box_2
    dis = abs(x3 - x1) + abs(y3 - y1) + abs(x4 - x2) + abs(y4 - y2)
    dis_2 = abs(x3 - x1) + abs(y3 - y1)
    dis_3 = abs(x4 - x2) + abs(y4 - y2)
    return dis + min(dis_2, dis_3)


def compute_iou(rec1: list, rec2: list) -> float:
    """
    computing IoU
    Args:
        rec1 (list): (x1, y1, x2, y2)
        rec2 (list): (x1, y1, x2, y2)
    Returns:
        float: Intersection over Union
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[0], rec2[0])
    right_line = min(rec1[2], rec2[2])
    top_line = max(rec1[1], rec2[1])
    bottom_line = min(rec1[3], rec2[3])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0.0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect)) * 1.0


def _whether_y_overlap_exceeds_threshold(bbox1, bbox2, overlap_ratio_threshold=0.6):
    """
    Determines whether the vertical overlap between two bounding boxes exceeds a given threshold.

    Args:
        bbox1 (tuple): The first bounding box defined as (left, top, right, bottom).
        bbox2 (tuple): The second bounding box defined as (left, top, right, bottom).
        overlap_ratio_threshold (float): The threshold ratio to determine if the overlap is significant.
                                         Defaults to 0.6.

    Returns:
        bool: True if the vertical overlap divided by the minimum height of the two bounding boxes
              exceeds the overlap_ratio_threshold, otherwise False.
    """
    _, y1_0, _, y1_1 = bbox1
    _, y2_0, _, y2_1 = bbox2

    overlap = max(0, min(y1_1, y2_1) - max(y1_0, y2_0))
    min_height = min(y1_1 - y1_0, y2_1 - y2_0)

    return (overlap / min_height) > overlap_ratio_threshold


def _sort_box_by_y_projection(boxes, line_height_iou_threshold=0.6):
    """
    Sorts a list of bounding boxes based on their spatial arrangement.

    The function first sorts the boxes by their top y-coordinate to group them into lines.
    Within each line, the boxes are then sorted by their x-coordinate.

    Args:
        boxes (list): A list of bounding boxes, where each box is defined as [left, top, right, bottom].
        line_height_iou_threshold (float): The Intersection over Union (IoU) threshold for grouping boxes into the same line.

    Returns:
        list: A list of indices representing the order of the boxes after sorting by their spatial arrangement.
    """

    if not boxes:
        return []

    indexed_boxes = list(enumerate(boxes))
    indexed_boxes.sort(key=lambda item: item[1][1])

    lines = []
    first_index, first_box = indexed_boxes[0]
    current_line = [(first_index, first_box)]
    current_y0, current_y1 = first_box[1], first_box[3]

    for index, box in indexed_boxes[1:]:
        y0, y1 = box[1], box[3]
        if _whether_y_overlap_exceeds_threshold(
            (0, current_y0, 0, current_y1),
            (0, y0, 0, y1),
            line_height_iou_threshold,
        ):
            current_line.append((index, box))
            current_y0 = min(current_y0, y0)
            current_y1 = max(current_y1, y1)
        else:
            lines.append(current_line)
            current_line = [(index, box)]
            current_y0, current_y1 = y0, y1

    if current_line:
        lines.append(current_line)

    for line in lines:
        line.sort(key=lambda item: item[1][0])

    sorted_indices = [index for line in lines for index, _ in line]

    return sorted_indices


def match_table_and_ocr(
    cell_box_list: list, ocr_dt_boxes: list, cell_sort_by_y_projection: bool = False
) -> dict:
    """
    match table and ocr

    Args:
        cell_box_list (list): bbox for table cell, 2 points, [left, top, right, bottom]
        ocr_dt_boxes (list): bbox for ocr, 2 points, [left, top, right, bottom]
        cell_sort_by_y_projection (bool): Whether to sort the matched OCR boxes by y-projection.

    Returns:
        dict: matched dict, key is table index, value is ocr index
    """
    matched = {}
    for i, ocr_box in enumerate(np.array(ocr_dt_boxes)):
        ocr_box = ocr_box.astype(np.float32)
        distances = []
        for j, table_box in enumerate(cell_box_list):
            distances.append(
                (distance(table_box, ocr_box), 1.0 - compute_iou(table_box, ocr_box))
            )  # compute iou and l1 distance
        sorted_distances = distances.copy()
        # select det box by iou and l1 distance
        sorted_distances = sorted(sorted_distances, key=lambda item: (item[1], item[0]))
        if distances.index(sorted_distances[0]) not in matched.keys():
            matched[distances.index(sorted_distances[0])] = [i]
        else:
            matched[distances.index(sorted_distances[0])].append(i)

    if cell_sort_by_y_projection:
        for cell_index in matched:
            input_boxes = [ocr_dt_boxes[i] for i in matched[cell_index]]
            sorted_indices = _sort_box_by_y_projection(input_boxes, 0.7)
            sorted_indices = [matched[cell_index][i] for i in sorted_indices]
            matched[cell_index] = sorted_indices

    return matched


def get_html_result(
    matched_index: dict, ocr_contents: dict, pred_structures: list
) -> str:
    """
    Generates HTML content based on the matched index, OCR contents, and predicted structures.

    Args:
        matched_index (dict): A dictionary containing matched indices.
        ocr_contents (dict): A dictionary of OCR contents.
        pred_structures (list): A list of predicted HTML structures.

    Returns:
        str: Generated HTML content as a string.
    """
    pred_html = []
    td_index = 0
    head_structure = pred_structures[0:3]
    html = "".join(head_structure)
    table_structure = pred_structures[3:-3]
    for tag in table_structure:
        if "</td>" in tag:
            if "<td></td>" == tag:
                pred_html.extend("<td>")
            if td_index in matched_index.keys():
                b_with = False
                if (
                    "<b>" in ocr_contents[matched_index[td_index][0]]
                    and len(matched_index[td_index]) > 1
                ):
                    b_with = True
                    pred_html.extend("<b>")
                for i, td_index_index in enumerate(matched_index[td_index]):
                    content = ocr_contents[td_index_index]
                    if len(matched_index[td_index]) > 1:
                        if len(content) == 0:
                            continue
                        if content[0] == " ":
                            content = content[1:]
                        if "<b>" in content:
                            content = content[3:]
                        if "</b>" in content:
                            content = content[:-4]
                        if len(content) == 0:
                            continue
                        if i != len(matched_index[td_index]) - 1 and " " != content[-1]:
                            content += " "
                    pred_html.extend(content)
                if b_with:
                    pred_html.extend("</b>")
            if "<td></td>" == tag:
                pred_html.append("</td>")
            else:
                pred_html.append(tag)
            td_index += 1
        else:
            pred_html.append(tag)
    html += "".join(pred_html)
    end_structure = pred_structures[-3:]
    html += "".join(end_structure)
    return html


def get_table_recognition_res(
    table_box: list,
    table_structure_pred: dict,
    overall_ocr_res: OCRResult,
    cells_texts_list: list,
    use_table_cells_ocr_results: bool,
    cell_sort_by_y_projection: bool = False,
) -> SingleTableRecognitionResult:
    """
    Retrieve table recognition result from cropped image info, table structure prediction, and overall OCR result.

    Args:
        table_box (list): Information about the location of cropped image, including the bounding box.
        table_structure_pred (dict): Predicted table structure.
        overall_ocr_res (OCRResult): Overall OCR result from the input image.
        cells_texts_list (list): OCR results with cells.
        use_table_cells_ocr_results (bool): whether to use OCR results with cells.
        cell_sort_by_y_projection (bool): Whether to sort the matched OCR boxes by y-projection.

    Returns:
        SingleTableRecognitionResult: An object containing the single table recognition result.
    """
    table_box = np.array([table_box])
    table_ocr_pred = get_sub_regions_ocr_res(overall_ocr_res, table_box)

    crop_start_point = [table_box[0][0], table_box[0][1]]
    img_shape = overall_ocr_res["doc_preprocessor_res"]["output_img"].shape[0:2]

    if len(table_structure_pred["bbox"]) == 0 or len(table_ocr_pred["rec_boxes"]) == 0:
        pred_html = " ".join(list(table_structure_pred["structure"]))
        if len(table_structure_pred["bbox"]) != 0:
            convert_table_structure_pred_bbox(
                table_structure_pred, crop_start_point, img_shape
            )
            table_cells_result = table_structure_pred["cell_box_list"]
        else:
            table_cells_result = []
        single_img_res = {
            "cell_box_list": table_cells_result,
            "table_ocr_pred": table_ocr_pred,
            "pred_html": pred_html,
        }
        return SingleTableRecognitionResult(single_img_res)

    convert_table_structure_pred_bbox(table_structure_pred, crop_start_point, img_shape)

    structures = table_structure_pred["structure"]
    cell_box_list = table_structure_pred["cell_box_list"]

    if use_table_cells_ocr_results == True:
        ocr_dt_boxes = cell_box_list
        ocr_texts_res = cells_texts_list
    else:
        ocr_dt_boxes = table_ocr_pred["rec_boxes"]
        ocr_texts_res = table_ocr_pred["rec_texts"]

    matched_index = match_table_and_ocr(
        cell_box_list, ocr_dt_boxes, cell_sort_by_y_projection=cell_sort_by_y_projection
    )
    pred_html = get_html_result(matched_index, ocr_texts_res, structures)

    single_img_res = {
        "cell_box_list": cell_box_list,
        "table_ocr_pred": table_ocr_pred,
        "pred_html": pred_html,
    }
    return SingleTableRecognitionResult(single_img_res)
