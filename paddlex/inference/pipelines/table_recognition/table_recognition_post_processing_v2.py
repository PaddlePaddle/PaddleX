# copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.
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
import math
from typing import Any, Dict, Optional
import numpy as np
from ..layout_parsing.utils import get_sub_regions_ocr_res
from ..components import convert_points_to_boxes
from .result import SingleTableRecognitionResult
from ..ocr.result import OCRResult


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
    cell_points_list: list, crop_start_point: list, img_shape: tuple
) -> None:
    """
    Convert the predicted table structure bounding boxes to the original image coordinate system.

    Args:
        cell_points_list (list):  Bounding boxes ('bbox').
        crop_start_point (list): A list of two integers representing the starting point (x, y) of the cropped image region.
        img_shape (tuple): A tuple of two integers representing the shape (height, width) of the original image.

    Returns:
        cell_points_list (list):  Bounding boxes ('bbox').
    """

    ori_cell_points_list = get_ori_image_coordinate(
        crop_start_point[0], crop_start_point[1], cell_points_list
    )
    ori_cell_points_list = np.reshape(ori_cell_points_list, (-1, 4, 2))
    cell_box_list = convert_points_to_boxes(ori_cell_points_list)

    img_height, img_width = img_shape
    cell_box_list = np.clip(
        cell_box_list, 0, [img_width, img_height, img_width, img_height]
    )
    return cell_box_list


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
    center1_x = (x1 + x2) / 2
    center1_y = (y1 + y2) / 2
    center2_x = (x3 + x4) / 2
    center2_y = (y3 + y4) / 2
    dis = math.sqrt((center2_x - center1_x) ** 2 + (center2_y - center1_y) ** 2)
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

def compute_inter(rec1, rec2):
    """
    computing intersection over rec2_area
    Args:
        rec1 (list): (x1, y1, x2, y2)
        rec2 (list): (x1, y1, x2, y2)
    Returns:
        float: Intersection over rec2_area
    """
    x1_1, y1_1, x2_1, y2_1 = rec1
    x1_2, y1_2, x2_2, y2_2 = rec2
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    inter_width = max(0, x_right - x_left)
    inter_height = max(0, y_bottom - y_top)
    inter_area = inter_width * inter_height
    rec2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    if rec2_area == 0:
        return 0 
    iou = inter_area / rec2_area
    return iou

def match_table_and_ocr(cell_box_list: list, ocr_dt_boxes: list) -> dict:
    """
    match table and ocr

    Args:
        cell_box_list (list): bbox for table cell, 2 points, [left, top, right, bottom]
        ocr_dt_boxes (list): bbox for ocr, 2 points, [left, top, right, bottom]

    Returns:
        dict: matched dict, key is table index, value is ocr index
    """
    matched = {}
    del_ocr = []
    for i, table_box in enumerate(cell_box_list):
        if len(table_box) == 8:
            table_box = [
                np.min(table_box[0::2]),
                np.min(table_box[1::2]),
                np.max(table_box[0::2]),
                np.max(table_box[1::2]),
            ]
        for j, ocr_box in enumerate(np.array(ocr_dt_boxes)):
            if compute_inter(table_box, ocr_box) > 0.8:
                if i not in matched.keys():
                    matched[i] = [j]
                else:
                    matched[i].append(j)
                del_ocr.append(j)
    miss_ocr = []
    miss_ocr_index = []
    for m in range(len(ocr_dt_boxes)):
        if m not in del_ocr:
            miss_ocr.append(ocr_dt_boxes[m])
            miss_ocr_index.append(m)
    if len(miss_ocr) != 0:
        for k, miss_ocr_box in enumerate(miss_ocr):
            distances = []
            for q, table_box in enumerate(cell_box_list):
                if len(table_box) == 0:
                    continue
                if len(table_box) == 8:
                    table_box = [
                        np.min(table_box[0::2]),
                        np.min(table_box[1::2]),
                        np.max(table_box[0::2]),
                        np.max(table_box[1::2]),
                    ]
                distances.append(
                    (distance(table_box, miss_ocr_box), 1.0 - compute_iou(table_box, miss_ocr_box))
                )  # compute iou and l1 distance
            sorted_distances = distances.copy()
            # select det box by iou and l1 distance
            sorted_distances = sorted(sorted_distances, key=lambda item: (item[1], item[0]))
            if distances.index(sorted_distances[0]) not in matched.keys():
                matched[distances.index(sorted_distances[0])] = [miss_ocr_index[k]]
            else:
                matched[distances.index(sorted_distances[0])].append(miss_ocr_index[k])
    # print(matched)
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
                if len(matched_index[td_index])==0:
                    continue
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

def sort_table_cells_boxes(boxes):
    """
    Sort the input list of bounding boxes.

    Args:
        boxes (list of lists): The input list of bounding boxes, where each bounding box is formatted as [x1, y1, x2, y2].

    Returns:
        sorted_boxes (list of lists): The list of bounding boxes sorted.
    """

    boxes_sorted_by_y = sorted(boxes, key=lambda box: box[1])
    rows = []
    current_row = []
    current_y = None
    tolerance = 10
    for box in boxes_sorted_by_y:
        x1, y1, x2, y2 = box
        if current_y is None:
            current_row.append(box)
            current_y = y1
        else:
            if abs(y1 - current_y) <= tolerance:
                current_row.append(box)
            else:
                current_row.sort(key=lambda x: x[0])
                rows.append(current_row)
                current_row = [box]
                current_y = y1
    if current_row:
        current_row.sort(key=lambda x: x[0])
        rows.append(current_row)
    sorted_boxes = [box for row in rows for box in row] 
    return sorted_boxes

def convert_to_four_point_coordinates(boxes):
    """
    Convert bounding boxes from [x1, y1, x2, y2] format to 
    [x1, y1, x2, y1, x2, y2, x1, y2] format.

    Parameters:
    - boxes: A list of bounding boxes, each defined as a list of integers 
             in the format [x1, y1, x2, y2].

    Returns:
    - A list of bounding boxes, each converted to the format 
      [x1, y1, x2, y1, x2, y2, x1, y2].
    """
    # Initialize an empty list to store the converted bounding boxes
    converted_boxes = []

    # Loop over each box in the input list
    for box in boxes:
        x1, y1, x2, y2 = box
        
        # Define the four corner points
        top_left = (x1, y1)
        top_right = (x2, y1)
        bottom_right = (x2, y2)
        bottom_left = (x1, y2)

        # Create a new list for the converted box
        converted_box = [
            top_left[0], top_left[1],  # Top-left corner
            top_right[0], top_right[1],  # Top-right corner
            bottom_right[0], bottom_right[1],  # Bottom-right corner
            bottom_left[0], bottom_left[1]   # Bottom-left corner
        ]

        # Append the converted box to the list
        converted_boxes.append(converted_box)

    return converted_boxes


def get_table_recognition_res(
    table_box: list,
    table_structure_result: list,
    table_cells_result: list,
    overall_ocr_res: OCRResult,
) -> SingleTableRecognitionResult:
    """
    Retrieve table recognition result from cropped image info, table structure prediction, and overall OCR result.

    Args:
        table_box (list): Information about the location of cropped image, including the bounding box.
        table_structure_result (list): Predicted table structure.
        table_cells_result (list): Predicted table cells.
        overall_ocr_res (OCRResult): Overall OCR result from the input image.

    Returns:
        SingleTableRecognitionResult: An object containing the single table recognition result.
    """

    table_cells_result = convert_to_four_point_coordinates(table_cells_result)

    table_box = np.array([table_box])
    table_ocr_pred = get_sub_regions_ocr_res(overall_ocr_res, table_box)

    crop_start_point = [table_box[0][0], table_box[0][1]]
    img_shape = overall_ocr_res["doc_preprocessor_res"]["output_img"].shape[0:2]

    table_cells_result = convert_table_structure_pred_bbox(
        table_cells_result, crop_start_point, img_shape
    )

    ocr_dt_boxes = table_ocr_pred["rec_boxes"]
    ocr_texts_res = table_ocr_pred["rec_texts"]

    table_cells_result = sort_table_cells_boxes(table_cells_result)

    matched_index = match_table_and_ocr(table_cells_result, ocr_dt_boxes)
    pred_html = get_html_result(matched_index, ocr_texts_res, table_structure_result)

    single_img_res = {
        "cell_box_list": table_cells_result,
        "table_ocr_pred": table_ocr_pred,
        "pred_html": pred_html,
    }

    return SingleTableRecognitionResult(single_img_res)
