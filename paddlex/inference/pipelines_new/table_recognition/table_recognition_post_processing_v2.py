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
    for i, ocr_box in enumerate(np.array(ocr_dt_boxes)):
        ocr_box = ocr_box.astype(np.float32)
        distances = []
        for j, table_box in enumerate(cell_box_list):
            if len(table_box) == 8:
                    table_box = [
                        np.min(table_box[0::2]),
                        np.min(table_box[1::2]),
                        np.max(table_box[0::2]),
                        np.max(table_box[1::2]),
                    ]
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


def sort_table_cells_boxes(boxes):
    '''
    Sort the input list of bounding boxes by using the DBSCAN algorithm to cluster based on the top-left y-coordinate (y1), and then sort within each line from left to right based on the top-left x-coordinate (x1).

    Args:
        boxes (list of lists): The input list of bounding boxes, where each bounding box is formatted as [x1, y1, x2, y2].

    Returns:
        sorted_boxes (list of lists): The list of bounding boxes sorted.
    '''
    import numpy as np
    from sklearn.cluster import DBSCAN

    # Extract the top-left y-coordinates (y1)
    y1_coords = np.array([box[1] for box in boxes])
    y1_coords = y1_coords.reshape(-1, 1)  # Convert to a 2D array

    # Choose an appropriate eps parameter based on the range of y-values
    y_range = y1_coords.max() - y1_coords.min()
    eps = y_range / 50  # Adjust the denominator as needed

    # Perform clustering using DBSCAN
    db = DBSCAN(eps=eps, min_samples=1).fit(y1_coords)
    labels = db.labels_

    # Group bounding boxes by their labels
    clusters = {}
    for label, box in zip(labels, boxes):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(box)

    # Sort rows based on y-coordinates
    # Compute the average y1 value for each row and sort from top to bottom
    sorted_rows = sorted(clusters.items(), key=lambda item: np.mean([box[1] for box in item[1]]))

    # Within each row, sort by x1 coordinate
    sorted_boxes = []
    for label, row in sorted_rows:
        row_sorted = sorted(row, key=lambda x: x[0])
        sorted_boxes.extend(row_sorted)

    return sorted_boxes


def get_table_recognition_res(
    table_box: list, table_structure_result: list, table_cells_result: list, overall_ocr_res: OCRResult
) -> SingleTableRecognitionResult:
    """
    Retrieve table recognition result from cropped image info, table structure prediction, and overall OCR result.

    Args:
        table_box (list): Information about the location of cropped image, including the bounding box.
        table_structure_pred (dict): Predicted table structure.
        overall_ocr_res (OCRResult): Overall OCR result from the input image.

    Returns:
        SingleTableRecognitionResult: An object containing the single table recognition result.
    """
    table_box = np.array([table_box])
    table_ocr_pred = get_sub_regions_ocr_res(overall_ocr_res, table_box)

    crop_start_point = [table_box[0][0], table_box[0][1]]
    img_shape = overall_ocr_res["doc_preprocessor_res"]["output_img"].shape[0:2]

    ocr_dt_boxes = table_ocr_pred["rec_boxes"]
    ocr_texts_res = table_ocr_pred["rec_texts"]

    table_cells_result = sort_table_cells_boxes(table_cells_result)
    ocr_dt_boxes = sort_table_cells_boxes(ocr_dt_boxes)

    matched_index = match_table_and_ocr(table_cells_result, ocr_dt_boxes)
    pred_html = get_html_result(matched_index, ocr_texts_res, table_structure_result)

    single_img_res = {
        "cell_box_list": table_cells_result,
        "table_ocr_pred": table_ocr_pred,
        "pred_html": pred_html,
    }

    return SingleTableRecognitionResult(single_img_res)
