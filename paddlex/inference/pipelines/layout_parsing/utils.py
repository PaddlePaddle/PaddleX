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

__all__ = [
    "get_sub_regions_ocr_res",
    "get_layout_ordering",
    "get_single_block_parsing_res",
    "recursive_img_array2path",
    "get_show_color",
    "sorted_layout_boxes",
]

import numpy as np
import copy
import cv2
from PIL import Image
import uuid
from pathlib import Path
from typing import Optional, Union, List, Tuple, Dict, Any
from ..ocr.result import OCRResult
from ...models.object_detection.result import DetResult
from ..components import convert_points_to_boxes


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


def _whether_y_overlap_exceeds_threshold(
    bbox1: Union[list, tuple],
    bbox2: Union[list, tuple],
    overlap_ratio_threshold: float = 0.6,
) -> bool:
    """
    Determines whether the vertical overlap between two bounding boxes exceeds a given threshold.

    Args:
        bbox1 (list or tuple): The first bounding box defined as (left, top, right, bottom).
        bbox2 (list or tuple): The second bounding box defined as (left, top, right, bottom).
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


def _adjust_span_text(span: List[str], prepend: bool = False, append: bool = False):
    """
    Adjust the text of a span by prepending or appending a newline.

    Args:
        span (list): A list where the second element is the text of the span.
        prepend (bool): If True, prepend a newline to the text.
        append (bool): If True, append a newline to the text.

    Returns:
        None: The function modifies the span in place.
    """
    if prepend:
        span[1] = "\n" + span[1]
    if append:
        span[1] = span[1] + "\n"


def _format_line(
    line: List[List[Union[List[int], str]]],
    layout_min: int,
    layout_max: int,
    is_reference: bool = False,
) -> None:
    """
    Format a line of text spans based on layout constraints.

    Args:
        line (list): A list of spans, where each span is a list containing a bounding box and text.
        layout_min (int): The minimum x-coordinate of the layout bounding box.
        layout_max (int): The maximum x-coordinate of the layout bounding box.
        is_reference (bool): A flag indicating whether the line is a reference line, which affects formatting rules.

    Returns:
        None: The function modifies the line in place.
    """
    first_span = line[0]
    end_span = line[-1]

    if not is_reference:
        if first_span[0][0] - layout_min > 10:
            _adjust_span_text(first_span, prepend=True)
        if layout_max - end_span[0][2] > 10:
            _adjust_span_text(end_span, append=True)
    else:
        if first_span[0][0] - layout_min < 5:
            _adjust_span_text(first_span, prepend=True)
        if layout_max - end_span[0][2] > 20:
            _adjust_span_text(end_span, append=True)


def _sort_ocr_res_by_y_projection(
    label: Any,
    layout_bbox: Tuple[int, int, int, int],
    ocr_res: Dict[str, List[Any]],
    line_height_iou_threshold: float = 0.7,
) -> Dict[str, List[Any]]:
    """
    Sorts OCR results based on their spatial arrangement, grouping them into lines and blocks.

    Args:
        label (Any): The label associated with the OCR results. It's not used in the function but might be
                     relevant for other parts of the calling context.
        layout_bbox (Tuple[int, int, int, int]): A tuple representing the layout bounding box, defined as
                                                 (left, top, right, bottom).
        ocr_res (Dict[str, List[Any]]): A dictionary containing OCR results with the following keys:
            - "boxes": A list of bounding boxes, each defined as [left, top, right, bottom].
            - "rec_texts": A corresponding list of recognized text strings for each box.
        line_height_iou_threshold (float): The threshold for determining whether two boxes belong to
                                           the same line based on their vertical overlap. Defaults to 0.7.

    Returns:
        Dict[str, List[Any]]: A dictionary with the same structure as `ocr_res`, but with boxes and texts sorted
                              and grouped into lines and blocks.
    """
    assert (
        ocr_res["boxes"] and ocr_res["rec_texts"]
    ), "OCR results must contain 'boxes' and 'rec_texts'"

    boxes = ocr_res["boxes"]
    rec_texts = ocr_res["rec_texts"]

    x_min, _, x_max, _ = layout_bbox
    inline_x_min = min([box[0] for box in boxes])
    inline_x_max = max([box[2] for box in boxes])

    spans = list(zip(boxes, rec_texts))

    spans.sort(key=lambda span: span[0][1])
    spans = [list(span) for span in spans]

    lines = []
    current_line = [spans[0]]
    current_y0, current_y1 = spans[0][0][1], spans[0][0][3]

    for span in spans[1:]:
        y0, y1 = span[0][1], span[0][3]
        if _whether_y_overlap_exceeds_threshold(
            (0, current_y0, 0, current_y1),
            (0, y0, 0, y1),
            line_height_iou_threshold,
        ):
            current_line.append(span)
            current_y0 = min(current_y0, y0)
            current_y1 = max(current_y1, y1)
        else:
            lines.append(current_line)
            current_line = [span]
            current_y0, current_y1 = y0, y1

    if current_line:
        lines.append(current_line)

    for line in lines:
        line.sort(key=lambda span: span[0][0])
        if label == "reference":
            line = _format_line(line, inline_x_min, inline_x_max, is_reference=True)
        else:
            line = _format_line(line, x_min, x_max)

    # Flatten lines back into a single list for boxes and texts
    ocr_res["boxes"] = [span[0] for line in lines for span in line]
    ocr_res["rec_texts"] = [span[1] + " " for line in lines for span in line]

    return ocr_res


def get_single_block_parsing_res(
    overall_ocr_res: OCRResult,
    layout_det_res: DetResult,
    table_res_list: list,
    seal_res_list: list,
) -> OCRResult:
    """
    Extract structured information from OCR and layout detection results.

    Args:
        overall_ocr_res (OCRResult): An object containing the overall OCR results, including detected text boxes and recognized text. The structure is expected to have:
            - "input_img": The image on which OCR was performed.
            - "dt_boxes": A list of detected text box coordinates.
            - "rec_texts": A list of recognized text corresponding to the detected boxes.

        layout_det_res (DetResult): An object containing the layout detection results, including detected layout boxes and their labels. The structure is expected to have:
            - "boxes": A list of dictionaries with keys "coordinate" for box coordinates and "label" for the type of content.

        table_res_list (list): A list of table detection results, where each item is a dictionary containing:
            - "layout_bbox": The bounding box of the table layout.
            - "pred_html": The predicted HTML representation of the table.

        seal_res_list (List): A list of seal detection results. The details of each item depend on the specific application context.

    Returns:
        list: A list of structured boxes where each item is a dictionary containing:
            - "label": The label of the content (e.g., 'table', 'chart', 'image').
            - The label as a key with either table HTML or image data and text.
            - "layout_bbox": The coordinates of the layout box.
    """

    single_block_layout_parsing_res = []
    input_img = overall_ocr_res["doc_preprocessor_res"]["output_img"]

    for box_info in layout_det_res["boxes"]:
        layout_bbox = box_info["coordinate"]
        label = box_info["label"]
        rec_res = {"boxes": [], "rec_texts": [], "flag": False}
        seg_start_flag = True
        seg_end_flag = True

        if label == "table":
            for i, table_res in enumerate(table_res_list):
                if (
                    _calculate_overlap_area_div_minbox_area_ratio(
                        layout_bbox, table_res["cell_box_list"][0]
                    )
                    > 0.5
                ):
                    single_block_layout_parsing_res.append(
                        {
                            "label": label,
                            f"{label}": table_res["pred_html"],
                            "layout_bbox": layout_bbox,
                            "seg_start_flag": seg_start_flag,
                            "seg_end_flag": seg_end_flag,
                        },
                    )
                    del table_res_list[i]
                    break
        else:
            overall_text_boxes = overall_ocr_res["rec_boxes"]
            for box_no in range(len(overall_text_boxes)):
                if (
                    _calculate_overlap_area_div_minbox_area_ratio(
                        layout_bbox, overall_text_boxes[box_no]
                    )
                    > 0.5
                ):
                    rec_res["boxes"].append(overall_text_boxes[box_no])
                    rec_res["rec_texts"].append(
                        overall_ocr_res["rec_texts"][box_no],
                    )
                    rec_res["flag"] = True

            if rec_res["flag"]:
                rec_res = _sort_ocr_res_by_y_projection(
                    label, layout_bbox, rec_res, 0.7
                )
                rec_res_first_bbox = rec_res["boxes"][0]
                rec_res_end_bbox = rec_res["boxes"][-1]
                if rec_res_first_bbox[0] - layout_bbox[0] < 10:
                    seg_start_flag = False
                if layout_bbox[2] - rec_res_end_bbox[2] < 10:
                    seg_end_flag = False
                if label == "formula":
                    rec_res["rec_texts"] = [
                        rec_res_text.replace("$", "")
                        for rec_res_text in rec_res["rec_texts"]
                    ]

            if label in ["chart", "image", "seal"]:
                single_block_layout_parsing_res.append(
                    {
                        "label": label,
                        f"{label}": {
                            "img": input_img[
                                int(layout_bbox[1]) : int(layout_bbox[3]),
                                int(layout_bbox[0]) : int(layout_bbox[2]),
                            ],
                        },
                        "layout_bbox": layout_bbox,
                        "seg_start_flag": seg_start_flag,
                        "seg_end_flag": seg_end_flag,
                    },
                )
            else:
                single_block_layout_parsing_res.append(
                    {
                        "label": label,
                        f"{label}": "".join(rec_res["rec_texts"]),
                        "layout_bbox": layout_bbox,
                        "seg_start_flag": seg_start_flag,
                        "seg_end_flag": seg_end_flag,
                    },
                )

    return single_block_layout_parsing_res


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


def _recursive_yx_cut(
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
            _recursive_yx_cut(
                x_sorted_boxes_chunk[x_interval_indices],
                x_sorted_indices_chunk[x_interval_indices],
                res,
            )


def _recursive_xy_cut(
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
            _recursive_xy_cut(
                y_sorted_boxes_chunk[y_interval_indices],
                y_sorted_indices_chunk[y_interval_indices],
                res,
            )


def sort_by_xycut(
    block_bboxes: Union[np.ndarray, List[List[int]]],
    direction: int = 0,
    min_gap: int = 1,
) -> List[int]:
    """
    Sort bounding boxes using recursive XY cut method based on the specified direction.

    Args:
        block_bboxes (Union[np.ndarray, List[List[int]]]): An array or list of bounding boxes,
                                                           where each box is represented as
                                                           [x_min, y_min, x_max, y_max].
        direction (int): Direction for the initial cut. Use 1 for Y-axis first and 0 for X-axis first.
                         Defaults to 0.
        min_gap (int): Minimum gap width to consider a separation between segments. Defaults to 1.

    Returns:
        List[int]: A list of indices representing the order of sorted bounding boxes.
    """
    block_bboxes = np.asarray(block_bboxes).astype(int)
    res = []

    if direction == 1:
        _recursive_yx_cut(
            block_bboxes,
            np.arange(len(block_bboxes)),
            res,
            min_gap,
        )
    else:
        _recursive_xy_cut(
            block_bboxes,
            np.arange(len(block_bboxes)),
            res,
            min_gap,
        )

    return res


def _img_array2path(data: np.ndarray) -> str:
    """
    Save an image array to disk and return the relative file path.

    Args:
        data (np.ndarray): An image represented as a numpy array with 3 dimensions (H, W, C).

    Returns:
        dict: A dictionary with a single key-value pair formatted as:
              {"imgs/image_{uuid4_hex}.png": PIL.Image.Image}

    Raises:
        ValueError: If the input data is not a valid image array.
    """
    if isinstance(data, np.ndarray) and data.ndim == 3:
        # Generate a unique filename using UUID
        img_name = f"image_{uuid.uuid4().hex}.png"

        return {f"imgs/{img_name}": Image.fromarray(data)}
    else:
        raise ValueError(
            "Input data must be a 3-dimensional numpy array representing an image."
        )


def recursive_img_array2path(
    data: Union[Dict[str, Any], List[Any]],
    labels: List[str] = [],
) -> None:
    """
    Recursively process a dictionary or list to save image arrays to disk
    and replace them with file paths.

    Args:
        data (Union[Dict[str, Any], List[Any]]): The data structure that may contain image arrays.
        save_path (Union[str, Path]): The base path where images should be saved.
        labels (List[str]): List of keys to check for image arrays in dictionaries.

    Returns:
        None: This function modifies the input data structure in place.
    """
    if isinstance(data, dict):
        for k, v in data.items():
            if k in labels and isinstance(v, np.ndarray) and v.ndim == 3:
                data[k] = _img_array2path(v)
            else:
                recursive_img_array2path(v, labels)
    elif isinstance(data, list):
        for item in data:
            recursive_img_array2path(item, labels)


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


def _remove_overlap_blocks(
    blocks: List[Dict[str, List[int]]], threshold: float = 0.65, smaller: bool = True
) -> Tuple[List[Dict[str, List[int]]], List[Dict[str, List[int]]]]:
    """
    Remove overlapping blocks based on a specified overlap ratio threshold.

    Args:
        blocks (List[Dict[str, List[int]]]): List of block dictionaries, each containing a 'layout_bbox' key.
        threshold (float): Ratio threshold to determine significant overlap.
        smaller (bool): If True, the smaller block in overlap is removed.

    Returns:
        Tuple[List[Dict[str, List[int]]], List[Dict[str, List[int]]]]:
            A tuple containing the updated list of blocks and a list of dropped blocks.
    """
    dropped_blocks = []
    dropped_indexes = set()

    # Iterate over each pair of blocks to find overlaps
    for i, block1 in enumerate(blocks):
        for j in range(i + 1, len(blocks)):
            block2 = blocks[j]
            # Skip blocks that are already marked for removal
            if i in dropped_indexes or j in dropped_indexes:
                continue
            # Check for overlap and determine which block to remove
            overlap_box_index = _get_minbox_if_overlap_by_ratio(
                block1["layout_bbox"],
                block2["layout_bbox"],
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
        dropped_blocks.append(blocks[index])
        del blocks[index]

    return blocks, dropped_blocks


def _get_text_median_width(blocks: List[Dict[str, any]]) -> float:
    """
    Calculate the median width of blocks labeled as "text".

    Args:
        blocks (List[Dict[str, any]]): List of block dictionaries, each containing a 'layout_bbox' and 'label'.

    Returns:
        float: The median width of text blocks, or infinity if no text blocks are found.
    """
    widths = [
        block["layout_bbox"][2] - block["layout_bbox"][0]
        for block in blocks
        if block.get("label") == "text"
    ]
    return np.median(widths) if widths else float("inf")


def _get_layout_property(
    blocks: List[Dict[str, any]],
    median_width: float,
    no_mask_labels: List[str],
    threshold: float = 0.8,
) -> Tuple[List[Dict[str, any]], bool]:
    """
    Determine the layout (single or double column) of text blocks.

    Args:
        blocks (List[Dict[str, any]]): List of block dictionaries containing 'label' and 'layout_bbox'.
        median_width (float): Median width of text blocks.
        no_mask_labels (List[str]): Labels of blocks to be considered for layout analysis.
        threshold (float): Threshold for determining layout overlap.

    Returns:
        Tuple[List[Dict[str, any]], bool]: Updated list of blocks with layout information and a boolean
        indicating if the double layout area is greater than the single layout area.
    """
    blocks.sort(
        key=lambda x: (
            x["layout_bbox"][0],
            (x["layout_bbox"][2] - x["layout_bbox"][0]),
        ),
    )
    check_single_layout = {}
    page_min_x, page_max_x = float("inf"), 0
    double_label_area = 0
    single_label_area = 0

    for i, block in enumerate(blocks):
        page_min_x = min(page_min_x, block["layout_bbox"][0])
        page_max_x = max(page_max_x, block["layout_bbox"][2])
    page_width = page_max_x - page_min_x

    for i, block in enumerate(blocks):
        if block["label"] not in no_mask_labels:
            continue

        x_min_i, _, x_max_i, _ = block["layout_bbox"]
        layout_length = x_max_i - x_min_i
        cover_count, cover_with_threshold_count = 0, 0
        match_block_with_threshold_indexes = []

        for j, other_block in enumerate(blocks):
            if i == j or other_block["label"] not in no_mask_labels:
                continue

            x_min_j, _, x_max_j, _ = other_block["layout_bbox"]
            x_match_min, x_match_max = max(
                x_min_i,
                x_min_j,
            ), min(x_max_i, x_max_j)
            match_block_iou = (x_match_max - x_match_min) / (x_max_j - x_min_j)

            if match_block_iou > 0:
                cover_count += 1
                if match_block_iou > threshold:
                    cover_with_threshold_count += 1
                    match_block_with_threshold_indexes.append(
                        (j, match_block_iou),
                    )
                x_min_i = x_match_max
                if x_min_i >= x_max_i:
                    break

        if (
            layout_length > median_width * 1.3
            and (cover_with_threshold_count >= 2 or cover_count >= 2)
        ) or layout_length > 0.6 * page_width:
            # if layout_length > median_width * 1.3 and (cover_with_threshold_count >= 2):
            block["layout"] = "double"
            double_label_area += (block["layout_bbox"][2] - block["layout_bbox"][0]) * (
                block["layout_bbox"][3] - block["layout_bbox"][1]
            )
        else:
            block["layout"] = "single"
            check_single_layout[i] = match_block_with_threshold_indexes

    # Check single-layout block
    for i, single_layout in check_single_layout.items():
        if single_layout:
            index, match_iou = single_layout[-1]
            if match_iou > 0.9 and blocks[index]["layout"] == "double":
                blocks[i]["layout"] = "double"
                double_label_area += (
                    blocks[i]["layout_bbox"][2] - blocks[i]["layout_bbox"][0]
                ) * (blocks[i]["layout_bbox"][3] - blocks[i]["layout_bbox"][1])
            else:
                single_label_area += (
                    blocks[i]["layout_bbox"][2] - blocks[i]["layout_bbox"][0]
                ) * (blocks[i]["layout_bbox"][3] - blocks[i]["layout_bbox"][1])

    return blocks, (double_label_area > single_label_area)


def _get_bbox_direction(input_bbox: List[float], ratio: float = 1.0) -> bool:
    """
    Determine if a bounding box is horizontal or vertical.

    Args:
        input_bbox (List[float]): Bounding box [x_min, y_min, x_max, y_max].
        ratio (float): Ratio for determining orientation. Default is 1.0.

    Returns:
        bool: True if the bounding box is considered horizontal, False if vertical.
    """
    width = input_bbox[2] - input_bbox[0]
    height = input_bbox[3] - input_bbox[1]
    return width * ratio >= height


def _get_projection_iou(
    input_bbox: List[float], match_bbox: List[float], is_horizontal: bool = True
) -> float:
    """
    Calculate the IoU of lines between two bounding boxes.

    Args:
        input_bbox (List[float]): First bounding box [x_min, y_min, x_max, y_max].
        match_bbox (List[float]): Second bounding box [x_min, y_min, x_max, y_max].
        is_horizontal (bool): Whether to compare horizontally or vertically.

    Returns:
        float: Line IoU. Returns 0 if there is no overlap.
    """
    if is_horizontal:
        x_match_min = max(input_bbox[0], match_bbox[0])
        x_match_max = min(input_bbox[2], match_bbox[2])
        overlap = max(0, x_match_max - x_match_min)
        input_width = input_bbox[2] - input_bbox[0]
    else:
        y_match_min = max(input_bbox[1], match_bbox[1])
        y_match_max = min(input_bbox[3], match_bbox[3])
        overlap = max(0, y_match_max - y_match_min)
        input_width = input_bbox[3] - input_bbox[1]

    return overlap / input_width if input_width > 0 else 0.0


def _get_sub_category(
    blocks: List[Dict[str, Any]], title_labels: List[str]
) -> List[Dict[str, Any]]:
    """
    Determine the layout of title and text blocks.

    Args:
        blocks (List[Dict[str, Any]]): List of block dictionaries.
        title_labels (List[str]): List of labels considered as titles.

    Returns:
        List[Dict[str, Any]]: Updated list of blocks with title-text layout information.
    """

    sub_title_labels = ["paragraph_title"]
    vision_labels = ["image", "table", "chart", "figure"]

    for i, block1 in enumerate(blocks):
        block1.setdefault("title_text", [])
        block1.setdefault("sub_title", [])
        block1.setdefault("vision_footnote", [])
        block1.setdefault("sub_label", block1["label"])

        if (
            block1["label"] not in title_labels
            and block1["label"] not in sub_title_labels
            and block1["label"] not in vision_labels
        ):
            continue

        bbox1 = block1["layout_bbox"]
        x1, y1, x2, y2 = bbox1
        is_horizontal_1 = _get_bbox_direction(block1["layout_bbox"])
        left_up_title_text_distance = float("inf")
        left_up_title_text_index = -1
        left_up_title_text_direction = None
        right_down_title_text_distance = float("inf")
        right_down_title_text_index = -1
        right_down_title_text_direction = None

        for j, block2 in enumerate(blocks):
            if i == j:
                continue

            bbox2 = block2["layout_bbox"]
            x1_prime, y1_prime, x2_prime, y2_prime = bbox2
            is_horizontal_2 = _get_bbox_direction(bbox2)
            match_block_iou = _get_projection_iou(
                bbox2,
                bbox1,
                is_horizontal_1,
            )

            def distance_(is_horizontal, is_left_up):
                if is_horizontal:
                    if is_left_up:
                        return (y1 - y2_prime + 2) // 5 + x1_prime / 5000
                    else:
                        return (y1_prime - y2 + 2) // 5 + x1_prime / 5000

                else:
                    if is_left_up:
                        return (x1 - x2_prime + 2) // 5 + y1_prime / 5000
                    else:
                        return (x1_prime - x2 + 2) // 5 + y1_prime / 5000

            block_iou_threshold = 0.1
            if block1["label"] in sub_title_labels:
                match_block_iou = _calculate_overlap_area_div_minbox_area_ratio(
                    bbox2,
                    bbox1,
                )
                block_iou_threshold = 0.7

            if is_horizontal_1:
                if match_block_iou >= block_iou_threshold:
                    left_up_distance = distance_(True, True)
                    right_down_distance = distance_(True, False)
                    if (
                        y2_prime <= y1
                        and left_up_distance <= left_up_title_text_distance
                    ):
                        left_up_title_text_distance = left_up_distance
                        left_up_title_text_index = j
                        left_up_title_text_direction = is_horizontal_2
                    elif (
                        y1_prime > y2
                        and right_down_distance < right_down_title_text_distance
                    ):
                        right_down_title_text_distance = right_down_distance
                        right_down_title_text_index = j
                        right_down_title_text_direction = is_horizontal_2
            else:
                if match_block_iou >= block_iou_threshold:
                    left_up_distance = distance_(False, True)
                    right_down_distance = distance_(False, False)
                    if (
                        x2_prime <= x1
                        and left_up_distance <= left_up_title_text_distance
                    ):
                        left_up_title_text_distance = left_up_distance
                        left_up_title_text_index = j
                        left_up_title_text_direction = is_horizontal_2
                    elif (
                        x1_prime > x2
                        and right_down_distance < right_down_title_text_distance
                    ):
                        right_down_title_text_distance = right_down_distance
                        right_down_title_text_index = j
                        right_down_title_text_direction = is_horizontal_2

        height = bbox1[3] - bbox1[1]
        width = bbox1[2] - bbox1[0]
        title_text_weight = [0.8, 0.8]

        title_text, sub_title, vision_footnote = [], [], []

        def get_sub_category_(
            title_text_direction,
            title_text_index,
            label,
            is_left_up=True,
        ):
            direction_ = [1, 3] if is_left_up else [2, 4]
            if (
                title_text_direction == is_horizontal_1
                and title_text_index != -1
                and (label == "text" or label == "paragraph_title")
            ):
                bbox2 = blocks[title_text_index]["layout_bbox"]
                if is_horizontal_1:
                    height1 = bbox2[3] - bbox2[1]
                    width1 = bbox2[2] - bbox2[0]
                    if label == "text":
                        if (
                            _nearest_edge_distance(bbox1, bbox2)[0] <= 15
                            and block1["label"] in vision_labels
                            and width1 < width
                            and height1 < 0.5 * height
                        ):
                            blocks[title_text_index]["sub_label"] = "vision_footnote"
                            vision_footnote.append(bbox2)
                        elif (
                            height1 < height * title_text_weight[0]
                            and (width1 < width or width1 > 1.5 * width)
                            and block1["label"] in title_labels
                        ):
                            blocks[title_text_index]["sub_label"] = "title_text"
                            title_text.append((direction_[0], bbox2))
                    elif (
                        label == "paragraph_title"
                        and block1["label"] in sub_title_labels
                    ):
                        sub_title.append(bbox2)
                else:
                    height1 = bbox2[3] - bbox2[1]
                    width1 = bbox2[2] - bbox2[0]
                    if label == "text":
                        if (
                            _nearest_edge_distance(bbox1, bbox2)[0] <= 15
                            and block1["label"] in vision_labels
                            and height1 < height
                            and width1 < 0.5 * width
                        ):
                            blocks[title_text_index]["sub_label"] = "vision_footnote"
                            vision_footnote.append(bbox2)
                        elif (
                            width1 < width * title_text_weight[1]
                            and block1["label"] in title_labels
                        ):
                            blocks[title_text_index]["sub_label"] = "title_text"
                            title_text.append((direction_[1], bbox2))
                    elif (
                        label == "paragraph_title"
                        and block1["label"] in sub_title_labels
                    ):
                        sub_title.append(bbox2)

        if (
            is_horizontal_1
            and abs(left_up_title_text_distance - right_down_title_text_distance) * 5
            > height
        ) or (
            not is_horizontal_1
            and abs(left_up_title_text_distance - right_down_title_text_distance) * 5
            > width
        ):
            if left_up_title_text_distance < right_down_title_text_distance:
                get_sub_category_(
                    left_up_title_text_direction,
                    left_up_title_text_index,
                    blocks[left_up_title_text_index]["label"],
                    True,
                )
            else:
                get_sub_category_(
                    right_down_title_text_direction,
                    right_down_title_text_index,
                    blocks[right_down_title_text_index]["label"],
                    False,
                )
        else:
            get_sub_category_(
                left_up_title_text_direction,
                left_up_title_text_index,
                blocks[left_up_title_text_index]["label"],
                True,
            )
            get_sub_category_(
                right_down_title_text_direction,
                right_down_title_text_index,
                blocks[right_down_title_text_index]["label"],
                False,
            )

        if block1["label"] in title_labels:
            if blocks[i].get("title_text") == []:
                blocks[i]["title_text"] = title_text

        if block1["label"] in sub_title_labels:
            if blocks[i].get("sub_title") == []:
                blocks[i]["sub_title"] = sub_title

        if block1["label"] in vision_labels:
            if blocks[i].get("vision_footnote") == []:
                blocks[i]["vision_footnote"] = vision_footnote

    return blocks


def get_layout_ordering(
    data: List[Dict[str, Any]],
    no_mask_labels: List[str] = [],
    already_sorted: bool = False,
) -> None:
    """
    Process layout parsing results to remove overlapping bounding boxes
    and assign an ordering index based on their positions.

    Modifies:
        The 'data' list by adding an 'index' to each block.

    Args:
        data (List[Dict[str, Any]]): List of block dictionaries with 'layout_bbox' and 'label'.
        no_mask_labels (List[str]): Labels for which overlapping removal is not performed.
        already_sorted (bool): Assumes data is already sorted by position if True.
    """
    if already_sorted:
        return data

    title_text_labels = ["doc_title"]
    title_labels = ["doc_title", "paragraph_title"]
    vision_labels = ["image", "table", "seal", "chart", "figure"]
    vision_title_labels = ["table_title", "chart_title", "figure_title"]

    parsing_result = data["sub_blocks"]
    parsing_result, _ = _remove_overlap_blocks(
        parsing_result,
        threshold=0.5,
        smaller=True,
    )
    parsing_result = _get_sub_category(parsing_result, title_text_labels)

    doc_flag = False
    median_width = _get_text_median_width(parsing_result)
    parsing_result, projection_direction = _get_layout_property(
        parsing_result,
        median_width,
        no_mask_labels=no_mask_labels,
        threshold=0.3,
    )
    # Convert bounding boxes to float and remove overlaps
    (
        double_text_blocks,
        title_text_blocks,
        title_blocks,
        vision_blocks,
        vision_title_blocks,
        vision_footnote_blocks,
        other_blocks,
    ) = ([], [], [], [], [], [], [])

    drop_indexes = []

    for index, block in enumerate(parsing_result):
        label = block["sub_label"]
        block["layout_bbox"] = list(map(int, block["layout_bbox"]))

        if label == "doc_title":
            doc_flag = True

        if label in no_mask_labels:
            if block["layout"] == "double":
                double_text_blocks.append(block)
                drop_indexes.append(index)
        elif label == "title_text":
            title_text_blocks.append(block)
            drop_indexes.append(index)
        elif label == "vision_footnote":
            vision_footnote_blocks.append(block)
            drop_indexes.append(index)
        elif label in vision_title_labels:
            vision_title_blocks.append(block)
            drop_indexes.append(index)
        elif label in title_labels:
            title_blocks.append(block)
            drop_indexes.append(index)
        elif label in vision_labels:
            vision_blocks.append(block)
            drop_indexes.append(index)
        else:
            other_blocks.append(block)
            drop_indexes.append(index)

    for index in sorted(drop_indexes, reverse=True):
        del parsing_result[index]

    if len(parsing_result) > 0:
        # single text label
        if len(double_text_blocks) > len(parsing_result) or projection_direction:
            parsing_result.extend(title_blocks + double_text_blocks)
            title_blocks = []
            double_text_blocks = []
            block_bboxes = [block["layout_bbox"] for block in parsing_result]
            block_bboxes.sort(
                key=lambda x: (
                    x[0] // max(20, median_width),
                    x[1],
                ),
            )
            block_bboxes = np.array(block_bboxes)
            sorted_indices = sort_by_xycut(
                block_bboxes,
                direction=1,
                min_gap=1,
            )
        else:
            block_bboxes = [block["layout_bbox"] for block in parsing_result]
            block_bboxes.sort(key=lambda x: (x[0] // 20, x[1]))
            block_bboxes = np.array(block_bboxes)
            sorted_indices = sort_by_xycut(
                block_bboxes,
                direction=0,
                min_gap=20,
            )

        sorted_boxes = block_bboxes[sorted_indices].tolist()

        for block in parsing_result:
            block["index"] = sorted_boxes.index(block["layout_bbox"]) + 1
            block["sub_index"] = sorted_boxes.index(block["layout_bbox"]) + 1

    def nearest_match_(input_blocks, distance_type="manhattan", is_add_index=True):
        for block in input_blocks:
            bbox = block["layout_bbox"]
            min_distance = float("inf")
            min_distance_config = [
                [float("inf"), float("inf")],
                float("inf"),
                float("inf"),
            ]  # for double text
            nearest_gt_index = 0
            for match_block in parsing_result:
                match_bbox = match_block["layout_bbox"]
                if distance_type == "nearest_iou_edge_distance":
                    distance, min_distance_config = _nearest_iou_edge_distance(
                        bbox,
                        match_bbox,
                        block["sub_label"],
                        vision_labels=vision_labels,
                        no_mask_labels=no_mask_labels,
                        median_width=median_width,
                        title_labels=title_labels,
                        title_text=block["title_text"],
                        sub_title=block["sub_title"],
                        min_distance_config=min_distance_config,
                        tolerance_len=10,
                    )
                elif distance_type == "title_text":
                    if (
                        match_block["label"] in title_labels + ["abstract"]
                        and match_block["title_text"] != []
                    ):
                        iou_left_up = _calculate_overlap_area_div_minbox_area_ratio(
                            bbox,
                            match_block["title_text"][0][1],
                        )
                        iou_right_down = _calculate_overlap_area_div_minbox_area_ratio(
                            bbox,
                            match_block["title_text"][-1][1],
                        )
                        iou = 1 - max(iou_left_up, iou_right_down)
                        distance = _manhattan_distance(bbox, match_bbox) * iou
                    else:
                        distance = float("inf")
                elif distance_type == "manhattan":
                    distance = _manhattan_distance(bbox, match_bbox)
                elif distance_type == "vision_footnote":
                    if (
                        match_block["label"] in vision_labels
                        and match_block["vision_footnote"] != []
                    ):
                        iou_left_up = _calculate_overlap_area_div_minbox_area_ratio(
                            bbox,
                            match_block["vision_footnote"][0],
                        )
                        iou_right_down = _calculate_overlap_area_div_minbox_area_ratio(
                            bbox,
                            match_block["vision_footnote"][-1],
                        )
                        iou = 1 - max(iou_left_up, iou_right_down)
                        distance = _manhattan_distance(bbox, match_bbox) * iou
                    else:
                        distance = float("inf")
                elif distance_type == "vision_body":
                    if (
                        match_block["label"] in vision_title_labels
                        and block["vision_footnote"] != []
                    ):
                        iou_left_up = _calculate_overlap_area_div_minbox_area_ratio(
                            match_bbox,
                            block["vision_footnote"][0],
                        )
                        iou_right_down = _calculate_overlap_area_div_minbox_area_ratio(
                            match_bbox,
                            block["vision_footnote"][-1],
                        )
                        iou = 1 - max(iou_left_up, iou_right_down)
                        distance = _manhattan_distance(bbox, match_bbox) * iou
                    else:
                        distance = float("inf")
                else:
                    raise NotImplementedError

                if distance < min_distance:
                    min_distance = distance
                    if is_add_index:
                        nearest_gt_index = match_block.get("index", 999)
                    else:
                        nearest_gt_index = match_block.get("sub_index", 999)

            if is_add_index:
                block["index"] = nearest_gt_index
            else:
                block["sub_index"] = nearest_gt_index

            parsing_result.append(block)

    # double text label
    double_text_blocks.sort(
        key=lambda x: (
            x["layout_bbox"][1] // 10,
            x["layout_bbox"][0] // median_width,
            x["layout_bbox"][1] ** 2 + x["layout_bbox"][0] ** 2,
        ),
    )
    nearest_match_(
        double_text_blocks,
        distance_type="nearest_iou_edge_distance",
    )
    parsing_result.sort(
        key=lambda x: (x["index"], x["layout_bbox"][1], x["layout_bbox"][0]),
    )

    for idx, block in enumerate(parsing_result):
        block["index"] = idx + 1
        block["sub_index"] = idx + 1

    # title label
    title_blocks.sort(
        key=lambda x: (
            x["layout_bbox"][1] // 10,
            x["layout_bbox"][0] // median_width,
            x["layout_bbox"][1] ** 2 + x["layout_bbox"][0] ** 2,
        ),
    )
    nearest_match_(title_blocks, distance_type="nearest_iou_edge_distance")

    if doc_flag:
        text_sort_labels = ["doc_title"]
        text_label_priority = {
            label: priority for priority, label in enumerate(text_sort_labels)
        }
        doc_titles = []
        for i, block in enumerate(parsing_result):
            if block["label"] == "doc_title":
                doc_titles.append(
                    (i, block["layout_bbox"][1], block["layout_bbox"][0]),
                )
        doc_titles.sort(key=lambda x: (x[1], x[2]))
        first_doc_title_index = doc_titles[0][0]
        parsing_result[first_doc_title_index]["index"] = 1
        parsing_result.sort(
            key=lambda x: (
                x["index"],
                text_label_priority.get(x["label"], 9999),
                x["layout_bbox"][1],
                x["layout_bbox"][0],
            ),
        )
    else:
        parsing_result.sort(
            key=lambda x: (
                x["index"],
                x["layout_bbox"][1],
                x["layout_bbox"][0],
            ),
        )

    for idx, block in enumerate(parsing_result):
        block["index"] = idx + 1
        block["sub_index"] = idx + 1

    # title-text label
    nearest_match_(title_text_blocks, distance_type="title_text")
    text_sort_labels = ["doc_title", "paragraph_title", "title_text"]
    text_label_priority = {
        label: priority for priority, label in enumerate(text_sort_labels)
    }
    parsing_result.sort(
        key=lambda x: (
            x["index"],
            text_label_priority.get(x["sub_label"], 9999),
            x["layout_bbox"][1],
            x["layout_bbox"][0],
        ),
    )

    for idx, block in enumerate(parsing_result):
        block["index"] = idx + 1
        block["sub_index"] = idx + 1

    # image,figure,chart,seal label
    nearest_match_(
        vision_title_blocks,
        distance_type="nearest_iou_edge_distance",
        is_add_index=False,
    )
    parsing_result.sort(
        key=lambda x: (
            x["sub_index"],
            x["layout_bbox"][1],
            x["layout_bbox"][0],
        ),
    )

    for idx, block in enumerate(parsing_result):
        block["sub_index"] = idx + 1

    # image,figure,chart,seal label
    nearest_match_(
        vision_blocks,
        distance_type="nearest_iou_edge_distance",
        is_add_index=False,
    )
    parsing_result.sort(
        key=lambda x: (
            x["sub_index"],
            x["layout_bbox"][1],
            x["layout_bbox"][0],
        ),
    )

    for idx, block in enumerate(parsing_result):
        block["sub_index"] = idx + 1

    # vision footnote label
    nearest_match_(
        vision_footnote_blocks,
        distance_type="vision_footnote",
        is_add_index=False,
    )
    text_label_priority = {"vision_footnote": 9999}
    parsing_result.sort(
        key=lambda x: (
            x["sub_index"],
            text_label_priority.get(x["sub_label"], 0),
            x["layout_bbox"][1],
            x["layout_bbox"][0],
        ),
    )

    for idx, block in enumerate(parsing_result):
        block["sub_index"] = idx + 1

    # header、footnote、header_image... label
    nearest_match_(other_blocks, distance_type="manhattan", is_add_index=False)

    return data


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


def _calculate_horizontal_distance(
    input_bbox: List[int],
    match_bbox: List[int],
    height: int,
    disperse: int,
    title_text: List[Tuple[int, List[int]]],
) -> float:
    """
    Calculate the horizontal distance between two bounding boxes, considering title text adjustments.

    Args:
        input_bbox (List[int]): The bounding box coordinates [x1, y1, x2, y2] of the input object.
        match_bbox (List[int]): The bounding box coordinates [x1', y1', x2', y2'] of the object to match against.
        height (int): The height of the input bounding box used for normalization.
        disperse (int): The dispersion factor used to normalize the horizontal distance.
        title_text (List[Tuple[int, List[int]]]): A list of tuples containing title text information and their bounding box coordinates.
                                                  Format: [(position_indicator, [x1, y1, x2, y2]), ...].

    Returns:
        float: The calculated horizontal distance taking into account the title text adjustments.
    """
    x1, y1, x2, y2 = input_bbox
    x1_prime, y1_prime, x2_prime, y2_prime = match_bbox

    # Determine vertical distance adjustment based on title text
    if y2 < y1_prime:
        if title_text and title_text[-1][0] == 2:
            y2 += title_text[-1][1][3] - title_text[-1][1][1]
        vertical_adjustment = (y1_prime - y2) * 0.5
    else:
        if title_text and title_text[0][0] == 1:
            y1 -= title_text[0][1][3] - title_text[0][1][1]
        vertical_adjustment = y1 - y2_prime

    # Calculate horizontal distance with adjustments
    horizontal_distance = (
        abs(x2_prime - x1) // disperse
        + vertical_adjustment // height
        + vertical_adjustment / 5000
    )

    return horizontal_distance


def _calculate_vertical_distance(
    input_bbox: List[int],
    match_bbox: List[int],
    width: int,
    disperse: int,
    title_text: List[Tuple[int, List[int]]],
) -> float:
    """
    Calculate the vertical distance between two bounding boxes, considering title text adjustments.

    Args:
        input_bbox (List[int]): The bounding box coordinates [x1, y1, x2, y2] of the input object.
        match_bbox (List[int]): The bounding box coordinates [x1', y1', x2', y2'] of the object to match against.
        width (int): The width of the input bounding box used for normalization.
        disperse (int): The dispersion factor used to normalize the vertical distance.
        title_text (List[Tuple[int, List[int]]]): A list of tuples containing title text information and their bounding box coordinates.
                                                  Format: [(position_indicator, [x1, y1, x2, y2]), ...].

    Returns:
        float: The calculated vertical distance taking into account the title text adjustments.
    """
    x1, y1, x2, y2 = input_bbox
    x1_prime, y1_prime, x2_prime, y2_prime = match_bbox

    # Determine horizontal distance adjustment based on title text
    if x1 > x2_prime:
        if title_text and title_text[0][0] == 3:
            x1 -= title_text[0][1][2] - title_text[0][1][0]
        horizontal_adjustment = (x1 - x2_prime) * 0.5
    else:
        if title_text and title_text[-1][0] == 4:
            x2 += title_text[-1][1][2] - title_text[-1][1][0]
        horizontal_adjustment = x1_prime - x2

    # Calculate vertical distance with adjustments
    vertical_distance = (
        abs(y2_prime - y1) // disperse
        + horizontal_adjustment // width
        + horizontal_adjustment / 5000
    )

    return vertical_distance


def _nearest_edge_distance(
    input_bbox: List[int],
    match_bbox: List[int],
    weight: List[float] = [1.0, 1.0, 1.0, 1.0],
    label: str = "text",
    no_mask_labels: List[str] = [],
    min_edge_distances_config: List[float] = [],
    tolerance_len: float = 10.0,
) -> Tuple[float, List[float]]:
    """
    Calculate the nearest edge distance between two bounding boxes, considering directional weights.

    Args:
        input_bbox (list): The bounding box coordinates [x1, y1, x2, y2] of the input object.
        match_bbox (list): The bounding box coordinates [x1', y1', x2', y2'] of the object to match against.
        weight (list, optional): Directional weights for the edge distances [left, right, up, down]. Defaults to [1, 1, 1, 1].
        label (str, optional): The label/type of the object in the bounding box (e.g., 'text'). Defaults to 'text'.
        no_mask_labels (list, optional): Labels for which no masking is applied when calculating edge distances. Defaults to an empty list.
        min_edge_distances_config (list, optional): Configuration for minimum edge distances [min_edge_distance_x, min_edge_distance_y].
        Defaults to [float('inf'), float('inf')].
        tolerance_len (float, optional): The tolerance length for adjusting edge distances. Defaults to 10.

    Returns:
        Tuple[float, List[float]]: A tuple containing:
            - The calculated minimum edge distance between the bounding boxes.
            - A list with the minimum edge distances in the x and y directions.
    """
    match_bbox_iou = _calculate_overlap_area_div_minbox_area_ratio(
        input_bbox,
        match_bbox,
    )
    if match_bbox_iou > 0 and label not in no_mask_labels:
        return 0, [0, 0]

    if not min_edge_distances_config:
        min_edge_distances_config = [float("inf"), float("inf")]
    min_edge_distance_x, min_edge_distance_y = min_edge_distances_config

    x1, y1, x2, y2 = input_bbox
    x1_prime, y1_prime, x2_prime, y2_prime = match_bbox

    direction_num = 0
    distance_x = float("inf")
    distance_y = float("inf")
    distance = [float("inf")] * 4

    # input_bbox is to the left of match_bbox
    if x2 < x1_prime:
        direction_num += 1
        distance[0] = x1_prime - x2
        if abs(distance[0] - min_edge_distance_x) <= tolerance_len:
            distance_x = min_edge_distance_x * weight[0]
        else:
            distance_x = distance[0] * weight[0]
    # input_bbox is to the right of match_bbox
    elif x1 > x2_prime:
        direction_num += 1
        distance[1] = x1 - x2_prime
        if abs(distance[1] - min_edge_distance_x) <= tolerance_len:
            distance_x = min_edge_distance_x * weight[1]
        else:
            distance_x = distance[1] * weight[1]
    elif match_bbox_iou > 0:
        distance[0] = 0
        distance_x = 0

    # input_bbox is above match_bbox
    if y2 < y1_prime:
        direction_num += 1
        distance[2] = y1_prime - y2
        if abs(distance[2] - min_edge_distance_y) <= tolerance_len:
            distance_y = min_edge_distance_y * weight[2]
        else:
            distance_y = distance[2] * weight[2]
        if label in no_mask_labels:
            distance_y = max(0.1, distance_y) * 100
    # input_bbox is below match_bbox
    elif y1 > y2_prime:
        direction_num += 1
        distance[3] = y1 - y2_prime
        if abs(distance[3] - min_edge_distance_y) <= tolerance_len:
            distance_y = min_edge_distance_y * weight[3]
        else:
            distance_y = distance[3] * weight[3]
    elif match_bbox_iou > 0:
        distance[2] = 0
        distance_y = 0

    if direction_num == 2:
        return (distance_x + distance_y), [
            min(distance[0], distance[1]),
            min(distance[2], distance[3]),
        ]
    else:
        return min(distance_x, distance_y), [
            min(distance[0], distance[1]),
            min(distance[2], distance[3]),
        ]


def _get_weights(label, horizontal):
    """Define weights based on the label and orientation."""
    if label == "doc_title":
        return (
            [1, 0.1, 0.1, 1] if horizontal else [0.2, 0.1, 1, 1]
        )  # left-down ,  right-left
    elif label in [
        "paragraph_title",
        "abstract",
        "figure_title",
        "chart_title",
        "image",
        "seal",
        "chart",
        "figure",
    ]:
        return [1, 1, 0.1, 1]  # down
    else:
        return [1, 1, 1, 0.1]  # up


def _nearest_iou_edge_distance(
    input_bbox: List[int],
    match_bbox: List[int],
    label: str,
    vision_labels: List[str],
    no_mask_labels: List[str],
    median_width: int = -1,
    title_labels: List[str] = [],
    title_text: List[Tuple[int, List[int]]] = [],
    sub_title: List[List[int]] = [],
    min_distance_config: List[float] = [],
    tolerance_len: float = 10.0,
) -> Tuple[float, List[float]]:
    """
    Calculate the nearest IOU edge distance between two bounding boxes, considering label types, title adjustments, and minimum distance configurations.
    This function computes the edge distance between two bounding boxes while considering their overlap (IOU) and various adjustments based on label types,
    title text, and subtitle information. It also applies minimum distance configurations and tolerance adjustments.

    Args:
        input_bbox (List[int]): The bounding box coordinates [x1, y1, x2, y2] of the input object.
        match_bbox (List[int]): The bounding box coordinates [x1', y1', x2', y2'] of the object to match against.
        label (str): The label/type of the object in the bounding box (e.g., 'image', 'text', etc.).
        vision_labels (List[str]): List of labels for vision-related objects (e.g., images, icons).
        no_mask_labels (List[str]): Labels for which no masking is applied when calculating edge distances.
        median_width (int, optional): The median width for title dispersion calculation. Defaults to -1.
        title_labels (List[str], optional): Labels that indicate the object is a title. Defaults to an empty list.
        title_text (List[Tuple[int, List[int]]], optional): Text content associated with title labels, in the format [(position_indicator, [x1, y1, x2, y2]), ...].
        sub_title (List[List[int]], optional): List of subtitle bounding boxes to adjust the input_bbox. Defaults to an empty list.
        min_distance_config (List[float], optional): Configuration for minimum distances [min_edge_distances_config, up_edge_distances_config, total_distance].
        tolerance_len (float, optional): The tolerance length for adjusting edge distances. Defaults to 10.0.

    Returns:
        Tuple[float, List[float]]: A tuple containing:
            - The calculated distance considering IOU and adjustments.
            - The updated minimum distance configuration.
    """

    x1, y1, x2, y2 = input_bbox
    x1_prime, y1_prime, x2_prime, y2_prime = match_bbox

    min_edge_distances_config, up_edge_distances_config, total_distance = (
        min_distance_config
    )

    iou_distance = 0

    if label in vision_labels:
        horizontal1 = horizontal2 = True
    else:
        horizontal1 = _get_bbox_direction(input_bbox)
        horizontal2 = _get_bbox_direction(match_bbox, 3)

    if (
        horizontal1 != horizontal2
        or _get_projection_iou(input_bbox, match_bbox, horizontal1) < 0.01
    ):
        iou_distance = 1
    elif label == "doc_title" or (label in title_labels and title_text):
        # Calculate distance for titles
        disperse = max(1, median_width)
        width = x2 - x1
        height = y2 - y1
        if horizontal1:
            return (
                _calculate_horizontal_distance(
                    input_bbox,
                    match_bbox,
                    height,
                    disperse,
                    title_text,
                ),
                min_distance_config,
            )
        else:
            return (
                _calculate_vertical_distance(
                    input_bbox,
                    match_bbox,
                    width,
                    disperse,
                    title_text,
                ),
                min_distance_config,
            )

    # Adjust input_bbox based on sub_title
    if sub_title:
        for sub in sub_title:
            x1_, y1_, x2_, y2_ = sub
            x1, y1, x2, y2 = (
                min(x1, x1_),
                min(
                    y1,
                    y1_,
                ),
                max(x2, x2_),
                max(y2, y2_),
            )
        input_bbox = [x1, y1, x2, y2]

    # Calculate edge distance
    weight = _get_weights(label, horizontal1)
    if label == "abstract":
        tolerance_len *= 3
    edge_distance, edge_distance_config = _nearest_edge_distance(
        input_bbox,
        match_bbox,
        weight,
        label=label,
        no_mask_labels=no_mask_labels,
        min_edge_distances_config=min_edge_distances_config,
        tolerance_len=tolerance_len,
    )

    # Weights for combining distances
    iou_edge_weight = [10**6, 10**3, 1, 0.001]

    # Calculate up and left edge distances
    up_edge_distance = y1_prime
    left_edge_distance = x1_prime
    if (
        label in no_mask_labels or label == "paragraph_title" or label in vision_labels
    ) and y1 > y2_prime:
        up_edge_distance = -y2_prime
        left_edge_distance = -x2_prime

    min_up_edge_distance = up_edge_distances_config
    if abs(min_up_edge_distance - up_edge_distance) <= tolerance_len:
        up_edge_distance = min_up_edge_distance

    # Calculate total distance
    distance = (
        iou_distance * iou_edge_weight[0]
        + edge_distance * iou_edge_weight[1]
        + up_edge_distance * iou_edge_weight[2]
        + left_edge_distance * iou_edge_weight[3]
    )

    # Update minimum distance configuration if a smaller distance is found
    if total_distance > distance:
        edge_distance_config = [
            min(min_edge_distances_config[0], edge_distance_config[0]),
            min(min_edge_distances_config[1], edge_distance_config[1]),
        ]
        min_distance_config = [
            edge_distance_config,
            min(up_edge_distance, up_edge_distances_config),
            distance,
        ]

    return distance, min_distance_config


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
