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

__all__ = ["cal_ocr_word_box"]

import numpy as np

# from .convert_points_and_boxes import convert_points_to_boxes


def cal_ocr_word_box(rec_str, box, rec_word_info):
    """Calculate the detection frame for each word based on the results of recognition and detection of ocr"""

    col_num, word_list, word_col_list, state_list = rec_word_info
    box = box.tolist()
    bbox_x_start = box[0][0]
    bbox_x_end = box[1][0]
    bbox_y_start = box[0][1]
    bbox_y_end = box[2][1]

    cell_width = (bbox_x_end - bbox_x_start) / col_num

    word_box_list = []
    word_box_content_list = []
    cn_width_list = []
    cn_col_list = []
    for word, word_col, state in zip(word_list, word_col_list, state_list):
        if state == "cn":
            if len(word_col) != 1:
                char_seq_length = (word_col[-1] - word_col[0] + 1) * cell_width
                char_width = char_seq_length / (len(word_col) - 1)
                cn_width_list.append(char_width)
            cn_col_list += word_col
            word_box_content_list += word
        else:
            cell_x_start = bbox_x_start + int(word_col[0] * cell_width)
            cell_x_end = bbox_x_start + int((word_col[-1] + 1) * cell_width)
            cell = (
                (cell_x_start, bbox_y_start),
                (cell_x_end, bbox_y_start),
                (cell_x_end, bbox_y_end),
                (cell_x_start, bbox_y_end),
            )
            word_box_list.append(cell)
            word_box_content_list.append("".join(word))
    if len(cn_col_list) != 0:
        if len(cn_width_list) != 0:
            avg_char_width = np.mean(cn_width_list)
        else:
            avg_char_width = (bbox_x_end - bbox_x_start) / len(rec_str)
        for center_idx in cn_col_list:
            center_x = (center_idx + 0.5) * cell_width
            cell_x_start = max(int(center_x - avg_char_width / 2), 0) + bbox_x_start
            cell_x_end = (
                min(int(center_x + avg_char_width / 2), bbox_x_end - bbox_x_start)
                + bbox_x_start
            )
            cell = (
                (cell_x_start, bbox_y_start),
                (cell_x_end, bbox_y_start),
                (cell_x_end, bbox_y_end),
                (cell_x_start, bbox_y_end),
            )
            word_box_list.append(cell)
    word_box_list = sort_boxes(word_box_list, y_thresh=12)
    return word_box_content_list, word_box_list


def sort_boxes(boxes, y_thresh=10):

    box_centers = [np.mean(box, axis=0) for box in boxes]
    items = list(zip(boxes, box_centers))
    items.sort(key=lambda x: x[1][1])

    lines = []
    current_line = []
    last_y = None
    for box, center in items:
        if last_y is None or abs(center[1] - last_y) < y_thresh:
            current_line.append((box, center))
        else:
            lines.append(current_line)
            current_line = [(box, center)]
        last_y = center[1]
    if current_line:
        lines.append(current_line)

    final_box = []
    for line in lines:
        line = sorted(line, key=lambda x: x[1][0])
        final_box.extend(box for box, center in line)

    return final_box
