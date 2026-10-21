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

__all__ = ["get_neighbor_boxes_idx", "TableRec"]

import os
import pickle
from copy import deepcopy
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd


def get_neighbor_boxes_idx(src_boxes: np.ndarray, ref_box: np.ndarray) -> list:
    """
    Retrieve indices of source boxes that are neighbors to the reference box.

    Parameters:
    src_boxes (np.ndarray): An array of bounding boxes with shape (N, 4),
                            where N is the number of boxes and each box is represented
                            by [x1, y1, x2, y2].
    ref_box (np.ndarray): A single bounding box represented by [x1, y1, x2, y2].

    Returns:
    list: A list of indices of the source boxes that are close to the
          reference box based on the intersection area.
    """
    match_idx_list = []
    if len(src_boxes) > 0:
        x1 = np.maximum(ref_box[0], src_boxes[:, 0])
        y1 = np.maximum(ref_box[1], src_boxes[:, 1])
        x2 = np.minimum(ref_box[2], src_boxes[:, 2])
        y2 = np.minimum(ref_box[3], src_boxes[:, 3])
        pub_w = x2 - x1
        pub_h = y2 - y1
        match_idx = np.where((pub_w > 0) & (pub_h < 3) & (pub_h > -15))[0]
        match_idx_list.extend(match_idx)
    return match_idx_list


class TableRec:
    def __init__(
        self,
        md_path="./paddlex/inference/pipelines/table_recognition/xgb.pickle",
    ) -> None:
        """An Table Recognition Method

        Args:
            md_path: the path of xgb model
        """
        self.table_ocr = TableOCR(md_path)

    def predict(self, img_path, result_of_ocr, result_of_det_cells_model):
        img = self.read_img(img_path=img_path)
        frame_lines = self.classify_table(img)
        if len(frame_lines) == 0:
            pred_bounds = self.predict_no_frame(
                img, img_path, result_of_det_cells_model, result_of_ocr
            )
        else:
            pred_bounds = self.predict_three_lines_frame(img, result_of_det_cells_model)
        return pred_bounds

    def predict_no_frame(self, img, img_path, result_of_det_cells_model, result_of_ocr):
        """recognize table without frame and lines

        Args:
            img (array): binary array
            img_path (str): path of image
            result_of_det_cells_model (Result): result of det cells model in paddlex

        Returns:
            list: list of bounds, [(x1, y1, x2, y2), ...]
        """
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        output = result_of_det_cells_model
        pred_bounds = []
        for res in output:
            pred_dicts = res._to_json()["res"]["boxes"]
            for k in range(len(pred_dicts)):
                pred_bounds.append(pred_dicts[k]["coordinate"])
        regions = self.table_ocr.predict(img_path, result_of_ocr)
        cells_info = self.merge_predictions(img, pred_bounds, regions)
        pred_bounds = [d["bound"] for d in cells_info]
        pred_bounds = self.frame_alignment(img, pred_bounds)
        pred_bounds = self.deal_columns(pred_bounds, regions, img=img)
        return pred_bounds

    def predict_three_lines_frame(self, img, result_of_det_cells_model):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        output = result_of_det_cells_model
        pred_bounds = []
        for res in output:
            pred_dicts = res._to_json()["res"]["boxes"]
            for k in range(len(pred_dicts)):
                pred_bounds.append(pred_dicts[k]["coordinate"])

        return pred_bounds

    def deal_columns(self, pred_bounds, regions, img, scope_iou_thresh=0.3):
        """Add missing columns

        Args:
            pred_bounds (list): [(x1, y1, x2, y2), ...]
            regions (lis): list of dict, [{'bound': [x1, y1, x2, y2], 'text_boxes': [[x1, y1, w, h], ...], 'empty_cell': 0|1}, ...]
        Returns:
            bounds (list):  [(x1, y1, x2, y2), ...]
        """
        # get columns
        pred_bounds = list(sorted(pred_bounds, key=lambda x: (x[0], x[1])))
        pred_col_subgraphs = self.col_subgraphs(
            pred_bounds, scope_iou_thresh=scope_iou_thresh
        )
        region_bounds = [d["bound"] for d in regions]
        region_col_subgraphs = self.col_subgraphs(region_bounds)

        pred_first_col = pred_col_subgraphs["0"]
        region_first_col = region_col_subgraphs["0"]
        if len(region_first_col["cells"]) > 2:
            r_x1, r_x2 = region_first_col["scope"]
        else:
            r_x1, r_x2 = pred_first_col["scope"]

        for k, v in pred_col_subgraphs.items():
            cells = v["cells"]
            new_cells = []
            for cell in cells:
                x1, y1, x2, y2 = cell
                tmp_iou = self.compute_scope_iou_within_scope1((r_x1, r_x2), (x1, x2))
                if tmp_iou > 0.6:
                    if r_x2 - x1 > 5:
                        new_cells.append((x1, y1, r_x2, y2))
                    if x2 - r_x2 > 5:
                        new_cells.append((r_x2, y1, x2, y2))
                else:
                    new_cells.append(cell)

            pred_col_subgraphs[k]["cells"] = new_cells

        new_pred_bounds = []
        for k, v in pred_col_subgraphs.items():
            new_pred_bounds.extend(v["cells"])

        return new_pred_bounds

    def col_subgraphs(self, bounds, scope_iou_thresh=0.6):
        col_subgraphs = {}
        for d in bounds:
            x1, y1, x2, y2 = d
            if len(col_subgraphs) == 0:
                col_subgraphs[str(len(col_subgraphs))] = {
                    "scope": [x1, x2],
                    "cells": [d],
                }
                continue
            col_x1, col_x2 = col_subgraphs[str(len(col_subgraphs) - 1)]["scope"]
            if x1 >= col_x1 - 3 and x2 <= col_x2 + 3:
                iou = 1
            else:
                iou = self.compute_scope_iou((col_x1, col_x2), (x1, x2))
            if iou >= scope_iou_thresh:
                if d in col_subgraphs[str(len(col_subgraphs) - 1)]["cells"]:
                    continue
                col_subgraphs[str(len(col_subgraphs) - 1)]["cells"].append(d)
                col_x1 = min(col_x1, x1)
                col_x2 = max(col_x2, x2)
                col_subgraphs[str(len(col_subgraphs) - 1)]["scope"] = [col_x1, col_x2]
            else:
                new_key = str(len(col_subgraphs))
                col_subgraphs[new_key] = {}
                col_subgraphs[new_key]["scope"] = [x1, x2]
                col_subgraphs[new_key]["cells"] = [d]
        return col_subgraphs

    def frame_alignment(self, img, bounds):
        """align all the bounds into frame

        Args:
            bounds (list): the bounds of cells. [(x1, y1, x2, y2), ......]
        """

        def get_weighted_value(bound_votes, st, ed):
            """Include end

            Args:
                bound_votes (list): Record the vote of new bounds of the predicted bounds.
                st (int): Start of no zero vote
                ed (int): End of no zero vote

            Returns:
                _type_: _description_
            """
            votes = np.sum(bound_votes[st : ed + 1])
            new_b = 0
            for q in range(st, ed + 1):
                new_b += q * bound_votes[q] / votes
            new_b = int(new_b)
            return new_b

        def get_new_bounds(bound_votes, align_thresh):
            new_bounds = []
            i = 0
            st = -1
            ed = -1
            while i < len(bound_votes):
                if bound_votes[i] == 0 and st == -1:
                    i += 1
                    continue
                elif bound_votes[i] == 0 and st != -1:
                    stop_guard = False
                    if st + align_thresh - 1 < len(bound_votes):
                        if np.sum(bound_votes[st : st + align_thresh]) > np.sum(
                            bound_votes[st:i]
                        ):
                            pass
                        else:
                            stop_guard = True
                    else:
                        if np.sum(bound_votes[st:-1]) > np.sum(bound_votes[st:i]):
                            pass
                        else:
                            stop_guard = True

                    if stop_guard:
                        ed = i - 1
                        new_b = get_weighted_value(bound_votes, st, ed)
                        new_bounds.append(new_b)
                        st = -1
                        ed = -1
                    else:
                        ed = i
                elif bound_votes[i] != 0 and st == -1:
                    st = i
                else:
                    ed = i

                if ed - st >= align_thresh - 1:
                    new_b = get_weighted_value(bound_votes, st, ed)
                    st = -1
                    ed = -1
                    new_bounds.append(new_b)

                i += 1

            return new_bounds

        row_votes = [0 for i in range(img.shape[0])]
        col_votes = [0 for i in range(img.shape[1])]
        for bound in bounds:
            x1, y1, x2, y2 = bound
            if y1 < img.shape[0]:
                row_votes[int(y1)] += 1
            else:
                row_votes[-1] += 1
            if y2 < img.shape[0]:
                row_votes[int(y2)] += 1
            else:
                row_votes[-1] += 1

            if x1 < img.shape[1]:
                col_votes[int(x1)] += 1
            else:
                col_votes[-1] += 1
            if x2 < img.shape[1]:
                col_votes[int(x2)] += 1
            else:
                col_votes[-1] += 1

        # get new row bounds and col bounds
        row_bounds = get_new_bounds(bound_votes=row_votes, align_thresh=7)
        col_bounds = get_new_bounds(bound_votes=col_votes, align_thresh=11)

        new_bounds = []
        for k in range(len(bounds)):
            x1, y1, x2, y2 = bounds[k]
            for _y in row_bounds:
                if abs(y1 - _y) < 10:
                    y1 = _y
                if abs(y2 - _y) < 10:
                    y2 = _y
            for _x in col_bounds:
                if abs(x1 - _x) < 10:
                    x1 = _x
                if abs(x2 - _x) < 10:
                    x2 = _x
            new_bounds.append((int(x1), int(y1), int(x2), int(y2)))
        return new_bounds

    def merge_predictions(self, img, pred_bounds, regions, match_iou_thresh=0.5):
        """merge predictions

        Args:
            img (Array): gray image
            pred_bounds (list): [(x1, y1, x2, y2), ...]
            regions (list): list of dict, [{'bound': [x1, y1, x2, y2], 'text_boxes': [[x1, y1, w, h], ...], 'empty_cell': 0|1}, ...]

        Returns:
            cells_info (list): list of dict
        """
        add_cells_info = []
        for i in range(len(regions)):
            region_info = regions[i]
            region_bound = region_info["bound"]
            miss_guard = True
            for pred_bound in pred_bounds:
                region_iou = compute_iou(region_bound, pred_bound)
                if region_iou >= match_iou_thresh:
                    miss_guard = False
                    break
            if miss_guard:
                if "empty_cell" in region_info.keys():
                    x1, y1, x2, y2 = region_bound
                    x1 = int(x1)
                    y1 = int(y1)
                    x2 = int(x2)
                    y2 = int(y2)
                    empty_cell = region_info["empty_cell"]
                else:
                    x1, y1, x2, y2 = region_bound
                    x1 = int(x1)
                    y1 = int(y1)
                    x2 = int(x2)
                    y2 = int(y2)
                    tmp_img = img[y1:y2, x1:x2]
                    if np.sum(tmp_img) < 5:
                        empty_cell = 1
                    else:
                        empty_cell = 0
                add_cells_info.append(
                    {"bound": (x1, y1, x2, y2), "empty_cell": empty_cell}
                )

        cells_info = add_cells_info
        for pred_bound in pred_bounds:
            x1, y1, x2, y2 = pred_bound
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            tmp_img = img[y1:y2, x1:x2]
            if np.sum(tmp_img) < 5:
                empty_cell = 1
            else:
                empty_cell = 0
            info = {"bound": (x1, y1, x2, y2), "empty_cell": empty_cell}
            add_cells_info.append(info)

        return cells_info

    def analyse_frame_lines(self, img, frame_lines, header_lines_thresh=0.6):
        frame_lines = list(sorted(frame_lines, key=lambda x: x[1]))
        max_length = -1
        max_line_x1 = -1
        max_line_x2 = -1
        for i in range(len(frame_lines)):
            line = frame_lines[i]
            x1, y1, x2, y2 = line
            if max_length < abs(x2 - x1):
                max_length = abs(x2 - x1)
                max_line_x1 = min(x1, x2)
                max_line_x2 = max(x1, x2)

        # get complex row lines
        frame_info = {
            "frame_width": max_length,
            "frame_x1": max_line_x1,
            "frame_x2": max_line_x2,
            "header_lines": [],
            "skip_index_cell_lines": [],
        }
        # # get header lines
        for line in frame_lines:
            x1, y1, x2, y2 = line
            if abs(x1 - x2) < max_length * header_lines_thresh:
                frame_info["header_lines"].append(line)

        old_header_lines = deepcopy(frame_info["header_lines"])
        for line in frame_lines:
            x1, y1, x2, y2 = line
            y = np.mean([y1, y2])
            if line in old_header_lines:
                continue
            for header_line in old_header_lines:
                h_x1, h_y1, h_x2, h_y2 = header_line
                tmp_y = np.mean([h_y1, h_y2])
                if abs(y - tmp_y) < 3:
                    frame_info["header_lines"].append(line)

        # # get merged index cells lines
        for line in frame_lines:
            x1, y1, x2, y2 = line
            if (
                abs(x2 - x1) >= max_length * header_lines_thresh
                and abs(x2 - x1) < (max_length - 20)
                and abs(max_line_x2 - max(x1, x2)) < 3
            ):
                frame_info["skip_index_cell_lines"].append(line)

        return frame_info

    def classify_table(self, img, min_length_thresh=0.1):
        h, w = img.shape
        lsd = cv2.createLineSegmentDetector(
            refine=None,
            scale=0.8,
            sigma_scale=0.6,
            quant=2.0,
            ang_th=22.5,
            log_eps=0,
            density_th=0.7,
            n_bins=1024,
        )
        lines, width, prec, nfa = lsd.detect(img)
        frame_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dis = euclidean_distance(point1=(x1, y1), point2=(x2, y2))
            if dis < min_length_thresh * w:
                continue
            frame_lines.append((x1, y1, x2, y2))

        return frame_lines

    def read_img(self, img_path: str):
        assert os.path.exists(img_path), f"img_path doesn't exists \n{img_path}"
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        return img

    def compute_scope_iou(self, scope1, scope2):
        """compute the iou of scope of rows or columns

        Args:
            scope1 (list): row scope | column scope, (y1, y2) | (x1, x2)
            scope2 (list): row scope | column scope, (y1, y2) | (x1, x2)
        """
        a1, a2 = scope1
        a1 = min(a1, a1)
        a2 = max(a1, a2)
        b1, b2 = scope2
        b1 = min(b1, b2)
        b2 = max(b1, b2)

        if a2 <= b1 or b2 <= a1:
            return 0

        s1 = max(a1, b1)
        s2 = min(a2, b2)
        l1 = min(a1, b1)
        l2 = max(a2, b2)

        iou = round((s2 - s1) / (l2 - l1), 4)
        return iou

    def compute_scope_iou_within_scope1(self, scope1, scope2):
        """compute the iou of scope of rows or columns

        Args:
            scope1 (list): row scope | column scope, (y1, y2) | (x1, x2)
            scope2 (list): row scope | column scope, (y1, y2) | (x1, x2)
        """
        a1, a2 = scope1
        a1 = min(a1, a1)
        a2 = max(a1, a2)
        b1, b2 = scope2
        b1 = min(b1, b2)
        b2 = max(b1, b2)

        if a2 <= b1 or b2 <= a1:
            return 0

        s1 = max(a1, b1)
        s2 = min(a2, b2)

        iou = round((s2 - s1) / (a2 - a1), 4)
        return iou


class TableOCR:
    def __init__(
        self,
        md_path,
    ) -> None:
        """An Table Recognition Method

        Args:
            md_path: the path of xgb model
        """
        self.model = pickle.load(open(md_path, "rb"))

    def predict(self, img_path, result_of_ocr):
        img = self.check_and_read_img(img_path=img_path)
        res_boxes = self.get_ocr_text_boxes(img_path, result_of_ocr)
        canvas = self.ocr_box_canvas(text_boxes=res_boxes, img_shape=img.shape)
        canvas = canvas * 255
        regions = self.split_into_region(canvas=canvas, text_boxes=res_boxes, img=img)
        regions = self.merge_same_cells_deal_complex_region(regions=regions, img=img)
        return regions

    def get_ocr_text_boxes(self, img_path, result_of_ocr):
        """get the text boxes of the ocr result of the img

        Args:
            img_path (str, optional): image path. Defaults to None.
            result_of_ocr (Result): the result of ocr model using pipeline of paddlex

        Returns:
            shrink_boxes (list): box [x, y, w, h]
        """
        ocr_res_json = result_of_ocr._to_json()["res"]
        rec_boxes = ocr_res_json["rec_boxes"]
        img = self.check_and_read_img(img_path=img_path)
        shrink_boxes = []
        for i in range(len(rec_boxes)):
            box = rec_boxes[i]
            box_img = self.get_box_img(box=box, img=img)
            assert (
                box_img.shape[0] != 0 and box_img.shape[1] != 0
            ), f"box_img is empty, img_path: {img_path}"
            shrink_box_img, shrink_box = self.shrink_text_box(
                box_img=box_img, origin_box=box
            )

            new_shrink_boxes, new_box_imgs = self.modify_text_boxes(
                text_box=shrink_box, box_img=shrink_box_img
            )

            for k in range(len(new_shrink_boxes)):
                tmp_box = new_shrink_boxes[k]
                shrink_boxes.append(tmp_box)

        return shrink_boxes

    def get_box_img(self, box, img):
        """_summary_

        Args:
            box (List): [x0, y0, x1, y1]
            img (np.ndarray): image matrix array

        Returns:
            _type_: _description_
        """
        box_img = img[box[1] : box[3], box[0] : box[2]]
        return box_img

    def check_and_read_img(self, img_path: str):
        assert os.path.exists(img_path), f"img_path doesn't exists \n{img_path}"
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        return img

    def shrink_text_box(self, box_img, origin_box):
        """only consider the table line is vertical or horizontal

        Args:
            box_img (np.ndarray): img
            origin_box (List): [x0, y0, x1, y1]
        Returns:
            shrink_box_img (np.ndarray): img after shrink
            shrink_box (List): [x, y, w, h]

        """
        line_thresh = 0.05 * box_img.shape[1]
        margin_thresh = 0.95 * box_img.shape[1]

        _, binary_image = cv2.threshold(box_img, 127, 1, cv2.THRESH_BINARY)

        # vertical analysis
        vertical_accum = []
        for i in range(binary_image.shape[0]):
            vertical_accum.append(np.sum(binary_image[i, :]))

        detail = {"margin": [], "line": []}
        margin_st = -1
        line_st = -1
        # detect line and margin
        for k in range(binary_image.shape[0]):
            if vertical_accum[k] < line_thresh:
                if margin_st != -1:
                    detail["margin"].append([margin_st, k - 1])
                    margin_st = -1
                if line_st == -1:
                    line_st = k
            elif vertical_accum[k] > margin_thresh:
                if line_st != -1:
                    detail["line"].append([line_st, k - 1])
                    line_st = -1
                if margin_st == -1:
                    margin_st = k
            else:
                if line_st != -1:
                    detail["line"].append([line_st, k - 1])
                    line_st = -1
                if margin_st != -1:
                    detail["margin"].append([margin_st, k - 1])
                    margin_st = -1

        if margin_st != -1:
            detail["margin"].append([margin_st, binary_image.shape[0] - 1])

        if len(detail["margin"]) == 1 and len(detail["line"]) == 0:
            return box_img, [
                origin_box[0],
                origin_box[1],
                origin_box[2] - origin_box[0],
                origin_box[3] - origin_box[1],
            ]

        ver_main_scope_len = 0
        ver_main_scope = [0, 0]
        for j in range(len(detail["margin"]) - 1):
            text_line_st = detail["margin"][j][1]
            text_line_ed = detail["margin"][j + 1][0]
            if text_line_ed - text_line_st >= ver_main_scope_len:
                ver_main_scope_len = text_line_ed - text_line_st
                ver_main_scope = [text_line_st, text_line_ed]

        if ver_main_scope == [0, 0]:
            ver_main_scope = [0, binary_image.shape[0]]
        vertical_shrink_box = binary_image[ver_main_scope[0] : ver_main_scope[1], :]

        # horizontal analysis
        line_thresh = 0.05 * vertical_shrink_box.shape[0]
        margin_thresh = 0.95 * vertical_shrink_box.shape[0]
        horizontal_accum = []
        for i in range(vertical_shrink_box.shape[1]):
            horizontal_accum.append(np.sum(vertical_shrink_box[:, i]))

        detail = {"margin": [], "line": []}
        margins = []
        margin_st = -1
        line_st = -1
        for k in range(vertical_shrink_box.shape[1]):
            if horizontal_accum[k] < line_thresh:
                if margin_st != -1:
                    if k - 1 - margin_st > 0:
                        detail["margin"].append([margin_st, k - 1])
                        margins.append(k - 1 - margin_st)
                    margin_st = -1
                if line_st == -1:
                    line_st = k
            elif horizontal_accum[k] > margin_thresh:
                if line_st != -1:
                    detail["line"].append([line_st, k - 1])
                    line_st = -1
                if margin_st == -1:
                    margin_st = k
            else:
                if line_st != -1:
                    line_st = -1
                if margin_st != -1:
                    if k - 1 - margin_st > 0:
                        detail["margin"].append([margin_st, k - 1])
                        margins.append(k - 1 - margin_st)
                    margin_st = -1

        if len(detail["margin"]) == 1 and len(detail["line"]) == 0:
            return box_img[ver_main_scope[0] : ver_main_scope[1], :], [
                origin_box[0],
                origin_box[1] + ver_main_scope[0],
                origin_box[2] - origin_box[0],
                ver_main_scope[1] - ver_main_scope[0],
            ]

        hor_main_scope = [0, 0]
        hor_main_scope_len = 0

        if len(detail["line"]) > 0:
            for n in range(len(detail["line"])):
                line = detail["line"][n]
                if n == 0 and line[0] != 0:
                    hor_main_scope = [0, line[0] - 1]
                    hor_main_scope_len = line[0] - 1
                elif n != 0:
                    if (
                        detail["line"][n][0] - 1 - (detail["line"][n - 1][1] + 1)
                        > hor_main_scope_len
                    ):
                        hor_main_scope = [
                            detail["line"][n - 1][1] + 1,
                            detail["line"][n][0] - 1,
                        ]
                        hor_main_scope_len = (
                            detail["line"][n][0] - 1 - (detail["line"][n - 1][1] + 1)
                        )

        if hor_main_scope == [0, 0]:
            hor_main_scope = [0, box_img.shape[1]]

        shrink_img = box_img[
            ver_main_scope[0] : ver_main_scope[1], hor_main_scope[0] : hor_main_scope[1]
        ]
        shrink_box = [
            origin_box[0] + hor_main_scope[0],
            origin_box[1] + ver_main_scope[0],
            hor_main_scope[1] - hor_main_scope[0],
            ver_main_scope[1] - ver_main_scope[0],
        ]

        return shrink_img, shrink_box

    def box_to_four_coordinates(self, box):
        """transform box from [x, y, w, h] to four points, x is vertical axis, and y is horizontal axis

        Args:
            box (list): [x, y, w, h]
        Returns:
            pts (list): four points of polygon
        """
        origin_x, origin_y, w, h = box
        pts = [
            (origin_x, origin_y),
            (origin_x + w, origin_y),
            (origin_x + w, origin_y + h),
            (origin_x, origin_y + h),
        ]
        return pts

    def transform_x1y1x2y2_into_four_coordinates(self, ocr_box):
        """

        Args:
            ocr_box (List): [x0, y0, x1, y1]
        Returns:
            pts (List): four points of ocr box
        """
        x0, y0, x1, y1 = ocr_box
        pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        return pts

    def ocr_box_canvas(self, text_boxes: List, img_shape: Tuple):
        """draw text boxes in a canvas

        Args:
            text_boxes (List): [x, y, width, height]
            img_shape (Tuple): img.shape
        Returns:
            canvas (np.ndarray): blank matrix with text box as 1
        """
        canvas = np.zeros(img_shape)
        for box in text_boxes:
            x, y, w, h = box
            canvas[y : y + h, x : x + w] = 1
        return canvas

    def split_into_region(
        self,
        canvas: np.ndarray,
        text_boxes: List,
        img: np.ndarray = None,
        iou_thresh=0.8,
    ):
        """split text boxes into region cell, detect empty cell inside

        Args:
            canvas (np.ndarray): blank matrix with text box as 1
            text_boxes (List): the small text boxes, [[x, y, w, h], ...]
            img (np.ndarray): for debug
        Returns:
            region (List): list of group of text boxes, [{'bound': [x1, y1, x2, y2], 'text_boxes': [[x1, y1, w, h], ...]}, ...]
        """
        subgraphs = self.split_into_subgraph(
            canvas=canvas, text_boxes=text_boxes, img=img
        )
        row_subgraphs = subgraphs["row"]
        col_subgraphs = subgraphs["col"]

        regions = []  # [{'bound': [x1, y1, x2, y2], 'text_boxes': [[x1, y1, w, h]]}]
        i = 0
        j = 0

        for i in row_subgraphs.keys():
            r_y1, r_y2 = row_subgraphs[i]["scope"]
            for j in col_subgraphs.keys():
                r_x1, r_x2 = col_subgraphs[j]["scope"]
                rect2 = [r_x1, r_y1, r_x2, r_y2]
                region = {"bound": rect2, "text_boxes": [], "empty_cell": 0}
                for box in text_boxes:
                    x1, y1, w, h = box
                    x2 = x1 + w
                    y2 = y1 + h
                    rect1 = [x1, y1, x2, y2]
                    iou = compute_iou(box1=rect1, box2=rect2)
                    if iou >= iou_thresh:
                        region["text_boxes"].append(box)

                if len(region["text_boxes"]) == 0:
                    region["empty_cell"] = 1

                regions.append(region)

        return regions

    def get_bounds(self, bin_canvas: np.ndarray, mode: str):
        """get margin bounds of row or column of one image

        Args:
            bin_canvas (np.ndarray): binary image to analyse
            mode (str): 'row'|'col'
        Returns:
            list: bounds [int, ...]
        """
        assert mode in ["row", "col"], f"Mode {mode} is not supported yet"
        margin_bounds = []
        st = -1
        ed = -1
        margin_st = -1
        margin_ed = -1
        if mode == "row":
            range_scope = bin_canvas.shape[0]
        elif mode == "col":
            range_scope = bin_canvas.shape[1]

        proj_values = []  # debug param
        for i in range(range_scope):
            if mode == "row":
                proj = np.sum(bin_canvas[i, :])
            elif mode == "col":
                proj = np.sum(bin_canvas[:, i])

            proj_values.append(proj)
            # start st point
            if proj != 0 and margin_st == -1:
                if i == 0:
                    st = i
                else:
                    st = i - 1
                margin_st = 0
                continue

            if proj == 0 and margin_st != -1:
                margin_st = i
                continue
            elif proj != 0 and margin_st > 0:
                margin_ed = i - 1
                margin_bounds.append([margin_st, margin_ed])
                margin_st = 0
                margin_ed = 0

            if proj != 0:
                ed = i

        margin_bounds = [int(np.mean(bound)) for bound in margin_bounds]

        margin_bounds.insert(0, st)
        margin_bounds.append(ed)

        return margin_bounds

    def merge_same_cells_deal_complex_region(
        self, regions: list, img: np.ndarray, iou_thresh=0.6, dis_thresh=25
    ):
        """merge same cell text ocr boxes,

        Args:
            regions (List): list of group of text boxes, [{'bound': [x1, y1, x2, y2], 'text_boxes': [[x1, y1, w, h], ...], 'empty_cell': 0|1}, ...]
            img (np.ndarray): image

        Returns:
            new_regions_info (List): list of dict, [{'bound': [x1, y1, x2, y2], 'text_boxes': [[x1, y1, w, h], ...], 'empty_cell': 0|1}, ...]
        """
        new_regions_info = []
        for i in range(len(regions)):
            text_boxes = regions[i]["text_boxes"]
            empty_cell_flag = regions[i]["empty_cell"]
            bound = regions[i]["bound"]
            if len(text_boxes) == 1 or empty_cell_flag:
                new_regions_info.append(regions[i])
                continue

            boxes_rel = (
                {}
            )  # {'index of box': {'same_cell': [index of boxes|int], 'same_row': [index of boxes|int}, 'same_col': [index of boxes|int]}
            for n in range(len(text_boxes)):
                box1 = text_boxes[n]
                boxes_rel[str(n)] = {"same_cell": [], "same_row": [], "same_col": []}
                for m in range(n + 1, len(text_boxes)):
                    box2 = text_boxes[m]
                    rel = self.get_boxes_rel(box1=box1, box2=box2, res_boxes=text_boxes)
                    if rel == 0:
                        boxes_rel[str(n)]["same_cell"].append(m)
                    elif rel == 1:
                        boxes_rel[str(n)]["same_row"].append(m)
                    elif rel == 2:
                        boxes_rel[str(n)]["same_col"].append(m)
                    elif rel == 3:
                        pass
                    if rel != 0:
                        dis, _ = self.cal_box_dis(box1, box2)
                        if dis < 0.2 * dis_thresh:
                            boxes_rel[str(n)]["same_cell"].append(m)

            # same cell merge
            same_cells_indexes = []
            for j in boxes_rel.keys():
                if len(boxes_rel[j]["same_cell"]) == 0:
                    continue

                repeat_guard = False
                cur_box = text_boxes[int(j)]
                # check repeat
                for k in range(len(same_cells_indexes)):
                    if int(j) in same_cells_indexes[k]:
                        repeat_guard = True
                        continue
                if repeat_guard:
                    continue

                same_cell_idxs = boxes_rel[j]["same_cell"]
                checked_same_cell_idxes = []
                for w in same_cell_idxs:
                    tmp_box = text_boxes[w]
                    dis, _ = self.cal_box_dis(cur_box, tmp_box)
                    if dis > dis_thresh:
                        continue
                    checked_same_cell_idxes.append(w)
                same_cell_idxs = checked_same_cell_idxes

                add_idxs = []
                for idx in same_cell_idxs:
                    tmp_idxes = boxes_rel[str(idx)]["same_cell"]
                    for _idx in tmp_idxes:
                        if _idx in same_cell_idxs:
                            continue
                        tmp_box = text_boxes[_idx]
                        dis, _ = self.cal_box_dis(cur_box, tmp_box)
                        if dis > dis_thresh:
                            continue
                        add_idxs.append(idx)
                complete_same_cell_idxs = same_cell_idxs + add_idxs + [int(j)]

                if len(complete_same_cell_idxs) == 1:
                    continue
                same_cells_indexes.append(complete_same_cell_idxs)

            # merge same cell text boxes
            cell_region_info = (
                []
            )  # [{'bound': [x1, y1, x2, y2], 'text_boxes_idxs': [], 'text_boxes': [x, y, w, h]}]
            deal_text_boxes_idxs = []
            for q in range(len(same_cells_indexes)):
                cell_boxes_indexes = same_cells_indexes[q]
                x1 = 1e5
                y1 = 1e5
                x2 = -1
                y2 = -1
                same_cells_text_boxes = []
                for p in cell_boxes_indexes:
                    tmp_x1, tmp_y1, tmp_w, tmp_h = text_boxes[p]
                    tmp_x2 = tmp_x1 + tmp_w
                    tmp_y2 = tmp_y1 + tmp_h
                    x1 = min(x1, tmp_x1)
                    x2 = max(x2, tmp_x2)
                    y1 = min(y1, tmp_y1)
                    y2 = max(y2, tmp_y2)

                merged_cell_info = {
                    "bound": (x1, y1, x2, y2),
                    "box": [x1, y1, x2 - x1, y2 - y1],
                    "text_boxes_idxs": cell_boxes_indexes,
                    "text_boxes": same_cells_text_boxes,
                    "empty_cell": 0,
                }
                cell_region_info.append(merged_cell_info)
                deal_text_boxes_idxs.extend(cell_boxes_indexes)

            # deal the remained text_boxes
            for q in range(len(text_boxes)):
                if q in deal_text_boxes_idxs:
                    continue
                _x, _y, _w, _h = text_boxes[q]
                text_bound = (_x, _y, _x + _w, _y + _h)
                cell_region_info.append(
                    {
                        "bound": text_bound,
                        "box": text_boxes[q],
                        "text_boxes_idxs": [q],
                        "text_boxes": [text_boxes[q]],
                        "empty_cell": 0,
                    }
                )

            # deal overlap
            deal_overlap_cell_region = []
            overlap_indexes = []
            for a in range(len(cell_region_info)):
                bound_a = cell_region_info[a]["bound"]
                if a in overlap_indexes:
                    continue
                append_guard = True
                for b in range(a + 1, len(cell_region_info)):
                    if b in overlap_indexes:
                        continue
                    bound_b = cell_region_info[b]["bound"]
                    iou = max(
                        compute_iou(bound_a, bound_b), compute_iou(bound_b, bound_a)
                    )
                    if iou > 0.3:
                        overlap_indexes.append(b)
                        cell_region_info[a]["text_boxes"].extend(
                            cell_region_info[b]["text_boxes"]
                        )
                        new_bound = (
                            min(bound_a[0], bound_b[0]),
                            min(bound_a[1], bound_b[1]),
                            max(bound_a[2], bound_b[2]),
                            max(bound_b[3], bound_a[3]),
                        )
                        new_text_boxes_idxs = cell_region_info[a][
                            "text_boxes_idxs"
                        ].extend(cell_region_info[b]["text_boxes_idxs"])
                        new_text_boxes = (
                            cell_region_info[a]["text_boxes"]
                            + cell_region_info[b]["text_boxes"]
                        )
                        new_cell_region = {
                            "bound": new_bound,
                            "box": (
                                new_bound[0],
                                new_bound[1],
                                new_bound[2] - new_bound[0],
                                new_bound[3] - new_bound[1],
                            ),
                            "text_boxes_idxs": new_text_boxes_idxs,
                            "text_boxes": new_text_boxes,
                            "empty_cell": 0,
                        }
                        deal_overlap_cell_region.append(new_cell_region)
                        append_guard = False
                        continue
                if append_guard:
                    deal_overlap_cell_region.append(cell_region_info[a])

            cell_region_info = deal_overlap_cell_region

            if len(cell_region_info) == 1:
                cell_region_info[0]["bound"] = bound
                new_regions_info.extend(cell_region_info)
                continue

            # deal complex region
            cell_region_info = self.split_complex_region(
                cell_region_info=cell_region_info, bound=bound, img=img, iou_thresh=0.6
            )

            new_regions_info.extend(cell_region_info)
            continue
        return new_regions_info

    def split_complex_region(
        self, cell_region_info: List, bound: List, img: np.ndarray, iou_thresh=0.6
    ):
        """split complex region into cells

        Args:
            cell_region_info (List): [{'bound': [x1, y1, x2, y2], 'text_boxes': [x, y, w, h], 'empty_cell': 0, 'box': [x1, y1, x2, y2], 'text_boxes_idxs': cell_boxes_indexes}]
            bound (List): bound of the complex region [x1, y1, x2, y2]
            img (np.ndarray): origin image
            iou_thresh (float, optional): _description_. Defaults to 0.6.

        Returns:
            new_regions_info (List): list of dict, [{'bound': [x1, y1, x2, y2], 'text_boxes': [[x1, y1, w, h], ...], 'empty_cell': 0|1, ...]
        """
        new_regions_info = []  # the same structure as cell_region_info
        # analyze row
        # #transform coordinate from absolute to relative
        q = 0
        inside_region_cell_boxes = []
        for q in cell_region_info:
            _x, _y, _x2, _y2 = q["bound"]
            _w = _x2 - _x
            _h = _y2 - _y
            inside_region_cell_boxes.append([_x - bound[0], _y - bound[1], _w, _h])

        img_shape = img[bound[1] : bound[3], bound[0] : bound[2]].shape
        inside_region_canvas = self.ocr_box_canvas(
            text_boxes=inside_region_cell_boxes, img_shape=img_shape
        )

        row_subgraphs = self.row_analyse(canvas=inside_region_canvas)
        # #transform coordinate from relative to absolute
        for s in range(len(row_subgraphs.keys())):
            q = list(row_subgraphs.keys())[s]
            if s == len(row_subgraphs.keys()) - 1:
                row_subgraphs[q]["scope"] = [
                    row_subgraphs[q]["scope"][0] + bound[1],
                    bound[3],
                ]
            else:
                row_subgraphs[q]["scope"] = [
                    row_subgraphs[q]["scope"][0] + bound[1],
                    row_subgraphs[q]["scope"][1] + bound[1],
                ]

        # case 1
        if len(row_subgraphs) == 1:
            sorted_cell_region_info = list(
                sorted(cell_region_info, key=lambda x: x["bound"][0])
            )
            # fresh bound
            new_bound_cell_region_info = self.fresh_bound(
                sorted_cell_region_info=sorted_cell_region_info, bound=bound
            )
            new_regions_info.extend(new_bound_cell_region_info)
            return new_regions_info
        else:
            # split rows
            for p in row_subgraphs.keys():
                subgraph_scope = row_subgraphs[p]["scope"]
                row_subgraphs[p][
                    "cells_info"
                ] = (
                    []
                )  # [{'cell_idx': int, 'bound': [x1, y1, x2, y2], 'cell_box': [x, y, w, h], 'text_boxes_idxs': list of int|indexes of text boxes which belong to the same cell, 'text_boxes': []}]
                row_subgraphs[p]["bound"] = [
                    bound[0],
                    subgraph_scope[0],
                    bound[2],
                    subgraph_scope[1],
                ]
                for q in range(len(cell_region_info)):
                    cell_bound = cell_region_info[q]["bound"]
                    x, y, x1, y1 = cell_bound
                    h = y1 - y
                    if y >= subgraph_scope[0] and y + h <= subgraph_scope[1]:
                        row_subgraphs[p]["cells_info"].append(
                            {
                                "cell_idx": q,
                                "bound": cell_region_info[q]["bound"],
                                "text_boxes_idxs": cell_region_info[q][
                                    "text_boxes_idxs"
                                ],
                                "text_boxes": cell_region_info[q]["text_boxes"],
                            }
                        )
                    elif y + h <= subgraph_scope[0] or y >= subgraph_scope[1]:
                        pass
                    else:
                        iou = 0
                        if (
                            y >= subgraph_scope[0]
                            and y < subgraph_scope[1]
                            and y + h > subgraph_scope[1]
                        ):
                            iou = round((subgraph_scope[1] - y) / h, 2)
                        elif (
                            y < subgraph_scope[0]
                            and y + h > subgraph_scope[0]
                            and y + h <= subgraph_scope[1]
                        ):
                            iou = round((y + h - subgraph_scope[0]) / h, 2)
                        elif y <= subgraph_scope[0] and y + h >= subgraph_scope[1]:
                            iou = round((subgraph_scope[1] - subgraph_scope[0]) / h, 2)
                        if iou >= iou_thresh:
                            row_subgraphs[p]["cells_info"].append(
                                {
                                    "cell_idx": q,
                                    "bound": cell_region_info[q]["bound"],
                                    "text_boxes_idxs": cell_region_info[q][
                                        "text_boxes_idxs"
                                    ],
                                    "text_boxes": cell_region_info[q]["text_boxes"],
                                }
                            )

            # get new split cells
            for k in range(len(row_subgraphs.keys())):
                q = list(row_subgraphs.keys())[k]
                tmp_row_info = row_subgraphs[q]
                if len(tmp_row_info["cells_info"]) == 1:
                    tmp_row_info["cells_info"][0]["bound"] = tmp_row_info["bound"]
                    new_regions_info.append(tmp_row_info["cells_info"][0])
                    continue

                row_subgraphs[p]["cells_info"] = list(
                    sorted(tmp_row_info["cells_info"], key=lambda x: x["bound"][0])
                )
                row_bound = row_subgraphs[p]["bound"]
                row_cells_info = row_subgraphs[p]["cells_info"]

                if len(row_cells_info) == 0:
                    continue

                # refresh the bound
                new_split_row_cell_info = self.fresh_bound(
                    sorted_cell_region_info=row_cells_info, bound=row_bound
                )

                new_regions_info.extend(new_split_row_cell_info)
                continue

        return new_regions_info

    def fresh_bound(self, sorted_cell_region_info, bound):
        q = 0
        x_st = bound[0]
        new_bound_cell_region_info = []
        for q in range(len(sorted_cell_region_info) - 1):
            cur_bound = sorted_cell_region_info[q]["bound"]
            next_bound = sorted_cell_region_info[q + 1]["bound"]
            x_ed = 0.5 * (cur_bound[2] + next_bound[0])
            sorted_cell_region_info[q]["bound"] = [x_st, bound[1], x_ed, bound[3]]
            new_bound_cell_region_info.append(sorted_cell_region_info[q])
            x_st = x_ed

        sorted_cell_region_info[-1]["bound"] = [x_st, bound[1], bound[2], bound[3]]
        new_bound_cell_region_info.append(sorted_cell_region_info[-1])
        return new_bound_cell_region_info

    def row_analyse(self, canvas: np.ndarray):
        row_proj = []
        for i in range(canvas.shape[0]):
            row_proj.append(np.sum(canvas[i, :]))

        row_subgraphs = {}

        row_st = -1
        for j in range(canvas.shape[0]):
            if row_proj[j] > 0 and row_st == -1:
                row_st = j
            elif row_proj[j] == 0 and row_st != -1:
                row_subgraphs[str(len(row_subgraphs))] = {
                    "scope": [row_st, j - 1],
                    "text_boxes": [],
                    "cells_info": [],
                }
                row_st = -1

        if row_st != -1:
            row_subgraphs[str(len(row_subgraphs))] = {
                "scope": [row_st, canvas.shape[0] - 1],
                "text_boxes": [],
                "cells_info": [],
            }

        correct_st = 0
        for k in range(len(row_subgraphs) - 1):
            correct_ed = int(
                (
                    row_subgraphs[str(k)]["scope"][1]
                    + row_subgraphs[str(k + 1)]["scope"][0]
                )
                / 2
            )
            row_subgraphs[str(k)]["scope"] = [correct_st, correct_ed]
            correct_st = correct_ed

        row_subgraphs[list(row_subgraphs.keys())[-1]]["scope"] = [
            correct_st,
            row_subgraphs[list(row_subgraphs.keys())[-1]]["scope"][1],
        ]

        return row_subgraphs

    def col_analyse(self, canvas: np.ndarray):
        col_proj = []
        for k in range(canvas.shape[1]):
            col_proj.append(np.sum(canvas[:, k]))

        col_subgraphs = {}

        col_st = -1
        for q in range(canvas.shape[1]):
            if col_proj[q] > 0 and col_st == -1:
                col_st = q
            elif col_proj[q] == 0 and col_st != -1:
                col_subgraphs[str(len(col_subgraphs))] = {
                    "scope": [col_st, q - 1],
                    "text_boxes": [],
                }
                col_st = -1

        if col_st != -1:
            col_subgraphs[str(len(col_subgraphs))] = {
                "scope": [col_st, canvas.shape[1] - 1],
                "text_boxes": [],
            }
        return col_subgraphs

    def split_into_subgraph(
        self,
        canvas: np.ndarray,
        text_boxes: List,
        img: np.ndarray = None,
        iou_thresh=0.6,
    ):
        """split text boxes into subgraphs

        Args:
            canvas (np.ndarray): blank matrix with text box as 1
            text_boxes (List): the small text boxes, [[x, y, w, h], ...]
            img (np.ndarray): for debug
        Returns:
            box_groups (List): list of group of text boxes
        """
        row_proj = []
        for i in range(canvas.shape[0]):
            row_proj.append(np.sum(canvas[i, :]))

        row_subgraphs = {}

        row_st = -1
        for j in range(canvas.shape[0]):
            if row_proj[j] > 0 and row_st == -1:
                row_st = j
            elif row_proj[j] == 0 and row_st != -1:
                row_subgraphs[str(len(row_subgraphs))] = {
                    "scope": [row_st, j - 1],
                    "text_boxes": [],
                }
                row_st = -1

        if row_st != -1:
            row_subgraphs[str(len(row_subgraphs))] = {
                "scope": [row_st, canvas.shape[0] - 1],
                "text_boxes": [],
            }

        col_proj = []
        for k in range(canvas.shape[1]):
            col_proj.append(np.sum(canvas[:, k]))

        col_subgraphs = {}

        col_st = -1
        for q in range(canvas.shape[1]):
            if col_proj[q] > 0 and col_st == -1:
                col_st = q
            elif col_proj[q] == 0 and col_st != -1:
                col_subgraphs[str(len(col_subgraphs))] = {
                    "scope": [col_st, q - 1],
                    "text_boxes": [],
                }
                col_st = -1

        if col_st != -1:
            col_subgraphs[str(len(col_subgraphs))] = {
                "scope": [col_st, canvas.shape[1] - 1],
                "text_boxes": [],
            }

        for box in text_boxes:
            x, y, w, h = box
            for n in range(len(row_subgraphs)):
                subgraph_scope = row_subgraphs[str(n)]["scope"]
                if y >= subgraph_scope[0] and y + h <= subgraph_scope[1]:
                    row_subgraphs[str(n)]["text_boxes"].append(box)
                elif y + h <= subgraph_scope[0] or y >= subgraph_scope[1]:
                    pass
                else:
                    iou = 0
                    if (
                        y >= subgraph_scope[0]
                        and y < subgraph_scope[1]
                        and y + h > subgraph_scope[1]
                    ):
                        iou = round((subgraph_scope[1] - y) / h, 2)
                    elif (
                        y < subgraph_scope[0]
                        and y + h > subgraph_scope[0]
                        and y + h <= subgraph_scope[1]
                    ):
                        iou = round((y + h - subgraph_scope[0]) / h, 2)
                    elif y <= subgraph_scope[0] and y + h >= subgraph_scope[1]:
                        iou = round((subgraph_scope[1] - subgraph_scope[0]) / h, 2)
                    if iou >= iou_thresh:
                        row_subgraphs[str(n)]["text_boxes"].append(box)

            for m in range(len(col_subgraphs)):
                subgraph_scope = col_subgraphs[str(m)]["scope"]
                if x >= subgraph_scope[0] and x + h <= subgraph_scope[1]:
                    col_subgraphs[str(m)]["text_boxes"].append(box)
                elif x + h <= subgraph_scope[0] or x >= subgraph_scope[1]:
                    pass
                else:
                    iou = 0
                    if (
                        x >= subgraph_scope[0]
                        and x < subgraph_scope[1]
                        and x + w > subgraph_scope[1]
                    ):
                        iou = round((subgraph_scope[1] - x) / w, 2)
                    elif (
                        x < subgraph_scope[0]
                        and x + w > subgraph_scope[0]
                        and x + w <= subgraph_scope[1]
                    ):
                        iou = round((x + w - subgraph_scope[0]) / w, 2)
                    elif x <= subgraph_scope[0] and x + w > subgraph_scope[1]:
                        iou = round((subgraph_scope[1] - subgraph_scope[0]) / w, 2)
                    if iou >= iou_thresh:
                        col_subgraphs[str(m)]["text_boxes"].append(box)

        # deal some error row
        i = 0
        for i in row_subgraphs.keys():
            if len(row_subgraphs[i]["text_boxes"]) > 1:
                continue
            y1, y2 = row_subgraphs[i]["scope"]
            if int(i) == 0:
                continue
            elif int(i) == len(row_subgraphs) - 1:
                row_subgraphs[str(int(i) - 1)]["text_boxes"].extend(
                    row_subgraphs[i]["text_boxes"]
                )
                y21, y22 = row_subgraphs[str(int(i) - 1)]["scope"]
                row_subgraphs[str(int(i) - 1)]["scope"] = [y21, y2]
            else:
                yl1, yl2 = row_subgraphs[str(int(i) - 1)]["scope"]
                yr1, yr2 = row_subgraphs[str(int(i) + 1)]["scope"]
                if y1 - yl2 <= yr1 - y2:
                    row_subgraphs[str(int(i) - 1)]["text_boxes"].extend(
                        row_subgraphs[i]["text_boxes"]
                    )
                    row_subgraphs[str(int(i) - 1)]["scope"] = [yl1, y2]
                else:
                    row_subgraphs[str(int(i) + 1)]["text_boxes"].extend(
                        row_subgraphs[i]["text_boxes"]
                    )
                    row_subgraphs[str(int(i) + 1)]["scope"] = [y1, yr2]
            row_subgraphs[i]["text_boxes"] = []
        row_subgraphs = {
            n: row_subgraphs[n]
            for n in row_subgraphs.keys()
            if row_subgraphs[n]["text_boxes"] != []
        }

        # deal some error col
        i = 0
        for i in col_subgraphs.keys():
            if len(col_subgraphs[i]["text_boxes"]) > 1:
                continue
            x1, x2 = col_subgraphs[i]["scope"]
            if int(i) == 0:
                continue
            elif int(i) == len(col_subgraphs) - 1:
                col_subgraphs[str(int(i) - 1)]["text_boxes"].extend(
                    col_subgraphs[i]["text_boxes"]
                )
                x21, x22 = col_subgraphs[str(int(i) - 1)]["scope"]
                col_subgraphs[str(int(i) - 1)]["scope"] = [x21, x2]
            else:
                xl1, xl2 = col_subgraphs[str(int(i) - 1)]["scope"]
                xr1, xr2 = col_subgraphs[str(int(i) + 1)]["scope"]
                if x1 - xl2 <= xr1 - x2:
                    col_subgraphs[str(int(i) - 1)]["text_boxes"].extend(
                        col_subgraphs[i]["text_boxes"]
                    )
                    col_subgraphs[str(int(i) - 1)]["scope"] = [xl1, x2]
                else:
                    col_subgraphs[str(int(i) + 1)]["text_boxes"].extend(
                        col_subgraphs[i]["text_boxes"]
                    )
                    col_subgraphs[str(int(i) + 1)]["scope"] = [x1, xr2]
            col_subgraphs[i]["text_boxes"] = []

        col_subgraphs = {
            n: col_subgraphs[n]
            for n in col_subgraphs.keys()
            if col_subgraphs[n]["text_boxes"] != []
        }

        subgraphs = {"row": row_subgraphs, "col": col_subgraphs}
        return subgraphs

    def modify_text_boxes(
        self, text_box: List, box_img: np.ndarray, col_split_percent=90
    ):
        """correct the results of ocr det model

        Args:
            text_boxes (List): the text boxes,  [x, y, w, h]
            box_img (np.ndarray): the text box image
        Returns:
            new_text_boxes (List): the text boxes
            new_box_img (np.ndarray): the binary image
        """
        _, bin_box_img = cv2.threshold(box_img, 127, 1, cv2.THRESH_BINARY)
        bin_box_img = 1 - bin_box_img
        x, y, w, h = text_box
        hor_proj = []
        for i in range(bin_box_img.shape[1]):
            hor_proj.append(np.sum(bin_box_img[:, i]))

        margins = []
        st = -1
        for j in range(len(hor_proj)):
            if j == 0 and hor_proj[j] == 0:
                continue
            if hor_proj[j] == 0 and hor_proj[j - 1] != 0:
                st = j
            elif hor_proj[j] != 0 and st != -1 and j - 1 - st > 0:
                margins.append({"scope": [st, j - 1], "length": j - 1 - st})
                st = -1

        if len(margins) <= 3:
            return [text_box], [box_img]

        median_thresh = np.percentile(
            [_d["length"] for _d in margins], col_split_percent
        )
        bonds = []
        for k in margins:
            if k["length"] > median_thresh:
                bonds.append(k["scope"])

        if len(bonds) == 0:
            return [text_box], [box_img]

        new_st = x
        new_text_boxes = []
        new_box_imgs = []
        for n in range(len(bonds)):
            if n == 0:
                if bonds[n][0] > 1 and h > 1:
                    new_text_boxes.append([new_st, y, bonds[n][0], h])
                    tmp_box_img = box_img[0:h, new_st - x : new_st - x + bonds[n][0]]
                else:
                    tmp_box_img = []
            else:
                if bonds[n][0] - bonds[n - 1][1] > 1 and h > 1:
                    new_text_boxes.append([new_st, y, bonds[n][0] - bonds[n - 1][1], h])
                    tmp_box_img = box_img[
                        0:h, new_st - x : new_st - x + bonds[n][0] - bonds[n - 1][0]
                    ]
                else:
                    tmp_box_img = []

            if tmp_box_img != []:
                new_box_imgs.append(tmp_box_img)
            new_st = x + bonds[n][1]

        if new_st != len(hor_proj) - 1:
            if x + w - new_st > 1 and h > 1:
                new_text_boxes.append([new_st, y, x + w - new_st, h])
                tmp_box_img = box_img[0:h, new_st - x : w]
                assert (
                    tmp_box_img != []
                ), f"[0:h, new_st-x: w]: [0:{h}, {new_st-x}: {w}], box_img: {box_img.shape}, new_st: {new_st}, w: {w}"
                new_box_imgs.append(tmp_box_img)

        return new_text_boxes, new_box_imgs

    def get_boxes_rel(self, box1: list, box2: list, res_boxes):
        """predict the relationship between two boxes

        Args:
            box1 (list): [x, y, w, h]
            box2 (list): [x, y, w, h]

        Returns:
            int: 0: same cell, 1: same row, 2: same col, 3: same no relation
        """
        ws = []
        hs = []
        for _box in res_boxes:
            ws.append(_box[2])
            hs.append(_box[3])
        median_w = np.median(ws)
        median_h = np.median(hs)

        x1, y1, w1, h1 = box1
        x12 = x1 + w1
        y12 = y1 + h1
        core_x1 = x1 + 0.5 * w1
        core_y1 = y1 + 0.5 * h1

        x2, y2, w2, h2 = box2
        x22 = x2 + w2
        y22 = y2 + h2
        core_x2 = x2 + 0.5 * w2
        core_y2 = y2 + 0.5 * h2

        core_x_diff = round((core_x1 - core_x2) / median_w, 4)
        core_y_diff = round((core_y1 - core_y2) / median_h, 4)
        lt_x_diff = round((x1 - x2) / median_w, 4)
        br_x_diff = round((x12 - x22) / median_w, 4)
        lt_y_diff = round((y1 - y2) / median_h, 4)
        br_y_diff = round((y12 - y22) / median_h, 4)
        w_diff = round((w1 - w2) / median_w, 4)
        h_diff = round((h1 - h2) / median_h, 4)

        x_data = [
            {
                "core_x_diff": core_x_diff,
                "core_y_diff": core_y_diff,
                "lt_x_diff": lt_x_diff,
                "br_x_diff": br_x_diff,
                "lt_y_diff": lt_y_diff,
                "br_y_diff": br_y_diff,
                "w_diff": w_diff,
                "h_diff": h_diff,
            }
        ]
        x_df = pd.DataFrame(
            data=x_data,
            columns=[
                "core_x_diff",
                "core_y_diff",
                "lt_x_diff",
                "br_x_diff",
                "lt_y_diff",
                "br_y_diff",
                "w_diff",
                "h_diff",
            ],
        )

        y = self.model.predict(x_df)[0]

        return y

    def cal_box_dis(self, box1: List, box2: List):
        """calculate the Euclidean distance between core of box

        Args:
            box1 (List): [x1, y1, w1, h1]
            box2 (List): [x2, y2, w2, h2]

        Returns:
            float: Euclidean distance

            Tuple: closest point pair
        """
        # pts on box1
        row_samples = np.linspace(box1[1], box1[1] + box1[3], num=10)
        col_samples = np.linspace(box1[0], box1[0] + box1[2], num=10)
        pts1 = []
        for i in row_samples:
            for j in col_samples:
                pts1.append((i, j))

        # pts on box2
        row_samples = np.linspace(box2[1], box2[1] + box2[3], num=4)
        col_samples = np.linspace(box2[0], box2[0] + box2[2], num=4)
        pts2 = []
        for i in row_samples:
            for j in col_samples:
                pts2.append((i, j))

        min_distance = float("inf")
        closest_pair = (None, None)

        for point1 in pts1:
            for point2 in pts2:
                distance = euclidean_distance(point1, point2)
                if distance < min_distance:
                    min_distance = distance
                    closest_pair = (point1, point2)

        return min_distance, closest_pair


def compute_iou(box1, box2):
    """
    Compute the Intersection over Union (IoU) between two rectangles.

    Args:
        box1 (array-like): [x1, y1, x2, y2] of the first rectangle.
        box2 (array-like): [x1, y1, x2, y2] of the second rectangle.

    Returns:
        float: The IoU between the two rectangles.
    """
    # Determine the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    # Calculate the area of intersection rectangle
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    # Calculate the area of both rectangles
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    # Calculate the IoU
    iou = intersection_area / float(box1_area)
    return iou


def euclidean_distance(point1, point2):
    """Calculate the Euclidean distance between two points
    Args:
        point1 (Tuple|List|Array): (x1, y1)
        point2 (Tuple|List|Array): (x2, y2)

    Returns:
        float: Euclidean distance
    """
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))
