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

import numpy as np
import re
import copy

__all__ = [
    "TableMatch",
    "convert_4point2rect",
    "get_ori_coordinate_for_table",
    "is_inside",
]


def deal_eb_token(master_token):
    """
    post process with <eb></eb>, <eb1></eb1>, ...
    emptyBboxTokenDict = {
        "[]": '<eb></eb>',
        "[' ']": '<eb1></eb1>',
        "['<b>', ' ', '</b>']": '<eb2></eb2>',
        "['\\u2028', '\\u2028']": '<eb3></eb3>',
        "['<sup>', ' ', '</sup>']": '<eb4></eb4>',
        "['<b>', '</b>']": '<eb5></eb5>',
        "['<i>', ' ', '</i>']": '<eb6></eb6>',
        "['<b>', '<i>', '</i>', '</b>']": '<eb7></eb7>',
        "['<b>', '<i>', ' ', '</i>', '</b>']": '<eb8></eb8>',
        "['<i>', '</i>']": '<eb9></eb9>',
        "['<b>', ' ', '\\u2028', ' ', '\\u2028', ' ', '</b>']": '<eb10></eb10>',
    }
    :param master_token:
    :return:
    """
    master_token = master_token.replace("<eb></eb>", "<td></td>")
    master_token = master_token.replace("<eb1></eb1>", "<td> </td>")
    master_token = master_token.replace("<eb2></eb2>", "<td><b> </b></td>")
    master_token = master_token.replace("<eb3></eb3>", "<td>\u2028\u2028</td>")
    master_token = master_token.replace("<eb4></eb4>", "<td><sup> </sup></td>")
    master_token = master_token.replace("<eb5></eb5>", "<td><b></b></td>")
    master_token = master_token.replace("<eb6></eb6>", "<td><i> </i></td>")
    master_token = master_token.replace("<eb7></eb7>", "<td><b><i></i></b></td>")
    master_token = master_token.replace("<eb8></eb8>", "<td><b><i> </i></b></td>")
    master_token = master_token.replace("<eb9></eb9>", "<td><i></i></td>")
    master_token = master_token.replace(
        "<eb10></eb10>", "<td><b> \u2028 \u2028 </b></td>"
    )
    return master_token


def deal_bb(result_token):
    """
    In our opinion, <b></b> always occurs in <thead></thead> text's context.
    This function will find out all tokens in <thead></thead> and insert <b></b> by manual.
    :param result_token:
    :return:
    """
    # find out <thead></thead> parts.
    thead_pattern = "<thead>(.*?)</thead>"
    if re.search(thead_pattern, result_token) is None:
        return result_token
    thead_part = re.search(thead_pattern, result_token).group()
    origin_thead_part = copy.deepcopy(thead_part)

    # check "rowspan" or "colspan" occur in <thead></thead> parts or not .
    span_pattern = (
        '<td rowspan="(\d)+" colspan="(\d)+">|<td colspan="(\d)+" rowspan="(\d)+">|<td rowspan'
        '="(\d)+">|<td colspan="(\d)+">'
    )
    span_iter = re.finditer(span_pattern, thead_part)
    span_list = [s.group() for s in span_iter]
    has_span_in_head = True if len(span_list) > 0 else False

    if not has_span_in_head:
        # <thead></thead> not include "rowspan" or "colspan" branch 1.
        # 1. replace <td> to <td><b>, and </td> to </b></td>
        # 2. it is possible to predict text include <b> or </b> by Text-line recognition,
        #    so we replace <b><b> to <b>, and </b></b> to </b>
        thead_part = (
            thead_part.replace("<td>", "<td><b>")
            .replace("</td>", "</b></td>")
            .replace("<b><b>", "<b>")
            .replace("</b></b>", "</b>")
        )
    else:
        # <thead></thead> include "rowspan" or "colspan" branch 2.
        # Firstly, we deal rowspan or colspan cases.
        # 1. replace > to ><b>
        # 2. replace </td> to </b></td>
        # 3. it is possible to predict text include <b> or </b> by Text-line recognition,
        #    so we replace <b><b> to <b>, and </b><b> to </b>

        # Secondly, deal ordinary cases like branch 1

        # replace ">" to "<b>"
        replaced_span_list = []
        for sp in span_list:
            replaced_span_list.append(sp.replace(">", "><b>"))
        for sp, rsp in zip(span_list, replaced_span_list):
            thead_part = thead_part.replace(sp, rsp)

        # replace "</td>" to "</b></td>"
        thead_part = thead_part.replace("</td>", "</b></td>")

        # remove duplicated <b> by re.sub
        mb_pattern = "(<b>)+"
        single_b_string = "<b>"
        thead_part = re.sub(mb_pattern, single_b_string, thead_part)

        mgb_pattern = "(</b>)+"
        single_gb_string = "</b>"
        thead_part = re.sub(mgb_pattern, single_gb_string, thead_part)

        # ordinary cases like branch 1
        thead_part = thead_part.replace("<td>", "<td><b>").replace("<b><b>", "<b>")

    # convert <tb><b></b></tb> back to <tb></tb>, empty cell has no <b></b>.
    # but space cell(<tb> </tb>)  is suitable for <td><b> </b></td>
    thead_part = thead_part.replace("<td><b></b></td>", "<td></td>")
    # deal with duplicated <b></b>
    thead_part = deal_duplicate_bb(thead_part)
    # deal with isolate span tokens, which causes by wrong predict by structure prediction.
    # eg.PMC5994107_011_00.png
    thead_part = deal_isolate_span(thead_part)
    # replace original result with new thead part.
    result_token = result_token.replace(origin_thead_part, thead_part)
    return result_token


def deal_isolate_span(thead_part):
    """
    Deal with isolate span cases in this function.
    It causes by wrong prediction in structure recognition model.
    eg. predict <td rowspan="2"></td> to <td></td> rowspan="2"></b></td>.
    :param thead_part:
    :return:
    """
    # 1. find out isolate span tokens.
    isolate_pattern = (
        '<td></td> rowspan="(\d)+" colspan="(\d)+"></b></td>|'
        '<td></td> colspan="(\d)+" rowspan="(\d)+"></b></td>|'
        '<td></td> rowspan="(\d)+"></b></td>|'
        '<td></td> colspan="(\d)+"></b></td>'
    )
    isolate_iter = re.finditer(isolate_pattern, thead_part)
    isolate_list = [i.group() for i in isolate_iter]

    # 2. find out span number, by step 1 results.
    span_pattern = (
        ' rowspan="(\d)+" colspan="(\d)+"|'
        ' colspan="(\d)+" rowspan="(\d)+"|'
        ' rowspan="(\d)+"|'
        ' colspan="(\d)+"'
    )
    corrected_list = []
    for isolate_item in isolate_list:
        span_part = re.search(span_pattern, isolate_item)
        spanStr_in_isolateItem = span_part.group()
        # 3. merge the span number into the span token format string.
        if spanStr_in_isolateItem is not None:
            corrected_item = "<td{}></td>".format(spanStr_in_isolateItem)
            corrected_list.append(corrected_item)
        else:
            corrected_list.append(None)

    # 4. replace original isolated token.
    for corrected_item, isolate_item in zip(corrected_list, isolate_list):
        if corrected_item is not None:
            thead_part = thead_part.replace(isolate_item, corrected_item)
        else:
            pass
    return thead_part


def deal_duplicate_bb(thead_part):
    """
    Deal duplicate <b> or </b> after replace.
    Keep one <b></b> in a <td></td> token.
    :param thead_part:
    :return:
    """
    # 1. find out <td></td> in <thead></thead>.
    td_pattern = (
        '<td rowspan="(\d)+" colspan="(\d)+">(.+?)</td>|'
        '<td colspan="(\d)+" rowspan="(\d)+">(.+?)</td>|'
        '<td rowspan="(\d)+">(.+?)</td>|'
        '<td colspan="(\d)+">(.+?)</td>|'
        "<td>(.*?)</td>"
    )
    td_iter = re.finditer(td_pattern, thead_part)
    td_list = [t.group() for t in td_iter]

    # 2. is multiply <b></b> in <td></td> or not?
    new_td_list = []
    for td_item in td_list:
        if td_item.count("<b>") > 1 or td_item.count("</b>") > 1:
            # multiply <b></b> in <td></td> case.
            # 1. remove all <b></b>
            td_item = td_item.replace("<b>", "").replace("</b>", "")
            # 2. replace <tb> -> <tb><b>, </tb> -> </b></tb>.
            td_item = td_item.replace("<td>", "<td><b>").replace("</td>", "</b></td>")
            new_td_list.append(td_item)
        else:
            new_td_list.append(td_item)

    # 3. replace original thead part.
    for td_item, new_td_item in zip(td_list, new_td_list):
        thead_part = thead_part.replace(td_item, new_td_item)
    return thead_part


def distance(box_1, box_2):
    """
    compute the distance between two boxes

    Args:
        box_1 (list): first rectangle box,eg.(x1, y1, x2, y2)
        box_2 (list): second rectangle box,eg.(x1, y1, x2, y2)

    Returns:
        int: the distance between two boxes

    """
    x1, y1, x2, y2 = box_1
    x3, y3, x4, y4 = box_2
    dis = abs(x3 - x1) + abs(y3 - y1) + abs(x4 - x2) + abs(y4 - y2)
    dis_2 = abs(x3 - x1) + abs(y3 - y1)
    dis_3 = abs(x4 - x2) + abs(y4 - y2)
    return dis + min(dis_2, dis_3)


def compute_iou(rec1, rec2):
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


def convert_4point2rect(bbox):
    """
    Convert 4 point coordinate to rectangle coordinate
    Args:
        bbox (list): list of 4 points, eg. [x1, y1, x2, y2,...] or [[x1,y1],[x2,y2],...]
    """
    if isinstance(bbox, list):
        bbox = np.array(bbox)
    if bbox.shape[0] == 8:
        bbox = np.reshape(bbox, (4, 2))
    x1 = min(bbox[:, 0])
    y1 = min(bbox[:, 1])
    x2 = max(bbox[:, 0])
    y2 = max(bbox[:, 1])
    return [x1, y1, x2, y2]


def get_ori_coordinate_for_table(x, y, table_bbox):
    """
    get the original coordinate from Cropped image to Original image.
    Args:
        x (int): x coordinate of cropped image
        y (int): y coordinate of cropped image
        table_bbox (list): list of table bounding boxes, eg. [[x1, y1, x2, y2, x3, y3, x4, y4]]
    Returns:
        list: list of original coordinates, eg. [[x1, y1, x2, y2, x3, y3, x4, y4]]
    """
    bbox_list = []
    for x1, y1, x2, y2, x3, y3, x4, y4 in table_bbox:
        x1 = x + x1
        y1 = y + y1
        x2 = x + x2
        y2 = y + y2
        x3 = x + x3
        y3 = y + y3
        x4 = x + x4
        y4 = y + y4
        bbox_list.append([x1, y1, x2, y2, x3, y3, x4, y4])
    return bbox_list


def is_inside(target_box, text_box):
    """
    check if text box is inside target box
    Args:
        target_box (list): target box where we want to detect, eg. [x1, y1, x2, y2]
        text_box (list): text box, eg. [x1, y1, x2, y2]
    Returns:
        bool: True if text box is inside target box
    """

    x1_1, y1_1, x2_1, y2_1 = target_box
    x1_2, y1_2, x2_2, y2_2 = text_box

    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    else:
        inter_area = 0

    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

    union_area = area1 + area2 - inter_area

    iou = inter_area / union_area if union_area != 0 else 0
    return iou > 0


class TableMatch(object):
    """
    match table html and ocr res
    """

    def __init__(self, filter_ocr_result=False):
        self.filter_ocr_result = filter_ocr_result

    def __call__(self, table_pred, ocr_pred):
        structures = table_pred["structure"]
        table_boxes = table_pred["bbox"]
        ocr_dt_ploys = ocr_pred["dt_polys"]
        ocr_text_res = ocr_pred["rec_text"]
        if self.filter_ocr_result:
            ocr_dt_ploys, ocr_text_res = self._filter_ocr_result(
                table_boxes, ocr_dt_ploys, ocr_text_res
            )
        matched_index = self.metch_table_and_ocr(table_boxes, ocr_dt_ploys)
        pred_html = self.get_html_result(matched_index, ocr_text_res, structures)
        return pred_html

    def metch_table_and_ocr(self, table_boxes, ocr_boxes):
        """
        match table bo

        Args:
            table_boxes (list): bbox for table, 4 points, [x1,y1,x2,y2,x3,y3,x4,y4]
            ocr_boxes (list): bbox for ocr, 4 points, [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]

        Returns:
            dict: matched dict, key is table index, value is ocr index
        """
        matched = {}
        for i, ocr_box in enumerate(np.array(ocr_boxes)):
            ocr_box = convert_4point2rect(ocr_box)
            distances = []
            for j, table_box in enumerate(table_boxes):
                table_box = convert_4point2rect(table_box)
                distances.append(
                    (
                        distance(table_box, ocr_box),
                        1.0 - compute_iou(table_box, ocr_box),
                    )
                )  # compute iou and l1 distance
            sorted_distances = distances.copy()
            # select det box by iou and l1 distance
            sorted_distances = sorted(
                sorted_distances, key=lambda item: (item[1], item[0])
            )
            if distances.index(sorted_distances[0]) not in matched.keys():
                matched[distances.index(sorted_distances[0])] = [i]
            else:
                matched[distances.index(sorted_distances[0])].append(i)
        return matched

    def get_html_result(self, matched_index, ocr_contents, pred_structures):
        pred_html = []
        td_index = 0
        head_structure = pred_structures[0:3]
        html = "".join(head_structure)
        table_structure = pred_structures[3]
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
                            if (
                                i != len(matched_index[td_index]) - 1
                                and " " != content[-1]
                            ):
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

    def _filter_ocr_result(self, pred_bboxes, dt_boxes, rec_res):
        y1 = pred_bboxes[:, 1::2].min()
        new_dt_boxes = []
        new_rec_res = []

        for box, rec in zip(dt_boxes, rec_res):
            if np.max(box[1::2]) < y1:
                continue
            new_dt_boxes.append(box)
            new_rec_res.append(rec)
        return new_dt_boxes, new_rec_res
