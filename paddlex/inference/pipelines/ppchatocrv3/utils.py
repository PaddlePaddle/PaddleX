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

import os
import re
import numpy as np
from pathlib import Path
from scipy.ndimage import rotate


def get_ocr_res(pipeline, input):
    """get ocr res"""
    ocr_res_list = []
    if isinstance(input, list):
        img = [im["img"] for im in input]
    elif isinstance(input, dict):
        img = input["img"]
    else:
        img = input
    for ocr_res in pipeline(img):
        ocr_res_list.append(ocr_res)
    if len(ocr_res_list) == 1:
        return ocr_res_list[0]
    else:
        return ocr_res_list


def get_oriclas_results(inputs, predictor, img_list):
    results = []
    for input, pred in zip(inputs, predictor(img_list)):
        results.append(pred)
        angle = int(pred["label_names"][0])
        input["img"] = rotate_image(input["img"], angle)
    return results


def get_uvdoc_results(inputs, predictor, img_list):
    results = []
    for input, pred in zip(inputs, predictor(img_list)):
        results.append(pred)
        input["img"] = np.array(pred["doctr_img"], dtype=np.uint8)
    return results


def get_predictor_res(predictor, input):
    """get ocr res"""
    result_list = []
    if isinstance(input, list):
        img = [im["img"] for im in input]
    elif isinstance(input, dict):
        img = input["img"]
    else:
        img = input
    for res in predictor(img):
        result_list.append(res)
    if len(result_list) == 1:
        return result_list[0]
    else:
        return result_list


def rotate_image(image_array, rotate_angle):
    """rotate image"""
    assert (
        rotate_angle >= 0 and rotate_angle < 360
    ), "rotate_angle must in [0-360), but get {rotate_angle}."
    return rotate(image_array, rotate_angle, reshape=True)


def get_table_text_from_html(all_table_res):
    all_table_ocr_res = []
    structure_res = []
    for table_res in all_table_res:
        table_list = []
        table_lines = re.findall("<tr>(.*?)</tr>", table_res["html"])
        single_table_ocr_res = []
        for td_line in table_lines:
            table_list.extend(re.findall("<td.*?>(.*?)</td>", td_line))
        for text in table_list:
            text = text.replace(" ", "")
            single_table_ocr_res.append(text)
        all_table_ocr_res.append(" ".join(single_table_ocr_res))
        structure_res.append(
            {
                "layout_bbox": table_res["layout_bbox"],
                "table": table_res["html"],
            }
        )
    return structure_res, all_table_ocr_res


def format_key(key_list):
    """format key"""
    if key_list == "":
        return "未内置默认字段，请输入确定的key"
    if isinstance(key_list, list):
        return key_list
    key_list = re.sub(r"[\t\n\r\f\v]", "", key_list)
    key_list = key_list.replace("，", ",").split(",")
    return key_list


def sorted_layout_boxes(res, w):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        res(list):ppstructure results
        w(int):image width
    return:
        sorted results(list)
    """
    num_boxes = len(res)
    if num_boxes == 1:
        res[0]["layout"] = "single"
        return res

    # Sort on the y axis first or sort it on the x axis
    sorted_boxes = sorted(res, key=lambda x: (x["layout_bbox"][1], x["layout_bbox"][0]))
    _boxes = list(sorted_boxes)

    new_res = []
    res_left = []
    res_mid = []
    res_right = []
    i = 0

    while True:
        if i >= num_boxes:
            break
        # Check if there are three columns of pictures
        if (
            _boxes[i]["layout_bbox"][0] > w / 4
            and _boxes[i]["layout_bbox"][0] + _boxes[i]["layout_bbox"][2] < 3 * w / 4
        ):
            _boxes[i]["layout"] = "double"
            res_mid.append(_boxes[i])
            i += 1
        # Check that the bbox is on the left
        elif (
            _boxes[i]["layout_bbox"][0] < w / 4
            and _boxes[i]["layout_bbox"][0] + _boxes[i]["layout_bbox"][2] < 3 * w / 5
        ):
            _boxes[i]["layout"] = "double"
            res_left.append(_boxes[i])
            i += 1
        elif (
            _boxes[i]["layout_bbox"][0] > 2 * w / 5
            and _boxes[i]["layout_bbox"][0] + _boxes[i]["layout_bbox"][2] < w
        ):
            _boxes[i]["layout"] = "double"
            res_right.append(_boxes[i])
            i += 1
        else:
            new_res += res_left
            new_res += res_right
            _boxes[i]["layout"] = "single"
            new_res.append(_boxes[i])
            res_left = []
            res_right = []
            i += 1

    res_left = sorted(res_left, key=lambda x: (x["layout_bbox"][1]))
    res_mid = sorted(res_mid, key=lambda x: (x["layout_bbox"][1]))
    res_right = sorted(res_right, key=lambda x: (x["layout_bbox"][1]))

    if res_left:
        new_res += res_left
    if res_mid:
        new_res += res_mid
    if res_right:
        new_res += res_right

    return new_res
