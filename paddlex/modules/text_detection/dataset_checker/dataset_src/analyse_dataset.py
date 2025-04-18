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


import json
import os
from collections import defaultdict

import numpy as np
from PIL import Image, ImageOps

from .....utils.deps import function_requires_deps, is_dep_available
from .....utils.file_interface import custom_open

if is_dep_available("opencv-contrib-python"):
    import cv2
if is_dep_available("matplotlib"):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg


# show data samples
def simple_analyse(dataset_path, max_recorded_sample_cnts=20, show_label=True):
    """
    Analyse the dataset samples by return not nore than
    max_recorded_sample_cnts image path and label path

    Args:
        dataset_path (str): dataset path
        max_recorded_sample_cnts (int, optional): the number to return. Default: 50.

    Returns:
        tuple: tuple of sample number, image path and label path for train, val and text subdataset.

    """
    tags = ["train", "val", "test"]
    sample_cnts = defaultdict(int)
    img_paths = defaultdict(list)
    lab_paths = defaultdict(list)
    lab_infos = defaultdict(list)
    res = [None] * 9
    delim = "\t"

    for tag in tags:
        file_list = os.path.join(dataset_path, f"{tag}.txt")
        if not os.path.exists(file_list):
            if tag in ("train", "val"):
                res.insert(0, "数据集不符合规范，请先通过数据校准")
                return res
            else:
                continue
        else:
            with custom_open(file_list, "r") as f:
                all_lines = f.readlines()

            # Each line corresponds to a sample
            sample_cnts[tag] = len(all_lines)

            for idx, line in enumerate(all_lines):
                parts = line.strip("\n").split(delim)
                if len(line.strip("\n")) < 1:
                    continue
                if tag in ("train", "val"):
                    valid_num_parts_lst = [2]
                else:
                    valid_num_parts_lst = [1, 2]
                if len(parts) not in valid_num_parts_lst and len(line.strip("\n")) > 1:
                    res.insert(0, "数据集的标注文件不符合规范")
                    return res

                if len(parts) == 2:
                    img_path, lab_path = parts
                else:
                    # len(parts) == 1
                    img_path = parts[0]
                    lab_path = None

                # check det label
                if len(img_paths[tag]) < max_recorded_sample_cnts:
                    img_path = os.path.join(dataset_path, img_path)
                    if lab_path is not None:
                        label = json.loads(lab_path)
                        boxes = []
                        for item in label:
                            if "points" not in item or "transcription" not in item:
                                res.insert(0, "数据集的标注文件不符合规范")
                                return res

                            box = np.array(item["points"])
                            if box.shape[1] != 2:
                                res.insert(0, "数据集的标注文件不符合规范")
                                return res
                            boxes.append(box)
                            txt = item["transcription"]
                            if not isinstance(txt, str):
                                res.insert(0, "数据集的标注文件不符合规范")
                                return res
                        if show_label:
                            lab_img = show_label_img(img_path, boxes)

                    img_paths[tag].append(img_path)
                    if show_label:
                        lab_paths[tag].append(lab_img)
                    else:
                        lab_infos[tag].append({"img_path": img_path, "box": boxes})

    if show_label:
        return (
            "完成数据分析",
            sample_cnts[tags[0]],
            sample_cnts[tags[1]],
            sample_cnts[tags[2]],
            img_paths[tags[0]],
            img_paths[tags[1]],
            img_paths[tags[2]],
            lab_paths[tags[0]],
            lab_paths[tags[1]],
            lab_paths[tags[2]],
        )
    else:
        return (
            "完成数据分析",
            sample_cnts[tags[0]],
            sample_cnts[tags[1]],
            sample_cnts[tags[2]],
            img_paths[tags[0]],
            img_paths[tags[1]],
            img_paths[tags[2]],
            lab_infos[tags[0]],
            lab_infos[tags[1]],
            lab_infos[tags[2]],
        )


@function_requires_deps("opencv-contrib-python")
def show_label_img(img_path, dt_boxes):
    """draw ocr detection label"""
    img = cv2.imread(img_path)
    for box in dt_boxes:
        box = np.array(box).astype(np.int32).reshape(-1, 2)
        cv2.polylines(img, [box], True, color=(0, 255, 0), thickness=3)
    return img[:, :, ::-1]


@function_requires_deps("matplotlib", "opencv-contrib-python")
def deep_analyse(dataset_path, output):
    """class analysis for dataset"""
    sample_results = simple_analyse(
        dataset_path, max_recorded_sample_cnts=float("inf"), show_label=False
    )
    lab_infos = sample_results[-3] + sample_results[-2] + sample_results[-1]
    defaultdict(int)
    img_shapes = []  # w, h
    ratios_w = []
    ratios_h = []
    for info in lab_infos:
        img = np.asarray(ImageOps.exif_transpose(Image.open(info["img_path"])))
        img_h, img_w = np.shape(img)[:2]
        img_shapes.append([img_w, img_h])
        for box in info["box"]:
            box = np.array(box).astype(np.int32).reshape(-1, 2)
            box_w, box_h = np.max(box, axis=0) - np.min(box, axis=0)
            ratio_w = box_w / img_w
            ratio_h = box_h / img_h
            ratios_w.append(ratio_w)
            ratios_h.append(ratio_h)
    m_w_img, m_h_img = np.mean(img_shapes, axis=0)  # mean img shape
    m_num_box = len(ratios_w) / len(lab_infos)  # num box per img

    ratio_w = [i * 1000 for i in ratios_w]
    ratio_h = [i * 1000 for i in ratios_h]
    w_bins = int((max(ratio_w) - min(ratio_w)) // 10)
    h_bins = int((max(ratio_h) - min(ratio_h)) // 10)

    fig, ax = plt.subplots()
    ax.hist(ratio_w, bins=w_bins, rwidth=0.8, color="yellowgreen")
    ax.set_xlabel("Width rate *1000")
    ax.set_ylabel("number")
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    bar_array = np.frombuffer(canvas.tostring_rgb(), dtype="uint8").reshape(
        int(height), int(width), 3
    )

    # pie
    fig, ax = plt.subplots()
    ax.hist(ratio_h, bins=h_bins, rwidth=0.8, color="pink")
    ax.set_xlabel("Height rate *1000")
    ax.set_ylabel("number")
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    pie_array = np.frombuffer(canvas.tostring_rgb(), dtype="uint8").reshape(
        int(height), int(width), 3
    )

    os.makedirs(output, exist_ok=True)
    fig_path = os.path.join(output, "histogram.png")
    img_array = np.concatenate((bar_array, pie_array), axis=1)
    cv2.imwrite(fig_path, img_array)
    return {"histogram": os.path.join("check_dataset", "histogram.png")}
    # return {
    #     "图像平均宽度": m_w_img,
    #     "图像平均高度": m_h_img,
    #     "每张图平均文本检测框数量": m_num_box,
    #     "检测框相对宽度分布图": fig1_path,
    #     "检测框相对高度分布图": fig2_path
    # }
