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
import os.path as osp
import shutil

import numpy as np
from PIL import Image, ImageDraw

from .....utils import logging
from .....utils.deps import function_requires_deps, is_dep_available
from .....utils.file_interface import custom_open

if is_dep_available("opencv-contrib-python"):
    import cv2


def convert_dataset(dataset_type, input_dir):
    """convert to paddlex official format"""
    if dataset_type == "LabelMe":
        return convert_labelme_dataset(input_dir)
    else:
        raise NotImplementedError(dataset_type)


@function_requires_deps("opencv-contrib-python")
def convert_labelme_dataset(input_dir):
    """convert labelme format to paddlex official format"""
    bg_name = "_background_"
    ignore_name = "__ignore__"

    # prepare dir
    output_img_dir = osp.join(input_dir, "images")
    output_annot_dir = osp.join(input_dir, "annotations")
    if not osp.exists(output_img_dir):
        os.makedirs(output_img_dir)
    if not osp.exists(output_annot_dir):
        os.makedirs(output_annot_dir)

    # collect class_names and set class_name_to_id
    class_names = []
    class_name_to_id = {}
    split_tags = ["train", "val"]
    for tag in split_tags:
        mapping_file = osp.join(input_dir, f"{tag}_anno_list.txt")
        with open(mapping_file, "r") as f:
            label_files = [
                osp.join(input_dir, line.strip("\n")) for line in f.readlines()
            ]
        for label_file in label_files:
            with custom_open(label_file, "r") as fp:
                data = json.load(fp)
                for shape in data["shapes"]:
                    cls_name = shape["label"]
                    if cls_name not in class_names:
                        class_names.append(cls_name)

        if ignore_name in class_names:
            class_name_to_id[ignore_name] = 255
            class_names.remove(ignore_name)
        if bg_name in class_names:
            class_names.remove(bg_name)
        class_name_to_id[bg_name] = 0
        for i, name in enumerate(class_names):
            class_name_to_id[name] = i + 1

        if len(class_names) > 256:
            raise ValueError(
                f"There are {len(class_names)} categories in the annotation file, "
                f"exceeding 256, Not compliant with paddlex official format!"
            )

        # create annotated images and copy origin images
        color_map = get_color_map_list(256)
        img_file_list = []
        label_file_list = []
        for i, label_file in enumerate(label_files):
            filename = osp.splitext(osp.basename(label_file))[0]
            annotated_img_path = osp.join(output_annot_dir, filename + ".png")
            with custom_open(label_file, "r") as f:
                data = json.load(f)
                img_path = osp.join(osp.dirname(label_file), data["imagePath"])
                if not os.path.exists(img_path):
                    logging.info("%s is not existed, skip this image" % img_path)
                    continue
                img_name = img_path.split("/")[-1]
                img_file_list.append(f"images/{img_name}")
                label_img_name = annotated_img_path.split("/")[-1]
                label_file_list.append(f"annotations/{label_img_name}")

                img = np.asarray(cv2.imread(img_path))
                lbl = shape2label(
                    img_size=img.shape,
                    shapes=data["shapes"],
                    class_name_mapping=class_name_to_id,
                )
                lbl_pil = Image.fromarray(lbl.astype(np.uint8), mode="P")
                lbl_pil.putpalette(color_map)
                lbl_pil.save(annotated_img_path)

                shutil.copy(img_path, output_img_dir)
        with custom_open(osp.join(input_dir, f"{tag}.txt"), "w") as fp:
            for img_path, lbl_path in zip(img_file_list, label_file_list):
                fp.write(f"{img_path} {lbl_path}\n")

    with custom_open(osp.join(input_dir, "class_name.txt"), "w") as fp:
        for name in class_names:
            fp.write(f"{name}{os.linesep}")
    with custom_open(osp.join(input_dir, "class_name_to_id.txt"), "w") as fp:
        for key, val in class_name_to_id.items():
            fp.write(f"{val}: {key}{os.linesep}")

    return input_dir


def get_color_map_list(num_classes):
    """get color map list"""
    num_classes += 1
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= ((lab >> 0) & 1) << (7 - j)
            color_map[i * 3 + 1] |= ((lab >> 1) & 1) << (7 - j)
            color_map[i * 3 + 2] |= ((lab >> 2) & 1) << (7 - j)
            j += 1
            lab >>= 3
    color_map = color_map[3:]
    return color_map


def shape2label(img_size, shapes, class_name_mapping):
    """根据输入的形状列表，将图像的标签矩阵填充为对应形状的类别编号"""
    label = np.zeros(img_size[:2], dtype=np.int32)
    for shape in shapes:
        points = shape["points"]
        class_name = shape["label"]
        label_mask = polygon2mask(img_size[:2], points)
        label[label_mask] = class_name_mapping[class_name]
    return label


def polygon2mask(img_size, points):
    """将给定形状的点转换成对应的掩膜"""
    label_mask = Image.fromarray(np.zeros(img_size[:2], dtype=np.uint8))
    image_draw = ImageDraw.Draw(label_mask)
    points_list = [tuple(point) for point in points]
    assert len(points_list) > 2, ValueError("Polygon must have points more than 2")
    image_draw.polygon(xy=points_list, outline=1, fill=1)
    return np.array(label_mask, dtype=bool)
