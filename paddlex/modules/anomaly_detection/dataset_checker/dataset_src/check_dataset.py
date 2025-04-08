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


import os
import os.path as osp

import numpy as np
from PIL import Image

from .....utils.deps import function_requires_deps, is_dep_available
from .....utils.errors import DatasetFileNotFoundError
from .....utils.file_interface import custom_open
from .....utils.logging import info
from .utils.visualizer import visualize

if is_dep_available("opencv-contrib-python"):
    import cv2


@function_requires_deps("opencv-contrib-python")
def check_dataset(dataset_dir, output, sample_num=10):
    """check dataset"""
    dataset_dir = osp.abspath(dataset_dir)
    if not osp.exists(dataset_dir) or not osp.isdir(dataset_dir):
        raise DatasetFileNotFoundError(file_path=dataset_dir)
    vis_save_dir = osp.join(output, "demo_img")
    if not osp.exists(vis_save_dir):
        os.makedirs(vis_save_dir)
    split_tags = ["train", "val"]
    attrs = dict()
    class_ids = set()
    for tag in split_tags:
        mapping_file = osp.join(dataset_dir, f"{tag}.txt")
        if not osp.exists(mapping_file):
            info(f"The mapping file ({mapping_file}) doesn't exist, ignored.")
            info(
                "If you are using MVTec_AD dataset, add args below in your training commands:"
            )
            info("-o CheckDataset.convert.enable=True")
            info("-o CheckDataset.convert.src_dataset_type=MVTec_AD")
            continue
        with custom_open(mapping_file, "r") as fp:
            lines = filter(None, (line.strip() for line in fp.readlines()))
            for i, line in enumerate(lines):
                img_file, ann_file = line.split(" ")
                img_file = osp.join(dataset_dir, img_file)
                ann_file = osp.join(dataset_dir, ann_file)
                assert osp.exists(img_file), FileNotFoundError(
                    f"{img_file} not exist, please check!"
                )
                assert osp.exists(ann_file), FileNotFoundError(
                    f"{ann_file} not exist, please check!"
                )
                img = np.array(cv2.imread(img_file), "uint8")
                ann = np.array(cv2.imread(ann_file), "uint8")[:, :, 0]
                assert img.shape[:2] == ann.shape, ValueError(
                    f"The shape of {img_file}:{img.shape[:2]} and "
                    f"{ann_file}:{ann.shape} must be the same!"
                )
                if tag == "val":
                    class_ids = class_ids | set(ann.reshape([-1]).tolist())
                if i < sample_num:
                    vis_img = visualize(img, ann)
                    vis_img = Image.fromarray(vis_img)
                    vis_save_path = osp.join(vis_save_dir, osp.basename(img_file))
                    vis_img.save(vis_save_path)
                    vis_save_path = osp.join(
                        "check_dataset", os.path.relpath(vis_save_path, output)
                    )
                    if f"{tag}_sample_paths" not in attrs:
                        attrs[f"{tag}_sample_paths"] = [vis_save_path]
                    else:
                        attrs[f"{tag}_sample_paths"].append(vis_save_path)
            if f"{tag}_samples" not in attrs:
                attrs[f"{tag}_samples"] = i + 1
    if 255 in class_ids:
        class_ids.remove(255)
    attrs["num_classes"] = len(class_ids)
    return attrs
