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

import os.path as osp
import pickle

from .....utils.errors import DatasetFileNotFoundError
from .....utils.misc import abspath


def check(dataset_dir):
    dataset_dir = abspath(dataset_dir)
    max_sample_num = 5

    if not osp.exists(dataset_dir) or not osp.isdir(dataset_dir):
        raise DatasetFileNotFoundError(file_path=dataset_dir)

    anno_file = osp.join(dataset_dir, "nuscenes_infos_train.pkl")
    if not osp.exists(anno_file):
        raise DatasetFileNotFoundError(file_path=anno_file)
    train_mate, train_classes = get_data(anno_file, max_sample_num)

    anno_file = osp.join(dataset_dir, "nuscenes_infos_val.pkl")
    if not osp.exists(anno_file):
        raise DatasetFileNotFoundError(file_path=anno_file)
    val_mate, val_classes = get_data(anno_file, max_sample_num)
    train_sample_paths = []
    val_sample_paths = []

    for item in train_mate:
        train_sample_paths.append(item["lidar_path"])

    for item in val_mate:
        val_sample_paths.append(item["lidar_path"])

    all_classes = set()
    for tc in train_classes:
        all_classes.add(tc)
    for vc in val_classes:
        all_classes.add(vc)
    num_classes = len(all_classes)
    meta = {
        "num_classes": num_classes,
        "train_meta": train_mate,
        "val_meta": val_mate,
        "train_sample_paths": train_sample_paths,
        "val_sample_paths": val_sample_paths,
    }
    return meta


def get_data(ann_file, max_sample_num):
    infos = data_infos(ann_file, max_sample_num)
    meta = []
    gt_class = set()
    for info in infos:
        image_paths = []
        cam_orders = [
            "CAM_FRONT_LEFT",
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK_RIGHT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
        ]
        for cam_type in cam_orders:
            cam_info = info["cams"][cam_type]
            cam_data_path = cam_info["data_path"]
            image_paths.append(cam_data_path)

        meta.append(
            {
                "sample_idx": info["token"],
                "lidar_path": info["lidar_path"],
                "image_paths": image_paths,
            }
        )
        class_names = info["gt_names"]

        for cls_name in class_names:
            gt_class.add(cls_name)

    return meta, gt_class


def data_infos(ann_file, max_sample_num):
    data = pickle.load(open(ann_file, "rb"))
    data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
    data_infos = data_infos[:max_sample_num]
    return data_infos
