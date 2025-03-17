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
import os.path as osp
import random
from collections import defaultdict
from PIL import Image, ImageOps

import numpy as np

from .....utils.errors import DatasetFileNotFoundError
from .utils.visualizer import draw_bbox


def check_train(dataset_dir, output, sample_num=10):
    """check dataset"""
    dataset_dir = osp.abspath(dataset_dir)
    if not osp.exists(dataset_dir) or not osp.isdir(dataset_dir):
        raise DatasetFileNotFoundError(file_path=dataset_dir)

    img_list_dir = osp.join(dataset_dir, "image_lists")
    if not osp.exists(img_list_dir) or not osp.isdir(img_list_dir):
        raise DatasetFileNotFoundError(file_path=img_list_dir)

    img_list = osp.join(img_list_dir, "mot.train")
    if not osp.exists(img_list):
        raise DatasetFileNotFoundError(
            file_path=img_list,
            solution=f"Ensure that `mot.train` exists in {img_list_dir}",
        )

    sample_paths = []
    labels = []
    identities = []
    with open(img_list, "r", encoding="utf-8") as f:
        all_lines = f.readlines()
        random.seed(123)
        random.shuffle(all_lines)
        sample_cnts = len(all_lines)
        for line in all_lines:
            file_name = line.strip("\n")
            img_path = osp.join(dataset_dir, file_name)
            if not osp.exists(img_path):
                raise DatasetFileNotFoundError(file_path=img_path)

            label_file = osp.splitext(img_path)[0] + ".txt"
            label_file = label_file.replace("images", "labels_with_ids")
            if not osp.exists(label_file):
                raise DatasetFileNotFoundError(file_path=label_file)

            label = np.loadtxt(label_file)
            for l in label:
                cls_id, identity = int(l[0]), int(l[1])
                labels.append(cls_id)
                identities.append(identity)

            vis_save_dir = osp.join(output, "demo_img/train")
            if not osp.exists(vis_save_dir):
                os.makedirs(vis_save_dir)

            if len(sample_paths) < sample_num:
                img = Image.open(img_path)
                img = ImageOps.exif_transpose(img)
                vis_im = draw_bbox(img, label[:, 0], label[:, 1], label[:, 2:])
                vis_path = osp.join(vis_save_dir, osp.basename(file_name))
                vis_im.save(vis_path)
                sample_path = osp.join(
                    "check_dataset", os.path.relpath(vis_path, output)
                )
                sample_paths.append(sample_path)

    num_classes = max(labels) + 1
    attrs = {}
    attrs["train_num_classes"] = num_classes
    attrs["train_num_identities"] = max(identities)
    attrs["train_samples"] = sample_cnts
    attrs["train_sample_paths"] = sample_paths
    return attrs


def check_val(dataset_dir, output, sample_num=10):
    """check dataset"""
    dataset_dir = osp.abspath(dataset_dir)
    if not osp.exists(dataset_dir) or not osp.isdir(dataset_dir):
        raise DatasetFileNotFoundError(file_path=dataset_dir)

    val_dir = osp.join(dataset_dir, "val")
    if not osp.exists(val_dir) or not osp.isdir(val_dir):
        raise DatasetFileNotFoundError(file_path=val_dir)

    imgs_dir = osp.join(val_dir, "images")
    if not osp.exists(imgs_dir) or not osp.isdir(imgs_dir):
        raise DatasetFileNotFoundError(file_path=imgs_dir)

    sample_paths = []
    sample_cnts = 0
    labels = []
    identities = []
    seqs = os.listdir(imgs_dir)
    seqs.sort()
    for seq in seqs:
        cur_cnts = 0
        imgs_root = os.path.join(imgs_dir, seq, "img1")
        if not osp.exists(imgs_root) or not osp.isdir(imgs_root):
            raise DatasetFileNotFoundError(file_path=imgs_root)

        label_file = os.path.join(imgs_dir, seq, "gt", "gt.txt")
        if not osp.exists(label_file):
            raise DatasetFileNotFoundError(file_path=label_file)

        frame_anns = defaultdict(list)
        label = np.loadtxt(label_file, delimiter=",")
        for line in label:
            frame_id = int(line[0])
            frame_anns[frame_id].append(line)
        cur_cnts = len(frame_anns)
        frame_list = list(frame_anns.keys())
        random.seed(123)
        random.shuffle(frame_list)
        for frame_id in frame_list:
            frame_ann = frame_anns[frame_id]
            anns = np.array(frame_ann)
            for ann in anns:
                identity, cls_id = int(ann[1]), int(ann[7])
                labels.append(cls_id)
                identities.append(identity)
            img_name = "{:06d}".format(frame_id)
            img_path = os.path.join(imgs_root, img_name + ".jpg")
            if not osp.exists(img_path):
                img_path = os.path.join(imgs_root, img_name + ".png")
            if not osp.exists(img_path):
                raise DatasetFileNotFoundError(file_path=img_path)

            vis_save_dir = osp.join(output, "demo_img/val")
            if not osp.exists(vis_save_dir):
                os.makedirs(vis_save_dir)

            if len(sample_paths) < sample_num:
                img = Image.open(img_path)
                img = ImageOps.exif_transpose(img)
                x, y, w, h = anns[:, 2], anns[:, 3], anns[:, 4], anns[:, 5]
                x += w / 2
                y += h / 2
                width, height = img.width, img.height
                xywh = np.stack([x, y, w, h], axis=1) / np.array(
                    [width, height, width, height]
                )
                vis_im = draw_bbox(img, anns[:, 7], anns[:, 1], xywh)
                vis_path = osp.join(vis_save_dir, osp.basename(img_path))
                vis_im.save(vis_path)
                sample_path = osp.join(
                    "check_dataset", os.path.relpath(vis_path, output)
                )
                sample_paths.append(sample_path)
        sample_cnts += cur_cnts

    num_classes = max(labels)
    attrs = {}
    attrs["val_num_classes"] = num_classes
    attrs["val_num_identities"] = max(identities)
    attrs["val_samples"] = sample_cnts
    attrs["val_sample_paths"] = sample_paths
    return attrs
