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
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageOps

from .....utils.deps import function_requires_deps
from .....utils.errors import DatasetFileNotFoundError
from .....utils.logging import error, info, warning
from .utils.visualizer import draw_bbox, draw_mask


def validate_read_order(annotations, image_id):
    """Validate the read_order field for all annotations of a single image.

    Args:
        annotations: list of all annotations for the image
        image_id: image ID

    Returns:
        bool: whether validation passed

    Raises:
        ValueError: missing or invalid read_order
    """
    # 1. Check completeness and type
    for ann in annotations:
        if "read_order" not in ann:
            raise ValueError(
                f"Annotation {ann['id']} in image {image_id} is missing 'read_order' field. "
                f"Layout Analysis model requires 'read_order' for all annotations."
            )
        if not isinstance(ann["read_order"], int) or ann["read_order"] < 0:
            raise ValueError(
                f"Annotation {ann['id']} in image {image_id} has invalid read_order: {ann.get('read_order')}. "
                f"Expected non-negative integer."
            )

    # 2. Check continuity and uniqueness
    read_orders = sorted([ann["read_order"] for ann in annotations])
    expected = list(range(len(read_orders)))

    if read_orders != expected:
        warning(
            f"Image {image_id}: read_order sequence {read_orders} is not continuous. "
            f"Expected {expected}. This may cause training issues."
        )
        return False

    return True


@function_requires_deps("pycocotools")
def check(dataset_dir, output, sample_num=10):
    """check dataset with read_order validation"""
    from pycocotools.coco import COCO

    info(f"Checking dataset: {dataset_dir}")
    dataset_dir = osp.abspath(dataset_dir)
    if not osp.exists(dataset_dir) or not osp.isdir(dataset_dir):
        raise DatasetFileNotFoundError(file_path=dataset_dir)

    sample_cnts = dict()
    sample_paths = defaultdict(list)
    read_order_stats = {}  # read_order validation statistics
    tags = ["instance_train", "instance_val"]

    for tag in tags:
        file_list = osp.join(dataset_dir, f"annotations/{tag}.json")
        if not osp.exists(file_list):
            if tag in ("instance_train", "instance_val"):
                raise DatasetFileNotFoundError(
                    file_path=file_list,
                    solution=f"Ensure that both `instance_train.json` and `instance_val.json` exist in "
                    f"{dataset_dir}/annotations",
                )
            else:
                continue

        with open(file_list, "r", encoding="utf-8") as f:
            jsondata = json.load(f)

        datanno = jsondata["annotations"]
        sample_cnts[tag] = len(datanno)
        coco = COCO(file_list)
        num_class = len(coco.getCatIds())

        # validate read_order
        img_anns = defaultdict(list)
        for ann in datanno:
            img_anns[ann["image_id"]].append(ann)

        valid_count = 0
        invalid_images = []
        for img_id, anns in img_anns.items():
            try:
                is_valid = validate_read_order(anns, img_id)
                if is_valid:
                    valid_count += 1
                else:
                    invalid_images.append(img_id)
            except ValueError as e:
                error(str(e))
                raise

        read_order_stats[tag] = {
            "total_images": len(img_anns),
            "valid_images": valid_count,
            "invalid_images": invalid_images,
            "pass_rate": valid_count / len(img_anns) if len(img_anns) > 0 else 0,
        }

        info(
            f"{tag}: read_order validation pass rate = {read_order_stats[tag]['pass_rate']:.2%}"
        )

        # visualization
        vis_save_dir = osp.join(output, "demo_img")
        image_info = jsondata["images"]
        sample_num = min(sample_num, len(image_info))
        if sample_num < 10:
            info("Only {} images in {}.json".format(len(image_info), tag))

        for i in range(sample_num):
            file_name = image_info[i]["file_name"]
            img_id = image_info[i]["id"]
            img_path = osp.join(dataset_dir, "images", file_name)
            if not osp.exists(img_path):
                raise DatasetFileNotFoundError(file_path=img_path)

            img = Image.open(img_path)
            img = ImageOps.exif_transpose(img)
            vis_im = draw_bbox(
                img, coco, img_id
            )  # draw_bbox renders read_order automatically
            vis_im = draw_mask(vis_im, coco, img_id)
            vis_path = osp.join(vis_save_dir, file_name)
            Path(vis_path).parent.mkdir(parents=True, exist_ok=True)
            vis_im.save(vis_path)
            sample_path = osp.join("check_dataset", os.path.relpath(vis_path, output))
            sample_paths[tag].append(sample_path)

    attrs = {}
    attrs["num_classes"] = num_class
    attrs["train_samples"] = sample_cnts["instance_train"]
    attrs["train_sample_paths"] = sample_paths["instance_train"]
    attrs["val_samples"] = sample_cnts["instance_val"]
    attrs["val_sample_paths"] = sample_paths["instance_val"]
    attrs["read_order_validation"] = read_order_stats

    return attrs
