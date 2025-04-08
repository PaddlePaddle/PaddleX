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
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps

from .....utils.deps import function_requires_deps, is_dep_available
from .....utils.errors import DatasetFileNotFoundError
from .utils.visualizer import draw_keypoint

if is_dep_available("pycocotools"):
    from pycocotools.coco import COCO


@function_requires_deps("pycocotools")
def check(dataset_dir, output, sample_num=10):
    """check dataset"""
    dataset_dir = osp.abspath(dataset_dir)
    if not osp.exists(dataset_dir) or not osp.isdir(dataset_dir):
        raise DatasetFileNotFoundError(file_path=dataset_dir)

    sample_cnts = dict()
    sample_paths = defaultdict(list)
    defaultdict(Counter)
    tags = ["instance_train", "instance_val"]
    for _, tag in enumerate(tags):
        file_list = osp.join(dataset_dir, f"annotations/{tag}.json")
        if not osp.exists(file_list):
            if tag in ("instance_train", "instance_val"):
                # train and val file lists must exist
                raise DatasetFileNotFoundError(
                    file_path=file_list,
                    solution=f"Ensure that both `instance_train.json` and `instance_val.json` exist in \
{dataset_dir}/annotations",
                )
            else:
                continue
        else:
            with open(file_list, "r", encoding="utf-8") as f:
                jsondata = json.load(f)

            coco = COCO(file_list)
            num_class = len(coco.getCatIds())

            vis_save_dir = osp.join(output, "demo_img")

            image_info = jsondata["images"]
            sample_cnts[tag] = len(image_info)
            sample_num = min(sample_num, len(image_info))
            for i in range(sample_num):
                file_name = image_info[i]["file_name"]
                img_id = image_info[i]["id"]
                img_path = osp.join(dataset_dir, "images", file_name)
                if not osp.exists(img_path):
                    raise DatasetFileNotFoundError(file_path=img_path)
                img = Image.open(img_path)
                img = ImageOps.exif_transpose(img)
                vis_im = draw_keypoint(img, coco, img_id)
                vis_path = osp.join(vis_save_dir, file_name)
                Path(vis_path).parent.mkdir(parents=True, exist_ok=True)
                vis_im.save(vis_path)
                sample_path = osp.join(
                    "check_dataset", os.path.relpath(vis_path, output)
                )
                sample_paths[tag].append(sample_path)

    attrs = {}
    attrs["num_classes"] = num_class
    attrs["train_samples"] = sample_cnts["instance_train"]
    attrs["train_sample_paths"] = sample_paths["instance_train"]

    attrs["val_samples"] = sample_cnts["instance_val"]
    attrs["val_sample_paths"] = sample_paths["instance_val"]
    return attrs
