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
import random
import shutil

from .....utils.deps import function_requires_deps, is_dep_available
from .....utils.file_interface import custom_open, write_json_file
from .....utils.logging import info

if is_dep_available("tqdm"):
    from tqdm import tqdm


def split_dataset(root_dir, train_rate, val_rate):
    """split dataset"""
    assert (
        train_rate + val_rate == 100
    ), f"The sum of train_rate({train_rate}), val_rate({val_rate}) should equal 100!"
    assert (
        train_rate > 0 and val_rate > 0
    ), f"The train_rate({train_rate}) and val_rate({val_rate}) should be greater than 0!"

    all_image_info_list = []
    all_category_dict = {}
    max_image_id = 0
    for fn in ["instance_train.json", "instance_val.json"]:
        anno_path = os.path.join(root_dir, "annotations", fn)
        if not os.path.exists(anno_path):
            info(f"The annotation file {anno_path} don't exists, has been ignored!")
            continue
        image_info_list, category_list, max_image_id = json2list(
            anno_path, max_image_id
        )
        all_image_info_list.extend(image_info_list)

        for category in category_list:
            if category["id"] not in all_category_dict:
                all_category_dict[category["id"]] = category

    total_num = len(all_image_info_list)
    random.shuffle(all_image_info_list)

    all_category_list = [all_category_dict[k] for k in all_category_dict]

    start = 0
    for fn, rate in [
        ("instance_train.json", train_rate),
        ("instance_val.json", val_rate),
    ]:
        end = start + round(total_num * rate / 100)
        save_path = os.path.join(root_dir, "annotations", fn)
        if os.path.exists(save_path):
            bak_path = save_path + ".bak"
            shutil.move(save_path, bak_path)
            info(f"The original annotation file {fn} has been backed up to {bak_path}.")
        assemble_write(all_image_info_list[start:end], all_category_list, save_path)
        start = end
    return root_dir


@function_requires_deps("tqdm")
def json2list(json_path, base_image_num):
    """load json as list"""
    assert os.path.exists(json_path), json_path
    with custom_open(json_path, "r") as f:
        data = json.load(f)
    image_info_dict = {}
    max_image_id = 0
    for image_info in data["images"]:
        # 得到全局唯一的image_id
        global_image_id = image_info["id"] + base_image_num
        max_image_id = max(global_image_id, max_image_id)
        image_info["id"] = global_image_id
        image_info_dict[global_image_id] = {"img": image_info, "anno": []}

    image_info_dict = {
        image_info["id"]: {"img": image_info, "anno": []}
        for image_info in data["images"]
    }
    info(f"Start loading annotation file {json_path}...")
    for anno in tqdm(data["annotations"]):
        global_image_id = anno["image_id"] + base_image_num
        anno["image_id"] = global_image_id
        image_info_dict[global_image_id]["anno"].append(anno)
    image_info_list = [
        (image_info_dict[image_info]["img"], image_info_dict[image_info]["anno"])
        for image_info in image_info_dict
    ]
    return image_info_list, data["categories"], max_image_id


def assemble_write(image_info_list, category_list, save_path):
    """assemble coco format and save to file"""
    coco_data = {"categories": category_list}
    image_list = [i[0] for i in image_info_list]
    all_anno_list = []
    for i in image_info_list:
        all_anno_list.extend(i[1])
    anno_list = []
    for i, anno in enumerate(all_anno_list):
        anno["id"] = i + 1
        anno_list.append(anno)

    coco_data["images"] = image_list
    coco_data["annotations"] = anno_list

    write_json_file(coco_data, save_path)
    info(f"The splited annotations has been save to {save_path}.")
