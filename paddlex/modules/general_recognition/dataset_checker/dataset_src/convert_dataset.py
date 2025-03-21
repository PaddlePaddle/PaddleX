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

from .....utils.errors import ConvertFailedError
from .....utils.file_interface import custom_open


def check_src_dataset(root_dir, dataset_type):
    """check src dataset format validity"""
    if dataset_type in ("LabelMe"):
        pass
    else:
        raise ConvertFailedError(
            message=f"数据格式转换失败！不支持{dataset_type}格式数据集。当前仅支持 LabelMe 格式。"
        )

    err_msg_prefix = f"数据格式转换失败！请参考上述`{dataset_type}格式数据集示例`检查待转换数据集格式。"

    for anno in ["label.txt", "annotations", "images"]:
        src_anno_path = os.path.join(root_dir, anno)
        if not os.path.exists(src_anno_path):
            raise ConvertFailedError(
                message=f"{err_msg_prefix}保证{src_anno_path}文件存在。"
            )
    return None


def convert(dataset_type, input_dir):
    """convert dataset to multilabel format"""
    # check format validity
    check_src_dataset(input_dir, dataset_type)

    if dataset_type in ("LabelMe"):
        convert_labelme_dataset(input_dir)
    else:
        raise ConvertFailedError(
            message=f"数据格式转换失败！不支持{dataset_type}格式数据集。当前仅支持 LabelMe 格式。"
        )


def convert_labelme_dataset(root_dir):
    os.path.join(root_dir, "images")
    anno_path = os.path.join(root_dir, "annotations")
    label_path = os.path.join(root_dir, "label.txt")
    train_rate = 50
    gallery_rate = 30
    query_rate = 20
    tags = ["train", "gallery", "query"]
    label_dict = {}
    image_files = []

    with custom_open(label_path, "r") as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            line = line.strip()
            label_dict[line] = str(idx)

    for json_file in os.listdir(anno_path):
        with custom_open(os.path.join(anno_path, json_file), "r") as f:
            data = json.load(f)
            filename = data["imagePath"].strip().split("/")[2]
            image_path = os.path.join("images", filename)
            for label, value in data["flags"].items():
                if value:
                    image_files.append(f"{image_path} {label_dict[label]}\n")

    start = 0
    image_num = len(image_files)
    rate_list = [train_rate, gallery_rate, query_rate]
    for i, tag in enumerate(tags):
        rate = rate_list[i]
        if rate == 0:
            continue

        end = start + round(image_num * rate / 100)
        if sum(rate_list[i + 1 :]) == 0:
            end = image_num

        txt_file = os.path.abspath(os.path.join(root_dir, tag + ".txt"))
        with custom_open(txt_file, "w") as f:
            m = 0
            for id in range(start, end):
                m += 1
                f.write(image_files[id])
        start = end
