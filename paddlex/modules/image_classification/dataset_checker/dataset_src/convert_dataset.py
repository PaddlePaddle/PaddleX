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

from .....utils.file_interface import custom_open


def convert(input_dir):
    """
    Convert json in file into imagenet format.
    """
    label_path = os.path.join(input_dir, "flags.txt")
    label_dict = {}
    label_content = []
    with custom_open(label_path, "r") as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            line = line.strip()
            label_dict[line] = str(idx)
            label_content.append(f"{str(idx)} {line}\n")
    with custom_open(os.path.join(input_dir, "label.txt"), "w") as f:
        f.write("".join(label_content))
    anno_path = os.path.join(input_dir, "annotations")
    os.listdir()
    train_list = os.path.join(input_dir, "train.txt")
    val_list = os.path.join(input_dir, "val.txt")
    label_info = []
    for json_file in os.listdir(anno_path):
        with custom_open(os.path.join(anno_path, json_file), "r") as f:
            data = json.load(f)
            file_name = os.path.join("images", data["imagePath"].strip().split("/")[2])
            for label, value in data["flags"].items():
                if value:
                    label_info.append(f"{file_name} {label_dict[label]}\n")
    with custom_open(train_list, "w") as file:
        file.write("".join(label_info))
    with custom_open(val_list, "w") as file:
        pass
