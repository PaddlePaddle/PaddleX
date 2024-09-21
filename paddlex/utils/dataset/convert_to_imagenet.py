# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import json
import random


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='./data')
    args = parser.parse_args()
    return args

def trans_to_imagenet(dataset_path, train_ratio=1.0):
    """
    Convert json in file into imagenet format.
    """
    # Write Label
    label_path = os.path.join(dataset_path, "flags.txt")
    label_dict = {}
    label_content = []
    with open(label_path, "r") as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            line = line.strip()
            label_dict[line] = str(idx)
            label_content.append(f"{str(idx)} {line}\n")
    with open(os.path.join(dataset_path, "label.txt"), "w", encoding='utf-8') as f:
        f.write("".join(label_content))
    
    # Split train and val
    anno_path = os.path.join(dataset_path, "annotations")
    jsons_path = os.listdir()
    train_list = os.path.join(dataset_path, "train.txt")
    val_list = os.path.join(dataset_path, "val.txt")
    label_info = []
    for json_file in os.listdir(anno_path):
        # If the file is a hidden file, skip this file.
        if json_file[0] == ".":
            continue
        with open(os.path.join(anno_path, json_file), "r", encoding='utf-8') as f:
            data = json.load(f)

            # In Windows, the separator is "\", and in Linux, it is "/".
            if "/" in data["imagePath"]:
                path_separator = '/'
            else:
                path_separator = '\\'

            file_name = os.path.join("images", data["imagePath"].strip().split(path_separator)[2])
            for label, value in data["flags"].items():
                if value:
                    label_info.append(f"{file_name} {label_dict[label]}\n")

    random.shuffle(label_info)
    train_num = int(len(label_info) * train_ratio)
    with open(train_list, "w", encoding='utf-8') as file:
        file.write("".join(label_info[0:train_num]))
    with open(val_list, "w", encoding='utf-8') as file:
        file.write("".join(label_info[train_num:]))
                
if __name__ == "__main__":
    args = parse_args()
    trans_to_imagenet(args.dataset_path)