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


import glob
import os.path
import shutil

import numpy as np

from .....utils.file_interface import custom_open
from .....utils.logging import info


def split_dataset(dataset_root, train_rate, val_rate):
    """
    将图像数据集按照比例分成训练集、验证集和测试集，并生成对应的.txt文件。

    Args:
        dataset_root (str): 数据集根目录路径。
        train_rate (int): 训练集占总数据集的比例（%）。
        val_rate (int): 验证集占总数据集的比例（%）。

    Returns:
        str: 数据划分结果信息。
    """
    sum_rate = train_rate + val_rate
    assert sum_rate == 100, f"训练集、验证集比例之和需要等于100，请修改后重试"
    assert (
        train_rate > 0 and val_rate > 0
    ), f"The train_rate({train_rate}) and val_rate({val_rate}) should be greater than 0!"

    image_dir = os.path.join(dataset_root, "images")
    tags = ["train.txt", "val.txt"]

    image_files = get_files(image_dir, ["png", "jpg", "jpeg", "PNG", "JPG", "JPEG"])
    label_files = get_labels_files(dataset_root, ["train.txt", "val.txt"])

    for tag in tags:
        src_file = os.path.join(dataset_root, tag)
        dst_file = os.path.join(dataset_root, f"{tag}.bak")
        info(
            f"The original annotation file {src_file} has been backed up to {dst_file}."
        )
        shutil.move(src_file, dst_file)

    image_num = len(image_files)
    label_num = len(label_files)
    assert image_num != 0, f"原始图像数量({image_num})为0, 请检查后重试"
    assert (
        image_num == label_num
    ), f"原始图像数量({image_num})和标注图像数量({label_num})不相等，请检查后重试"

    image_files = np.array(image_files)
    label_files = np.array(label_files)
    state = np.random.get_state()
    np.random.shuffle(image_files)
    np.random.set_state(state)
    np.random.shuffle(label_files)

    start = 0
    rate_list = [train_rate, val_rate]
    name_list = ["train", "val"]
    for i, name in enumerate(name_list):
        info("Creating {}.txt...".format(name))

        rate = rate_list[i]
        if rate == 0:
            txt_file = os.path.join(dataset_root, name + ".txt")
            with custom_open(txt_file, "w") as f:
                f.write("")
            continue

        end = start + round(image_num * rate / 100)
        if sum(rate_list[i + 1 :]) == 0:
            end = image_num

        txt_file = os.path.join(dataset_root, name + ".txt")
        with custom_open(txt_file, "w") as f:
            for id in range(start, end):
                right = label_files[id]
                f.write(right)
        start = end

    return dataset_root


def get_files(input_dir, format=["jpg", "png"]):
    """
    在给定目录下获取符合指定文件格式的所有文件路径

    Args:
        input_dir (str): 目标文件夹路径
        format (Union[str, List[str]]): 需要获取的文件格式, 可以是字符串或者字符串列表

    Returns:
        List[str]: 符合格式的所有文件路径列表，返回排序后的结果
    """
    res = []
    if not isinstance(format, (list, tuple)):
        format = [format]
    for item in format:
        pattern = os.path.join(input_dir, f"**/*.{item}")
        files = glob.glob(pattern, recursive=True)
        res.extend(files)
    return sorted(res)


def get_labels_files(input_dir, format=["train.txt", "val.txt"]):
    """
    在给定目录下获取符合指定文件格式的所有文件路径

    Args:
        input_dir (str): 目标文件夹路径
        format (Union[str, List[str]]): 需要获取的文件格式, 可以是字符串或者字符串列表

    Returns:
        List[str]: 符合格式的所有文件路径列表，返回排序后的结果
    """
    res = []
    if not isinstance(format, (list, tuple)):
        format = [format]
    for tag in format:
        file_list = os.path.join(input_dir, f"{tag}")
        if os.path.exists(file_list):
            with custom_open(file_list, "r") as f:
                all_lines = f.readlines()
                res.extend(all_lines)
    return sorted(res)
