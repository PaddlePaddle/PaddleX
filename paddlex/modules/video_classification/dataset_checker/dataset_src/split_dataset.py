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
from random import shuffle

from .....utils.file_interface import custom_open


def split_dataset(root_dir, train_rate, val_rate):
    """
    Split the image dataset into training, validation, and test sets according to the given ratios,
    and generate corresponding .txt files.

    Args:
        root_dir (str): Path to the root directory of the dataset.
        train_rate (int): Percentage of the dataset to be used as the training set.
        val_rate (int): Percentage of the dataset to be used as the validation set.

    Returns:
        str: Information about the dataset split results.
    """
    sum_rate = train_rate + val_rate
    assert (
        sum_rate == 100
    ), f"The sum of train_rate({train_rate}), val_rate({val_rate}) should equal 100!"
    assert (
        train_rate > 0 and val_rate > 0
    ), f"The train_rate({train_rate}) and val_rate({val_rate}) should be greater than 0!"
    tags = ["train", "val"]
    valid_path = False
    video_files = []
    for tag in tags:
        split_image_list = os.path.abspath(os.path.join(root_dir, f"{tag}.txt"))
        rename_image_list = os.path.abspath(os.path.join(root_dir, f"{tag}.txt.bak"))
        if os.path.exists(split_image_list):
            with custom_open(split_image_list, "r") as f:
                lines = f.readlines()
            video_files = video_files + lines
            valid_path = True
            if not os.path.exists(rename_image_list):
                os.rename(split_image_list, rename_image_list)

    assert (
        valid_path
    ), f"The files to be divided{tags[0]}.txt, {tags[1]}.txt, do not exist in the dataset directory."

    shuffle(video_files)
    start = 0
    video_num = len(video_files)
    rate_list = [train_rate, val_rate]
    for i, tag in enumerate(tags):

        rate = rate_list[i]
        if rate == 0:
            continue

        end = start + round(video_num * rate / 100)
        if sum(rate_list[i + 1 :]) == 0:
            end = video_num

        txt_file = os.path.abspath(os.path.join(root_dir, tag + ".txt"))
        with custom_open(txt_file, "w") as f:
            m = 0
            for id in range(start, end):
                m += 1
                f.write(video_files[id])
        start = end

    return root_dir
