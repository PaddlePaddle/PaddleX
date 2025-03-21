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


import math
import os
import os.path as osp
import shutil

import pandas as pd

from .....utils.logging import info


def split_dataset(root_dir, train_rate, val_rate):
    """split dataset"""
    assert (
        train_rate + val_rate == 100
    ), f"The sum of train_rate({train_rate}) and val_rate({val_rate}) should equal 100!"
    assert (
        train_rate > 0 and val_rate > 0
    ), f"The train_rate({train_rate}) and val_rate({val_rate}) should be greater than 0!"

    tags = ["train.csv", "val.csv"]
    df = pd.DataFrame()
    for tag in tags:
        if os.path.exists(osp.join(root_dir, tag)):
            df_one = pd.read_csv(osp.join(root_dir, tag))
        if df.empty:
            df = df_one
        else:
            df = pd.concat([df, df_one], axis=0, join="inner")
    df = df.drop_duplicates(keep="first")
    df_len = df.shape[0]
    point_train = math.floor((df_len * train_rate / 100))
    point_val = math.floor((df_len * (train_rate + val_rate) / 100))

    train_df = df.iloc[:point_train, :]
    val_df = df.iloc[point_train:point_val, :]

    df_dict = {"train.csv": train_df, "val.csv": val_df}
    for tag in df_dict.keys():
        save_path = osp.join(root_dir, tag)
        if os.path.exists(save_path):
            bak_path = save_path + ".bak"
            shutil.move(save_path, bak_path)
            info(
                f"The original annotation file {tag} has been backed up to {bak_path}."
            )
        df_dict[tag].to_csv(save_path, index=False)

    return root_dir
