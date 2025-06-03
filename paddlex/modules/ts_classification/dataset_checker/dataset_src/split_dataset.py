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


def split_dataset(root_dir, train_rate, val_rate, group_id="group_id"):
    """split dataset"""
    assert (
        train_rate + val_rate == 100
    ), f"The sum of train_rate({train_rate}) and val_rate({val_rate}) should equal 100!"
    assert (
        train_rate > 0 and val_rate > 0
    ), f"The train_rate({train_rate}) and val_rate({val_rate}) should be greater than 0!"

    tags = ["train.csv", "val.csv"]
    df = pd.DataFrame()
    group_unique = None
    for tag in tags:
        if os.path.exists(osp.join(root_dir, tag)):
            df_one = pd.read_csv(osp.join(root_dir, tag))
            cols = df_one.columns.values.tolist()
            assert (
                group_id in cols
            ), f"The default group_id '{group_id}' is not found in the df columns."
        if df.empty:
            df = df_one
            group_unique = sorted(df[group_id].unique())
        else:
            group_unique_one = sorted(df_one[group_id].unique())
            for id in group_unique_one:
                if id in group_unique:
                    df_one[group_id].replace(id, str(id) + "_", inplace=True)
                    group_unique.append(str(id) + "_")
            df = pd.concat([df, df_one], axis=0)
    df = df.drop_duplicates(keep="first")

    group_unique = df[group_id].unique()
    dfs = []  # separate multiple group
    for column in group_unique:
        df_one = df[df[group_id].isin([column])]
        df_one = df_one.drop_duplicates(subset=["time"], keep="first")
        dfs.append(df_one)
    group_len = len(dfs)
    point_train = math.floor((group_len * train_rate / 100))
    point_val = math.floor((group_len * (train_rate + val_rate) / 100))

    assert point_train > 0, f"The train_len is 0, the train_percent should be greater ."
    assert (
        point_val - point_train > 0
    ), f"The train_len is 0, the val_percent should be greater ."

    train_df = pd.concat(dfs[:point_train], axis=0)
    val_df = pd.concat(dfs[point_train:point_val], axis=0)
    df_dict = {"train.csv": train_df, "val.csv": val_df}
    if point_val < group_len - 1:
        test_df = pd.concat(dfs[point_val:], axis=0)
        df_dict.update({"test.csv": test_df})
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
