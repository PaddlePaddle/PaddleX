#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import numpy as np
import os

from paddle.io import Dataset
from .. import dataloader


class MixDataset(Dataset):
    def __init__(self, datasets_config):
        super().__init__()
        self.dataset_list = []
        start_idx = 0
        end_idx = 0
        for config_i in datasets_config:
            dataset_name = config_i.pop('name')
            dataset = getattr(dataloader, dataset_name)(**config_i)
            end_idx += len(dataset)
            self.dataset_list.append([end_idx, start_idx, dataset])
            start_idx = end_idx

        self.length = end_idx

    def __getitem__(self, idx):
        for dataset_i in self.dataset_list:
            if dataset_i[0] > idx:
                dataset_i_idx = idx - dataset_i[1]
                return dataset_i[2][dataset_i_idx]

    def __len__(self):
        return self.length

    def get_dataset_list(self):
        return self.dataset_list
