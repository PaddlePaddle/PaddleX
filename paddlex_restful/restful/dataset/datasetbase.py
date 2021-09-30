# copytrue (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

import pickle
import os.path as osp
import random
from .utils import copy_directory


class DatasetBase(object):
    def __init__(self, dataset_id, path):
        self.id = dataset_id
        self.path = path
        self.all_files = list()
        self.file_info = dict()
        self.label_info = dict()
        self.labels = list()
        self.train_files = list()
        self.val_files = list()
        self.test_files = list()
        self.class_train_file_list = dict()
        self.class_val_file_list = dict()
        self.class_test_file_list = dict()

    def copy_dataset(self, source_path, files):
        # 将原数据集拷贝至目标路径
        copy_directory(source_path, self.path, files)

    def dump_statis_info(self):
        # info['fields']指定了需要dump的信息
        info = dict()
        info['fields'] = [
            'file_info', 'label_info', 'labels', 'train_files', 'val_files',
            'test_files', 'class_train_file_list', 'class_val_file_list',
            'class_test_file_list'
        ]
        for field in info['fields']:
            if hasattr(self, field):
                info[field] = getattr(self, field)
        with open(osp.join(self.path, 'statis.pkl'), 'wb') as f:
            pickle.dump(info, f)

    def load_statis_info(self):
        with open(osp.join(self.path, 'statis.pkl'), 'rb') as f:
            info = pickle.load(f)
        for field in info['fields']:
            if field in info:
                setattr(self, field, info[field])

    def split(self, val_split, test_split):
        all_files = list(self.file_info.keys())
        random.shuffle(all_files)
        val_num = int(len(all_files) * val_split)
        test_num = int(len(all_files) * test_split)
        train_num = len(all_files) - val_num - test_num
        assert train_num > 0, "训练集样本数量需大于0"
        assert val_num > 0, "验证集样本数量需大于0"
        self.train_files = all_files[:train_num]
        self.val_files = all_files[train_num:train_num + val_num]
        self.test_files = all_files[train_num + val_num:]
        self.train_set = set(self.train_files)
        self.val_set = set(self.val_files)
        self.test_set = set(self.test_files)

        for label, file_list in self.label_info.items():
            self.class_train_file_list[label] = list()
            self.class_val_file_list[label] = list()
            self.class_test_file_list[label] = list()
            for f in file_list:
                if f in self.test_set:
                    self.class_test_file_list[label].append(f)
                if f in self.val_set:
                    self.class_val_file_list[label].append(f)
                if f in self.train_set:
                    self.class_train_file_list[label].append(f)
