# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

import os.path as osp
from ..utils import list_files
from .utils import is_pic, get_encoding, check_list_txt
from .datasetbase import DatasetBase


class ClsDataset(DatasetBase):
    def __init__(self, dataset_id, path):
        super().__init__(dataset_id, path)

    def check_dataset(self, source_path):
        self.all_files = list_files(source_path)
        # 对分类数据集进行统计分析
        self.file_info = dict()
        self.label_info = dict()
        # 校验已切分的数据集
        if osp.exists(osp.join(source_path, 'train_list.txt')):
            return self.check_splited_dataset(source_path)

        for f in self.all_files:
            if not is_pic(f):
                continue
            items = osp.split(f)
            if len(items) == 2:
                if " " in items[0]:
                    raise ValueError("类别-{}名称有误，分类数据集中类别名称不应包含空格".format(items[
                        0]))
                if items[0] not in self.label_info:
                    self.label_info[items[0]] = list()
                self.label_info[items[0]].append(f)
                self.file_info[f] = items[0]
        if len(self.label_info) < 2:
            raise ValueError("分类数据集中至少需要包含两种图像类别")
        self.labels = sorted(self.label_info.keys())
        for label in self.labels:
            self.class_train_file_list[label] = list()
            self.class_val_file_list[label] = list()
            self.class_test_file_list[label] = list()
        # 将数据集分析信息dump到本地
        self.dump_statis_info()

    def check_splited_dataset(self, source_path):
        labels_txt = osp.join(source_path, "labels.txt")
        train_list_txt = osp.join(source_path, "train_list.txt")
        val_list_txt = osp.join(source_path, "val_list.txt")
        test_list_txt = osp.join(source_path, "test_list.txt")
        for txt_file in [labels_txt, train_list_txt, val_list_txt]:
            if not osp.exists(txt_file):
                raise Exception(
                    "已切分的数据集下应该包含labels.txt, train_list.txt, val_list.txt文件")
        check_list_txt([train_list_txt, val_list_txt, test_list_txt])

        self.labels = open(
            labels_txt, 'r',
            encoding=get_encoding(labels_txt)).read().strip().split('\n')

        for txt_file in [train_list_txt, val_list_txt, test_list_txt]:
            if not osp.exists(txt_file):
                continue
            with open(txt_file, "r") as f:
                for line in f:
                    items = line.strip().split()

                    if not osp.exists(osp.join(source_path, items[0])):
                        raise Exception("数据目录{}中不存在图片文件{}".format(
                            osp.split(txt_file)[-1], items[0]))
                    dir_name = osp.split(osp.split(items[0])[0])[-1]
                    if dir_name != self.labels[int(items[1])]:
                        raise Exception("labels.txt中label顺序不准确")
                    img_file = osp.split(items[0])[-1]
                    if not is_pic(img_file) or img_file.startswith('.'):
                        raise ValueError("文件{}不是图片格式".format(img_file))
                    self.file_info[items[0]] = self.labels[int(items[1])]

                    if txt_file == train_list_txt:
                        self.train_files.append(items[0])
                        if self.labels[int(items[
                                1])] in self.class_train_file_list:
                            self.class_train_file_list[self.labels[int(items[
                                1])]].append(items[0])
                        else:
                            self.class_train_file_list[self.labels[int(items[
                                1])]] = list()
                            self.class_train_file_list[self.labels[int(items[
                                1])]].append(items[0])
                    elif txt_file == val_list_txt:
                        self.val_files.append(items[0])
                        if self.labels[int(items[
                                1])] in self.class_val_file_list:
                            self.class_val_file_list[self.labels[int(items[
                                1])]].append(items[0])
                        else:
                            self.class_val_file_list[self.labels[int(items[
                                1])]] = list()
                            self.class_val_file_list[self.labels[int(items[
                                1])]].append(items[0])
                    elif txt_file == test_list_txt:
                        self.test_files.append(items[0])
                        if self.labels[int(items[
                                1])] in self.class_test_file_list:
                            self.class_test_file_list[self.labels[int(items[
                                1])]].append(items[0])
                        else:
                            self.class_test_file_list[self.labels[int(items[
                                1])]] = list()
                            self.class_test_file_list[self.labels[int(items[
                                1])]].append(items[0])

        for img_file, label in self.file_info.items():
            if label not in self.label_info:
                self.label_info[label] = list()
            self.label_info[label].append(img_file)

        # 将数据集分析信息dump到本地
        self.dump_statis_info()

    def split(self, val_split, test_split):
        super().split(val_split, test_split)
        with open(
                osp.join(self.path, 'train_list.txt'), mode='w',
                encoding='utf-8') as f:
            for x in self.train_files:
                label = self.file_info[x]
                label_idx = self.labels.index(label)
                f.write('{} {}\n'.format(x, label_idx))
        with open(
                osp.join(self.path, 'val_list.txt'), mode='w',
                encoding='utf-8') as f:
            for x in self.val_files:
                label = self.file_info[x]
                label_idx = self.labels.index(label)
                f.write('{} {}\n'.format(x, label_idx))
        with open(
                osp.join(self.path, 'test_list.txt'), mode='w',
                encoding='utf-8') as f:
            for x in self.test_files:
                label = self.file_info[x]
                label_idx = self.labels.index(label)
                f.write('{} {}\n'.format(x, label_idx))
        with open(
                osp.join(self.path, 'labels.txt'), mode='w',
                encoding='utf-8') as f:
            for l in self.labels:
                f.write('{}\n'.format(l))
        self.dump_statis_info()
