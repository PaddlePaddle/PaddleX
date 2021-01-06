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
import cv2
from ..utils import list_files
from .utils import is_pic, replace_ext, get_encoding, check_list_txt, read_seg_ann
from .datasetbase import DatasetBase


class SegDataset(DatasetBase):
    def __init__(self, dataset_id, path):
        super().__init__(dataset_id, path)

    def check_dataset(self, source_path):
        if not osp.isdir(osp.join(source_path, 'Annotations')):
            raise ValueError("标注文件应该放在{}目录下".format(
                osp.join(source_path, 'Annotations')))
        if not osp.isdir(osp.join(source_path, 'JPEGImages')):
            raise ValueError("图片文件应该放在{}目录下".format(
                osp.join(source_path, 'JPEGImages')))

        labels_txt = osp.join(source_path, 'labels.txt')
        if osp.exists(labels_txt):
            with open(labels_txt, encoding=get_encoding(labels_txt)) as fid:
                lines = fid.readlines()
                for line in lines:
                    self.labels.append(line.strip())

        self.all_files = list_files(source_path)
        # 对语义分割数据集进行统计分析
        self.file_info = dict()
        self.label_info = dict()

        if osp.exists(osp.join(source_path, 'train_list.txt')):
            return self.check_splited_dataset(source_path)

        for f in self.all_files:
            if not is_pic(f):
                continue
            items = osp.split(f)
            if len(items) == 2 and items[0] == "JPEGImages":
                anno_name = replace_ext(items[1], "png")
                full_anno_path = osp.join(
                    (osp.join(source_path, 'Annotations')), anno_name)
                if osp.exists(full_anno_path):
                    self.file_info[f] = osp.join('Annotations', anno_name)

                # 解析PNG标注文件，获取类别信息
                labels, ann_img_shape = read_seg_ann(full_anno_path)
                img_shape = cv2.imread(osp.join(source_path, f)).shape
                if img_shape[0] != ann_img_shape[0] or img_shape[
                        1] != ann_img_shape[1]:
                    raise ValueError("文件{}与标注图片尺寸不一致".format(items[1]))
                for i in labels:
                    if str(i) not in self.label_info:
                        self.label_info[str(i)] = list()
                    self.label_info[str(i)].append(f)

        # 如果类标签的最大值大于类别数，统计相应的类别为零
        max_label = max([int(i) for i in self.label_info]) + 1
        for i in range(max_label):
            if str(i) not in self.label_info:
                self.label_info[str(i)] = list()

        if len(self.labels) == 0:
            self.labels = [int(i) for i in self.label_info]
            self.labels.sort()
            self.labels = [str(i) for i in self.labels]
        else:
            keys = list(self.label_info.keys())
            try:
                for key in keys:
                    label = self.labels[int(key)]
                    self.label_info[label] = self.label_info.pop(key)
            except:
                raise ValueError("标注信息与实际类别不一致")

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
        for txt_file in [train_list_txt, val_list_txt]:
            if not osp.exists(txt_file):
                raise Exception("已切分的数据集下应该包含train_list.txt, val_list.txt文件")
        check_list_txt([train_list_txt, val_list_txt, test_list_txt])

        if osp.exists(labels_txt):
            self.labels = open(
                labels_txt, 'r',
                encoding=get_encoding(labels_txt)).read().strip().split('\n')

        for txt_file in [train_list_txt, val_list_txt, test_list_txt]:
            if not osp.exists(txt_file):
                continue
            with open(txt_file, "r") as f:
                for line in f:
                    items = line.strip().split()
                    img_file, png_file = [items[0], items[1]]

                    if not osp.isfile(osp.join(source_path, png_file)):
                        raise ValueError("数据目录{}中不存在标注文件{}".format(
                            osp.split(txt_file)[-1], png_file))
                    if not osp.isfile(osp.join(source_path, img_file)):
                        raise ValueError("数据目录{}中不存在图片文件{}".format(
                            osp.split(txt_file)[-1], img_file))
                    if not png_file.split('.')[-1] == 'png':
                        raise ValueError("标注文件{}不是png文件".format(png_file))
                    img_file_name = osp.split(img_file)[-1]
                    if not is_pic(img_file_name) or img_file_name.startswith(
                            '.'):
                        raise ValueError("文件{}不是图片格式".format(img_file_name))

                    self.file_info[img_file] = png_file

                    if txt_file == train_list_txt:
                        self.train_files.append(img_file)
                    elif txt_file == val_list_txt:
                        self.val_files.append(img_file)
                    elif txt_file == test_list_txt:
                        self.test_files.append(img_file)

                    # 解析PNG标注文件
                    labels, ann_img_shape = read_seg_ann(
                        osp.join(source_path, png_file))
                    img_shape = cv2.imread(osp.join(source_path,
                                                    img_file)).shape
                    if img_shape[0] != ann_img_shape[0] or img_shape[
                            1] != ann_img_shape[1]:
                        raise ValueError("文件{}与标注图片尺寸不一致".format(
                            img_file_name))
                    for i in labels:
                        if str(i) not in self.label_info:
                            self.label_info[str(i)] = list()
                        self.label_info[str(i)].append(img_file)

        # 如果类标签的最大值大于类别数，统计相应的类别为零
        max_label = max([int(i) for i in self.label_info]) + 1
        for i in range(max_label):
            if str(i) not in self.label_info:
                self.label_info[str(i)] = list()

        if len(self.labels) == 0:
            self.labels = [int(i) for i in self.label_info]
            self.labels.sort()
            self.labels = [str(i) for i in self.labels]
        else:
            keys = list(self.label_info.keys())
            try:
                for key in keys:
                    label = self.labels[int(key)]
                    self.label_info[label] = self.label_info.pop(key)
            except:
                raise ValueError("标注信息与实际类别不一致")

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

        # 将数据集分析信息dump到本地
        self.dump_statis_info()

    def split(self, val_split, test_split):
        super().split(val_split, test_split)
        with open(
                osp.join(self.path, 'train_list.txt'), mode='w',
                encoding='utf-8') as f:
            for x in self.train_files:
                label = self.file_info[x]
                f.write('{} {}\n'.format(x, label))
        with open(
                osp.join(self.path, 'val_list.txt'), mode='w',
                encoding='utf-8') as f:
            for x in self.val_files:
                label = self.file_info[x]
                f.write('{} {}\n'.format(x, label))
        with open(
                osp.join(self.path, 'test_list.txt'), mode='w',
                encoding='utf-8') as f:
            for x in self.test_files:
                label = self.file_info[x]
                f.write('{} {}\n'.format(x, label))
        if not osp.exists(osp.join(self.path, 'labels.txt')):
            with open(
                    osp.join(self.path, 'labels.txt'), mode='w',
                    encoding='utf-8') as f:
                max_label = max([int(i) for i in self.labels]) + 1
                for i in range(max_label):
                    f.write('{}\n'.format(str(i)))
        self.dump_statis_info()
