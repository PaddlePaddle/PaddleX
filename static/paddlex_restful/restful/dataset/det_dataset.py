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
from .utils import is_pic, replace_ext, get_encoding, check_list_txt
from .datasetbase import DatasetBase
import xml.etree.ElementTree as ET


class DetDataset(DatasetBase):
    def __init__(self, dataset_id, path):
        super().__init__(dataset_id, path)

    def check_dataset(self, source_path):
        if not osp.isdir(osp.join(source_path, 'Annotations')):
            raise ValueError("标注文件应该放在{}目录下".format(
                osp.join(source_path, 'Annotations')))
        if not osp.isdir(osp.join(source_path, 'JPEGImages')):
            raise ValueError("图片文件应该放在{}目录下".format(
                osp.join(source_path, 'JPEGImages')))

        self.all_files = list_files(source_path)
        # 对检测数据集进行统计分析
        self.file_info = dict()
        self.label_info = dict()

        if osp.exists(osp.join(source_path, 'train_list.txt')):
            return self.check_splited_dataset(source_path)

        for f in self.all_files:
            if not is_pic(f):
                continue
            items = osp.split(f)
            if len(items) == 2 and items[0] == "JPEGImages":
                anno_name = replace_ext(items[1], "xml")
                full_anno_path = osp.join(
                    (osp.join(source_path, 'Annotations')), anno_name)
                if osp.exists(full_anno_path):
                    self.file_info[f] = osp.join('Annotations', anno_name)

                # 解析XML文件，获取类别信息
                try:
                    tree = ET.parse(full_anno_path)
                except:
                    raise Exception("文件{}不是一个良构的xml文件".format(anno_name))
                objs = tree.findall('object')
                for i, obj in enumerate(objs):
                    cname = obj.find('name').text
                    if cname not in self.label_info:
                        self.label_info[cname] = list()
                    if f not in self.label_info[cname]:
                        self.label_info[cname].append(f)

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
                    img_file, xml_file = [items[0], items[1]]

                    if not osp.isfile(osp.join(source_path, xml_file)):
                        raise ValueError("数据目录{}中不存在标注文件{}".format(
                            osp.split(txt_file)[-1], xml_file))
                    if not osp.isfile(osp.join(source_path, img_file)):
                        raise ValueError("数据目录{}中不存在图片文件{}".format(
                            osp.split(txt_file)[-1], img_file))
                    if not xml_file.split('.')[-1] == 'xml':
                        raise ValueError("标注文件{}不是xml文件".format(xml_file))
                    img_file_name = osp.split(img_file)[-1]
                    if not is_pic(img_file_name) or img_file_name.startswith(
                            '.'):
                        raise ValueError("文件{}不是图片格式".format(img_file))

                    self.file_info[img_file] = xml_file

                    if txt_file == train_list_txt:
                        self.train_files.append(img_file)
                    elif txt_file == val_list_txt:
                        self.val_files.append(img_file)
                    elif txt_file == test_list_txt:
                        self.test_files.append(img_file)

                    try:
                        tree = ET.parse(osp.join(source_path, xml_file))
                    except:
                        raise Exception("文件{}不是一个良构的xml文件".format(xml_file))
                    objs = tree.findall('object')
                    for i, obj in enumerate(objs):
                        cname = obj.find('name').text
                        if cname in self.labels:
                            if cname not in self.label_info:
                                self.label_info[cname] = list()
                            if img_file not in self.label_info[cname]:
                                self.label_info[cname].append(img_file)
                                if txt_file == train_list_txt:
                                    if cname in self.class_train_file_list:
                                        self.class_train_file_list[
                                            cname].append(img_file)
                                    else:
                                        self.class_train_file_list[
                                            cname] = list()
                                        self.class_train_file_list[
                                            cname].append(img_file)
                                elif txt_file == val_list_txt:
                                    if cname in self.class_val_file_list:
                                        self.class_val_file_list[cname].append(
                                            img_file)
                                    else:
                                        self.class_val_file_list[cname] = list(
                                        )
                                        self.class_val_file_list[cname].append(
                                            img_file)
                                elif txt_file == test_list_txt:
                                    if cname in self.class_test_file_list:
                                        self.class_test_file_list[
                                            cname].append(img_file)
                                    else:
                                        self.class_test_file_list[
                                            cname] = list()
                                        self.class_test_file_list[
                                            cname].append(img_file)
                        else:
                            raise Exception("文件{}与labels.txt文件信息不对应".format(
                                xml_file))

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
        with open(
                osp.join(self.path, 'labels.txt'), mode='w',
                encoding='utf-8') as f:
            for l in self.labels:
                f.write('{}\n'.format(l))
        self.dump_statis_info()
