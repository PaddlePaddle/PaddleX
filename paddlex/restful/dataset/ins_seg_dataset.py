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

import os
import os.path as osp
import random
from ..utils import list_files
from .utils import is_pic, replace_ext, MyEncoder, read_coco_ann, get_npy_from_coco_json
from .datasetbase import DatasetBase
import numpy as np
import json
from pycocotools.coco import COCO


class InsSegDataset(DatasetBase):
    def __init__(self, dataset_id, path):
        super().__init__(dataset_id, path)
        self.annotation_dict = None

    def check_dataset(self, source_path):
        if not osp.isdir(osp.join(source_path, 'JPEGImages')):
            raise ValueError("图片文件应该放在{}目录下".format(
                osp.join(source_path, 'JPEGImages')))

        self.all_files = list_files(source_path)
        # 对检测数据集进行统计分析
        self.file_info = dict()
        self.label_info = dict()
        # 若数据集已切分
        if osp.exists(osp.join(source_path, 'train.json')):
            return self.check_splited_dataset(source_path)

        if not osp.exists(osp.join(source_path, 'annotations.json')):
            raise ValueError("标注文件annotations.json应该放在{}目录下".format(
                source_path))

        filename_set = set()
        anno_set = set()
        for f in self.all_files:
            items = osp.split(f)
            if len(items) == 2 and items[0] == "JPEGImages":
                if not is_pic(f) or f.startswith('.'):
                    continue
                filename_set.add(items[1])
        # 解析包含标注信息的json文件
        try:
            coco = COCO(osp.join(source_path, 'annotations.json'))
            img_ids = coco.getImgIds()
            cat_ids = coco.getCatIds()
            anno_ids = coco.getAnnIds()

            catid2clsid = dict({catid: i for i, catid in enumerate(cat_ids)})
            cid2cname = dict({
                clsid: coco.loadCats(catid)[0]['name']
                for catid, clsid in catid2clsid.items()
            })
            for img_id in img_ids:
                img_anno = coco.loadImgs(img_id)[0]
                img_name = osp.split(img_anno['file_name'])[-1]
                anno_set.add(img_name)
                anno_dict = read_coco_ann(img_id, coco, cid2cname, catid2clsid)
                img_path = osp.join("JPEGImages", img_name)
                anno_path = osp.join("Annotations", img_name)
                anno = replace_ext(anno_path, "npy")
                self.file_info[img_path] = anno

                img_class = list(set(anno_dict["gt_class"]))
                for category_name in img_class:
                    if not category_name in self.label_info:
                        self.label_info[category_name] = [img_path]
                    else:
                        self.label_info[category_name].append(img_path)

            for label in sorted(self.label_info.keys()):
                self.labels.append(label)
        except:
            raise Exception("标注文件存在错误")

        if len(anno_set) > len(filename_set):
            sub_list = list(anno_set - filename_set)
            raise Exception("标注文件中{}等{}个信息无对应图片".format(sub_list[0],
                                                        len(sub_list)))

        # 生成每个图片对应的标注信息npy文件
        npy_path = osp.join(self.path, "Annotations")
        get_npy_from_coco_json(coco, npy_path, self.file_info)

        for label in self.labels:
            self.class_train_file_list[label] = list()
            self.class_val_file_list[label] = list()
            self.class_test_file_list[label] = list()

        # 将数据集分析信息dump到本地
        self.dump_statis_info()

    def check_splited_dataset(self, source_path):
        train_files_json = osp.join(source_path, "train.json")
        val_files_json = osp.join(source_path, "val.json")
        test_files_json = osp.join(source_path, "test.json")

        for json_file in [train_files_json, val_files_json]:
            if not osp.exists(json_file):
                raise Exception("已切分的数据集下应该包含train.json, val.json文件")

        filename_set = set()
        anno_set = set()
        # 获取全部图片名称
        for f in self.all_files:
            items = osp.split(f)
            if len(items) == 2 and items[0] == "JPEGImages":
                if not is_pic(f) or f.startswith('.'):
                    continue
                filename_set.add(items[1])

        img_id_index = 0
        anno_id_index = 0
        new_img_list = list()
        new_cat_list = list()
        new_anno_list = list()
        for json_file in [train_files_json, val_files_json, test_files_json]:
            if not osp.exists(json_file):
                continue
            coco = COCO(json_file)
            img_ids = coco.getImgIds()
            cat_ids = coco.getCatIds()
            anno_ids = coco.getAnnIds()

            catid2clsid = dict({catid: i for i, catid in enumerate(cat_ids)})
            clsid2catid = dict({i: catid for i, catid in enumerate(cat_ids)})
            cid2cname = dict({
                clsid: coco.loadCats(catid)[0]['name']
                for catid, clsid in catid2clsid.items()
            })

            # 由原train.json中的category生成新的category信息
            if json_file == train_files_json:
                cname2catid = dict({
                    coco.loadCats(catid)[0]['name']: clsid2catid[clsid]
                    for catid, clsid in catid2clsid.items()
                })
                new_cat_list = coco.loadCats(cat_ids)
            # 获取json中全部标注图片的名字
            for img_id in img_ids:
                img_anno = coco.loadImgs(img_id)[0]
                im_fname = img_anno['file_name']
                anno_set.add(im_fname)

                if json_file == train_files_json:
                    self.train_files.append(osp.join("JPEGImages", im_fname))
                elif json_file == val_files_json:
                    self.val_files.append(osp.join("JPEGImages", im_fname))
                elif json_file == test_files_json:
                    self.test_files.append(osp.join("JPEGImages", im_fname))
                # 获取每张图片的对应标注信息,并记录为npy格式
                anno_dict = read_coco_ann(img_id, coco, cid2cname, catid2clsid)
                img_path = osp.join("JPEGImages", im_fname)
                anno_path = osp.join("Annotations", im_fname)
                anno = replace_ext(anno_path, "npy")
                self.file_info[img_path] = anno

                # 生成label_info
                img_class = list(set(anno_dict["gt_class"]))
                for category_name in img_class:
                    if not category_name in self.label_info:
                        self.label_info[category_name] = [img_path]
                    else:
                        self.label_info[category_name].append(img_path)
                    if json_file == train_files_json:
                        if category_name in self.class_train_file_list:
                            self.class_train_file_list[category_name].append(
                                img_path)
                        else:
                            self.class_train_file_list[category_name] = list()
                            self.class_train_file_list[category_name].append(
                                img_path)
                    elif json_file == val_files_json:
                        if category_name in self.class_val_file_list:
                            self.class_val_file_list[category_name].append(
                                img_path)
                        else:
                            self.class_val_file_list[category_name] = list()
                            self.class_val_file_list[category_name].append(
                                img_path)
                    elif json_file == test_files_json:
                        if category_name in self.class_test_file_list:
                            self.class_test_file_list[category_name].append(
                                img_path)
                        else:
                            self.class_test_file_list[category_name] = list()
                            self.class_test_file_list[category_name].append(
                                img_path)

                # 生成新的图片信息
                new_img = img_anno
                new_img["id"] = img_id_index
                img_id_index += 1
                new_img_list.append(new_img)
                # 生成新的标注信息
                ins_anno_ids = coco.getAnnIds(imgIds=img_id, iscrowd=0)
                for ins_anno_id in ins_anno_ids:
                    anno = coco.loadAnns(ins_anno_id)[0]
                    new_anno = anno
                    new_anno["image_id"] = new_img["id"]
                    new_anno["id"] = anno_id_index
                    anno_id_index += 1
                    cat = coco.loadCats(anno["category_id"])[0]
                    new_anno_list.append(new_anno)

        if len(anno_set) > len(filename_set):
            sub_list = list(anno_set - filename_set)
            raise Exception("标注文件中{}等{}个信息无对应图片".format(sub_list[0],
                                                        len(sub_list)))

        for label in sorted(self.label_info.keys()):
            self.labels.append(label)

        self.annotation_dict = {
            "images": new_img_list,
            "categories": new_cat_list,
            "annotations": new_anno_list
        }

        # 若原数据集已切分，无annotations.json文件
        if not osp.exists(osp.join(self.path, "annotations.json")):
            json_file = open(osp.join(self.path, "annotations.json"), 'w+')
            json.dump(self.annotation_dict, json_file, cls=MyEncoder)
            json_file.close()

        # 生成每个图片对应的标注信息npy文件
        coco = COCO(osp.join(self.path, "annotations.json"))
        npy_path = osp.join(self.path, "Annotations")
        get_npy_from_coco_json(coco, npy_path, self.file_info)

        self.dump_statis_info()

    def split(self, val_split, test_split):
        all_files = list(self.file_info.keys())
        val_num = int(len(all_files) * val_split)
        test_num = int(len(all_files) * test_split)
        train_num = len(all_files) - val_num - test_num
        assert train_num > 0, "训练集样本数量需大于0"
        assert val_num > 0, "验证集样本数量需大于0"
        self.train_files = list()
        self.val_files = list()
        self.test_files = list()

        coco = COCO(osp.join(self.path, 'annotations.json'))
        img_ids = coco.getImgIds()
        cat_ids = coco.getCatIds()
        anno_ids = coco.getAnnIds()
        random.shuffle(img_ids)

        train_files_ids = img_ids[:train_num]
        val_files_ids = img_ids[train_num:train_num + val_num]
        test_files_ids = img_ids[train_num + val_num:]

        for img_id_list in [train_files_ids, val_files_ids, test_files_ids]:
            img_anno_ids = coco.getAnnIds(imgIds=img_id_list, iscrowd=0)
            imgs = coco.loadImgs(img_id_list)
            instances = coco.loadAnns(img_anno_ids)
            categories = coco.loadCats(cat_ids)
            img_dict = {
                "annotations": instances,
                "images": imgs,
                "categories": categories
            }

            if img_id_list == train_files_ids:
                for img in imgs:
                    self.train_files.append(
                        osp.join("JPEGImages", img["file_name"]))
                json_file = open(osp.join(self.path, 'train.json'), 'w+')
                json.dump(img_dict, json_file, cls=MyEncoder)
            elif img_id_list == val_files_ids:
                for img in imgs:
                    self.val_files.append(
                        osp.join("JPEGImages", img["file_name"]))
                json_file = open(osp.join(self.path, 'val.json'), 'w+')
                json.dump(img_dict, json_file, cls=MyEncoder)
            elif img_id_list == test_files_ids:
                for img in imgs:
                    self.test_files.append(
                        osp.join("JPEGImages", img["file_name"]))
                json_file = open(osp.join(self.path, 'test.json'), 'w+')
                json.dump(img_dict, json_file, cls=MyEncoder)

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

        self.dump_statis_info()
