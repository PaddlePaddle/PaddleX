# !/usr/bin/env python3
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2024 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Author: PaddlePaddle Authors
"""
import os
import shutil
import json
import random
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

from .....utils.file_interface import custom_open
from .....utils.errors import ConvertFailedError


class Indexer(object):
    """ Indexer """

    def __init__(self):
        """ init indexer """
        self._map = {}
        self.idx = 0

    def get_id(self, key):
        """ get id by key """
        if key not in self._map:
            self.idx += 1
            self._map[key] = self.idx
        return self._map[key]

    def get_list(self, key_name):
        """ return list containing key and id """
        map_list = []
        for key in self._map:
            val = self._map[key]
            map_list.append({key_name: key, 'id': val})
        return map_list


class Extension(object):
    """ Extension """

    def __init__(self, exts_list):
        """ init extension """
        self._exts_list = ['.' + ext for ext in exts_list]

    def __iter__(self):
        """ iterator """
        return iter(self._exts_list)

    def update(self, ext):
        """ update extension """
        self._exts_list.remove(ext)
        self._exts_list.insert(0, ext)


def check_src_dataset(root_dir, dataset_type):
    """ check src dataset format validity """
    if dataset_type == "LabelMe":
        anno_suffix = ".json"
    else:
        raise ConvertFailedError(
            message=f"数据格式转换失败！不支持{dataset_type}格式数据集。当前仅支持 LabelMe 格式。")

    err_msg_prefix = f"数据格式转换失败！请参考上述`{dataset_type}格式数据集示例`检查待转换数据集格式。"

    anno_map = {}
    for dst_anno, src_anno in [("instance_train.json", "train_anno_list.txt"),
                               ("instance_val.json", "val_anno_list.txt"),
                               ("instance_test.json", "test_anno_list.txt")]:
        src_anno_path = os.path.join(root_dir, src_anno)
        if not os.path.exists(src_anno_path):
            if dst_anno == "instance_train.json":
                raise ConvertFailedError(
                    message=f"{err_msg_prefix}保证{src_anno_path}文件存在。")
            continue
        with custom_open(src_anno_path, 'r') as f:
            anno_list = f.readlines()
        for anno_fn in anno_list:
            anno_fn = anno_fn.strip().split(' ')[-1]
            anno_path = os.path.join(root_dir, anno_fn)
            if not os.path.exists(anno_path):
                raise ConvertFailedError(
                    message=f"{err_msg_prefix}保证\"{src_anno_path}\"中的\"{anno_fn}\"文件存在。"
                )
        anno_map[dst_anno] = src_anno_path
    return anno_map


def convert(dataset_type, input_dir, output_dir):
    """ convert dataset to coco format """
    # check format validity
    anno_map = check_src_dataset(input_dir, dataset_type)
    if dataset_type == "LabelMe":
        convert_labelme_dataset(input_dir, anno_map, output_dir)
    else:
        raise ValueError
    return output_dir


def split_anno_list(root_dir, anno_map, output_dir):
    """Split anno list to 80% train and 20% val """

    train_anno_list = []
    val_anno_list = []
    if not os.path.exists(os.path.join(output_dir, "tmp")):
        os.makedirs(os.path.join(output_dir, "tmp"))
    with custom_open(anno_map["instance_train.json"], 'r') as f:
        src_anno = f.readlines()
    random.shuffle(src_anno)
    train_anno_list = src_anno[:int(len(src_anno) * 0.8)]
    val_anno_list = src_anno[int(len(src_anno) * 0.8):]
    with custom_open(
            os.path.join(output_dir, "tmp", "train_anno_list.txt"), 'w') as f:
        f.writelines(train_anno_list)
    with custom_open(
            os.path.join(output_dir, "tmp", "val_anno_list.txt"), 'w') as f:
        f.writelines(val_anno_list)
    anno_map["instance_train.json"] = os.path.join(output_dir, "tmp",
                                                   "train_anno_list.txt")
    anno_map["instance_val.json"] = os.path.join(output_dir, "tmp",
                                                 "val_anno_list.txt")
    print(
        f"{os.path.join(root_dir, 'val_anno_list.txt')}不存在，数据集已默认按照80%训练集，20%验证集划分."
    )
    return anno_map


def convert_labelme_dataset(root_dir, anno_map, output_dir):
    """ convert dataset labeled by LabelMe to coco format """
    label_indexer = Indexer()
    img_indexer = Indexer()

    if os.path.exists(output_dir):
        print(f"The output_dir ({output_dir}) exists, would be override.")
        shutil.rmtree(output_dir)
    print(
        f"Start copying the original files (such as images, etc.) to {output_dir}."
    )
    shutil.copytree(root_dir, output_dir)

    annotations_dir = os.path.join(output_dir, "annotations")
    if not os.path.exists(annotations_dir):
        os.makedirs(annotations_dir)

    # 不存在val_anno_list，对原始数据集进行划分
    if 'instance_val.json' not in anno_map:
        anno_map = split_anno_list(root_dir, anno_map, output_dir)

    for dst_anno in anno_map:
        labelme2coco(label_indexer, img_indexer, root_dir, anno_map[dst_anno],
                     os.path.join(annotations_dir, dst_anno))


def labelme2coco(label_indexer, img_indexer, root_dir, anno_path, save_path):
    """ convert json files generated by LabelMe to coco format and save to files """
    import pycocotools.mask as mask_util

    with custom_open(anno_path, 'r') as f:
        json_list = f.readlines()

    anno_num = 0
    anno_list = []
    image_list = []
    print(f"Start loading json annotation files from {anno_path} ...")
    for json_path in tqdm(json_list):
        json_path = json_path.strip()
        assert json_path.endswith(".json"), json_path
        with custom_open(os.path.join(root_dir, json_path.strip()), 'r') as f:
            labelme_data = json.load(f)

        img_id = img_indexer.get_id(labelme_data['imagePath'])
        height = labelme_data['imageHeight']
        width = labelme_data['imageWidth']
        image_list.append({
            'id': img_id,
            'file_name': labelme_data['imagePath'].split('/')[-1],
            'width': width,
            'height': height,
        })

        for shape in labelme_data['shapes']:
            assert shape[
                'shape_type'] == 'polygon', "Only polygon are supported."
            category_id = label_indexer.get_id(shape['label'])
            points = shape["points"]
            segmentation = [np.asarray(points).flatten().tolist()]
            mask = points_to_mask([height, width], points)
            mask = np.asfortranarray(mask.astype(np.uint8))
            mask = mask_util.encode(mask)
            area = float(mask_util.area(mask))
            bbox = mask_util.toBbox(mask).flatten().tolist()

            anno_num += 1
            anno_list.append({
                'image_id': img_id,
                'bbox': bbox,
                'segmentation': segmentation,
                'category_id': category_id,
                'id': anno_num,
                'iscrowd': 0,
                'area': area,
                'ignore': 0
            })

    category_list = label_indexer.get_list(key_name="name")
    data_coco = {
        'images': image_list,
        'categories': category_list,
        'annotations': anno_list
    }

    with custom_open(save_path, 'w') as f:
        json.dump(data_coco, f)
    print(f"The converted annotations has been save to {save_path}.")


def points_to_mask(img_shape, points):
    """convert polygon points to binary mask"""
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = Image.fromarray(mask)
    draw = ImageDraw.Draw(mask)
    xy = [tuple(point) for point in points]
    assert len(xy) > 2, "Polygon must have points more than 2"
    draw.polygon(xy=xy, outline=1, fill=1)
    mask = np.asarray(mask, dtype=bool)
    return mask
