#!/usr/bin/env python
# coding: utf-8
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import cv2
import json
import os
import os.path as osp
import shutil
import re
import numpy as np
import PIL.ImageDraw
import xml.etree.ElementTree as ET
from .base import MyEncoder, is_pic, get_encoding
from paddlex.utils import path_normalization
import paddlex.utils.logging as logging


class X2COCO(object):
    def __init__(self):
        self.images_list = []
        self.categories_list = []
        self.annotations_list = []

    def generate_categories_field(self, label, labels_list):
        category = {}
        category["supercategory"] = "component"
        category["id"] = len(labels_list) + 1
        category["name"] = label
        return category

    def generate_rectangle_anns_field(self, points, label, image_id, object_id,
                                      label_to_num):
        annotation = {}
        seg_points = np.asarray(points).copy()
        seg_points[1, :] = np.asarray(points)[2, :]
        seg_points[2, :] = np.asarray(points)[1, :]
        annotation["segmentation"] = [list(seg_points.flatten())]
        annotation["iscrowd"] = 0
        annotation["image_id"] = image_id + 1
        annotation["bbox"] = list(
            map(float, [
                points[0][0], points[0][1], points[1][0] - points[0][0],
                points[1][1] - points[0][1]
            ]))
        annotation["area"] = annotation["bbox"][2] * annotation["bbox"][3]
        annotation["category_id"] = label_to_num[label]
        annotation["id"] = object_id + 1
        return annotation

    def convert(self, image_dir, json_dir, dataset_save_dir):
        """转换。
        Args:
            image_dir (str): 图像文件存放的路径。
            json_dir (str): 与每张图像对应的json文件的存放路径。
            dataset_save_dir (str): 转换后数据集存放路径。
        """
        assert osp.exists(image_dir), "he image folder does not exist!"
        assert osp.exists(json_dir), "The json folder does not exist!"
        assert osp.exists(dataset_save_dir), "The save folder does not exist!"
        # Convert the image files.
        new_image_dir = osp.join(dataset_save_dir, "JPEGImages")
        if osp.exists(new_image_dir):
            shutil.rmtree(new_image_dir)
        os.makedirs(new_image_dir)
        for img_name in os.listdir(image_dir):
            if is_pic(img_name):
                shutil.copyfile(
                    osp.join(image_dir, img_name),
                    osp.join(new_image_dir, img_name))
        # Convert the json files.
        self.parse_json(new_image_dir, json_dir)
        coco_data = {}
        coco_data["images"] = self.images_list
        coco_data["categories"] = self.categories_list
        coco_data["annotations"] = self.annotations_list
        json_path = osp.join(dataset_save_dir, "annotations.json")
        f = open(json_path, "w")
        json.dump(coco_data, f, indent=4, cls=MyEncoder)
        f.close()


class LabelMe2COCO(X2COCO):
    """将使用LabelMe标注的数据集转换为COCO数据集。
    """

    def __init__(self):
        super(LabelMe2COCO, self).__init__()

    def generate_images_field(self, json_info, image_file, image_id):
        image = {}
        image["height"] = json_info["imageHeight"]
        image["width"] = json_info["imageWidth"]
        image["id"] = image_id + 1
        json_img_path = path_normalization(json_info["imagePath"])
        json_info["imagePath"] = osp.join(
            osp.split(json_img_path)[0], image_file)
        image["file_name"] = osp.split(json_info["imagePath"])[-1]
        return image

    def generate_polygon_anns_field(self, height, width, points, label,
                                    image_id, object_id, label_to_num):
        annotation = {}
        annotation["segmentation"] = [list(np.asarray(points).flatten())]
        annotation["iscrowd"] = 0
        annotation["image_id"] = image_id + 1
        annotation["bbox"] = list(
            map(float, self.get_bbox(height, width, points)))
        annotation["area"] = annotation["bbox"][2] * annotation["bbox"][3]
        annotation["category_id"] = label_to_num[label]
        annotation["id"] = object_id + 1
        return annotation

    def get_bbox(self, height, width, points):
        polygons = points
        mask = np.zeros([height, width], dtype=np.uint8)
        mask = PIL.Image.fromarray(mask)
        xy = list(map(tuple, polygons))
        PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
        mask = np.array(mask, dtype=bool)
        index = np.argwhere(mask == 1)
        rows = index[:, 0]
        clos = index[:, 1]
        left_top_r = np.min(rows)
        left_top_c = np.min(clos)
        right_bottom_r = np.max(rows)
        right_bottom_c = np.max(clos)
        return [
            left_top_c, left_top_r, right_bottom_c - left_top_c,
            right_bottom_r - left_top_r
        ]

    def parse_json(self, img_dir, json_dir):
        image_id = -1
        object_id = -1
        labels_list = []
        label_to_num = {}
        for img_file in os.listdir(img_dir):
            img_name_part = osp.splitext(img_file)[0]
            json_file = osp.join(json_dir, img_name_part + ".json")
            if not osp.exists(json_file):
                os.remove(osp.join(img_dir, img_file))
                continue
            image_id = image_id + 1
            with open(json_file, mode='r', \
                              encoding=get_encoding(json_file)) as j:
                json_info = json.load(j)
                img_info = self.generate_images_field(json_info, img_file,
                                                      image_id)
                self.images_list.append(img_info)
                for shapes in json_info["shapes"]:
                    object_id = object_id + 1
                    label = shapes["label"]
                    if label not in labels_list:
                        self.categories_list.append(\
                            self.generate_categories_field(label, labels_list))
                        labels_list.append(label)
                        label_to_num[label] = len(labels_list)
                    points = shapes["points"]
                    p_type = shapes["shape_type"]
                    if p_type == "polygon":
                        self.annotations_list.append(
                            self.generate_polygon_anns_field(
                                json_info["imageHeight"], json_info[
                                    "imageWidth"], points, label, image_id,
                                object_id, label_to_num))
                    if p_type == "rectangle":
                        points.append([points[0][0], points[1][1]])
                        points.append([points[1][0], points[0][1]])
                        self.annotations_list.append(
                            self.generate_rectangle_anns_field(
                                points, label, image_id, object_id,
                                label_to_num))


class EasyData2COCO(X2COCO):
    """将使用EasyData标注的检测或分割数据集转换为COCO数据集。
    """

    def __init__(self):
        super(EasyData2COCO, self).__init__()

    def generate_images_field(self, img_path, image_id):
        image = {}
        img = cv2.imread(img_path)
        image["height"] = img.shape[0]
        image["width"] = img.shape[1]
        image["id"] = image_id + 1
        img_path = path_normalization(img_path)
        image["file_name"] = osp.split(img_path)[-1]
        return image

    def generate_polygon_anns_field(self, points, segmentation, label,
                                    image_id, object_id, label_to_num):
        annotation = {}
        annotation["segmentation"] = segmentation
        annotation["iscrowd"] = 1 if len(segmentation) > 1 else 0
        annotation["image_id"] = image_id + 1
        annotation["bbox"] = list(
            map(float, [
                points[0][0], points[0][1], points[1][0] - points[0][0],
                points[1][1] - points[0][1]
            ]))
        annotation["area"] = annotation["bbox"][2] * annotation["bbox"][3]
        annotation["category_id"] = label_to_num[label]
        annotation["id"] = object_id + 1
        return annotation

    def parse_json(self, img_dir, json_dir):
        from pycocotools.mask import decode
        image_id = -1
        object_id = -1
        labels_list = []
        label_to_num = {}
        for img_file in os.listdir(img_dir):
            img_name_part = osp.splitext(img_file)[0]
            json_file = osp.join(json_dir, img_name_part + ".json")
            if not osp.exists(json_file):
                os.remove(osp.join(img_dir, img_file))
                continue
            image_id = image_id + 1
            with open(json_file, mode='r', \
                              encoding=get_encoding(json_file)) as j:
                json_info = json.load(j)
                img_info = self.generate_images_field(
                    osp.join(img_dir, img_file), image_id)
                self.images_list.append(img_info)
                for shapes in json_info["labels"]:
                    object_id = object_id + 1
                    label = shapes["name"]
                    if label not in labels_list:
                        self.categories_list.append(\
                            self.generate_categories_field(label, labels_list))
                        labels_list.append(label)
                        label_to_num[label] = len(labels_list)
                    points = [[shapes["x1"], shapes["y1"]],
                              [shapes["x2"], shapes["y2"]]]
                    if "mask" not in shapes:
                        points.append([points[0][0], points[1][1]])
                        points.append([points[1][0], points[0][1]])
                        self.annotations_list.append(
                            self.generate_rectangle_anns_field(
                                points, label, image_id, object_id,
                                label_to_num))
                    else:
                        mask_dict = {}
                        mask_dict[
                            'size'] = [img_info["height"], img_info["width"]]
                        mask_dict['counts'] = shapes['mask'].encode()
                        mask = decode(mask_dict)
                        contours, hierarchy = cv2.findContours(
                            (mask).astype(np.uint8), cv2.RETR_TREE,
                            cv2.CHAIN_APPROX_SIMPLE)
                        segmentation = []
                        for contour in contours:
                            contour_list = contour.flatten().tolist()
                            if len(contour_list) > 4:
                                segmentation.append(contour_list)
                        self.annotations_list.append(
                            self.generate_polygon_anns_field(
                                points, segmentation, label, image_id,
                                object_id, label_to_num))


class JingLing2COCO(X2COCO):
    """将使用EasyData标注的检测或分割数据集转换为COCO数据集。
    """

    def __init__(self):
        super(JingLing2COCO, self).__init__()

    def generate_images_field(self, json_info, image_id):
        image = {}
        image["height"] = json_info["size"]["height"]
        image["width"] = json_info["size"]["width"]
        image["id"] = image_id + 1
        json_info["path"] = path_normalization(json_info["path"])
        image["file_name"] = osp.split(json_info["path"])[-1]
        return image

    def generate_polygon_anns_field(self, height, width, points, label,
                                    image_id, object_id, label_to_num):
        annotation = {}
        annotation["segmentation"] = [list(np.asarray(points).flatten())]
        annotation["iscrowd"] = 0
        annotation["image_id"] = image_id + 1
        annotation["bbox"] = list(
            map(float, self.get_bbox(height, width, points)))
        annotation["area"] = annotation["bbox"][2] * annotation["bbox"][3]
        annotation["category_id"] = label_to_num[label]
        annotation["id"] = object_id + 1
        return annotation

    def get_bbox(self, height, width, points):
        polygons = points
        mask = np.zeros([height, width], dtype=np.uint8)
        mask = PIL.Image.fromarray(mask)
        xy = list(map(tuple, polygons))
        PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
        mask = np.array(mask, dtype=bool)
        index = np.argwhere(mask == 1)
        rows = index[:, 0]
        clos = index[:, 1]
        left_top_r = np.min(rows)
        left_top_c = np.min(clos)
        right_bottom_r = np.max(rows)
        right_bottom_c = np.max(clos)
        return [
            left_top_c, left_top_r, right_bottom_c - left_top_c,
            right_bottom_r - left_top_r
        ]

    def parse_json(self, img_dir, json_dir):
        image_id = -1
        object_id = -1
        labels_list = []
        label_to_num = {}
        for img_file in os.listdir(img_dir):
            img_name_part = osp.splitext(img_file)[0]
            json_file = osp.join(json_dir, img_name_part + ".json")
            if not osp.exists(json_file):
                os.remove(osp.join(img_dir, img_file))
                continue
            image_id = image_id + 1
            with open(json_file, mode='r', \
                              encoding=get_encoding(json_file)) as j:
                json_info = json.load(j)
                img_info = self.generate_images_field(json_info, image_id)
                self.images_list.append(img_info)
                anns_type = "bndbox"
                for i, obj in enumerate(json_info["outputs"]["object"]):
                    if i == 0:
                        if "polygon" in obj:
                            anns_type = "polygon"
                    else:
                        if anns_type not in obj:
                            continue
                    object_id = object_id + 1
                    label = obj["name"]
                    if label not in labels_list:
                        self.categories_list.append(\
                            self.generate_categories_field(label, labels_list))
                        labels_list.append(label)
                        label_to_num[label] = len(labels_list)
                    if anns_type == "polygon":
                        points = []
                        for j in range(int(len(obj["polygon"]) / 2.0)):
                            points.append([
                                obj["polygon"]["x" + str(j + 1)],
                                obj["polygon"]["y" + str(j + 1)]
                            ])
                        self.annotations_list.append(
                            self.generate_polygon_anns_field(
                                json_info["size"]["height"], json_info["size"][
                                    "width"], points, label, image_id,
                                object_id, label_to_num))
                    if anns_type == "bndbox":
                        points = []
                        points.append(
                            [obj["bndbox"]["xmin"], obj["bndbox"]["ymin"]])
                        points.append(
                            [obj["bndbox"]["xmax"], obj["bndbox"]["ymax"]])
                        points.append(
                            [obj["bndbox"]["xmin"], obj["bndbox"]["ymax"]])
                        points.append(
                            [obj["bndbox"]["xmax"], obj["bndbox"]["ymin"]])
                        self.annotations_list.append(
                            self.generate_rectangle_anns_field(
                                points, label, image_id, object_id,
                                label_to_num))


class VOC2COCO(X2COCO):
    """将使用VOC标注的数据集转换为COCO数据集。
    """

    def __init__(self):
        super(VOC2COCO, self).__init__()

    def generate_categories_field(self, label, labels_list):
        category = {}
        category["supercategory"] = "component"
        category["id"] = len(labels_list) + 1
        category["name"] = label
        return category

    def generate_images_field(self, xml_info, image_file, image_id):
        image = {}
        image["height"] = xml_info["imageHeight"]
        image["width"] = xml_info["imageWidth"]
        image["id"] = image_id + 1
        image["imagePath"] = image_file
        image["file_name"] = osp.split(image_file)[-1]
        return image

    def generate_label_list(self, xml_dir):
        xml_dir_dir = os.path.abspath(
            os.path.join(os.path.dirname(xml_dir), os.path.pardir))
        self.labels_list = []
        self.label_to_num = {}
        if osp.exists(osp.join(xml_dir_dir, 'labels.txt')):
            with open(osp.join(xml_dir_dir, 'labels.txt'), 'r') as fr:
                while True:
                    label = fr.readline().strip()
                    if not label:
                        break
                    if label not in self.labels_list:
                        self.categories_list.append(\
                            self.generate_categories_field(label, self.labels_list))
                        self.labels_list.append(label)
                        self.label_to_num[label] = len(self.labels_list)
            return
        logging.info(
            'labels.txt is not in the folder {}, so categories are ordered randomly in annotation.json.'.
            format(xml_dir_dir))
        return

    def parse_xml(self, xml_file):
        xml_info = {'im_info': {}, 'annotations': []}
        tree = ET.parse(xml_file)
        pattern = re.compile('<object>', re.IGNORECASE)
        obj_match = pattern.findall(str(ET.tostringlist(tree.getroot())))
        obj_tag = obj_match[0][1:-1]
        objs = tree.findall(obj_tag)
        pattern = re.compile('<size>', re.IGNORECASE)
        size_tag = pattern.findall(str(ET.tostringlist(tree.getroot())))[0][1:
                                                                            -1]
        size_element = tree.find(size_tag)
        pattern = re.compile('<width>', re.IGNORECASE)
        width_tag = pattern.findall(str(ET.tostringlist(size_element)))[0][1:
                                                                           -1]
        im_w = float(size_element.find(width_tag).text)
        pattern = re.compile('<height>', re.IGNORECASE)
        height_tag = pattern.findall(str(ET.tostringlist(size_element)))[0][1:
                                                                            -1]
        im_h = float(size_element.find(height_tag).text)
        xml_info['im_info']['imageWidth'] = im_w
        xml_info['im_info']['imageHeight'] = im_h
        for i, obj in enumerate(objs):
            pattern = re.compile('<name>', re.IGNORECASE)
            name_tag = pattern.findall(str(ET.tostringlist(obj)))[0][1:-1]
            cname = obj.find(name_tag).text.strip()
            pattern = re.compile('<bndbox>', re.IGNORECASE)
            box_tag = pattern.findall(str(ET.tostringlist(obj)))[0][1:-1]
            box_element = obj.find(box_tag)
            pattern = re.compile('<xmin>', re.IGNORECASE)
            xmin_tag = pattern.findall(str(ET.tostringlist(box_element)))[0][
                1:-1]
            x1 = float(box_element.find(xmin_tag).text)
            pattern = re.compile('<ymin>', re.IGNORECASE)
            ymin_tag = pattern.findall(str(ET.tostringlist(box_element)))[0][
                1:-1]
            y1 = float(box_element.find(ymin_tag).text)
            pattern = re.compile('<xmax>', re.IGNORECASE)
            xmax_tag = pattern.findall(str(ET.tostringlist(box_element)))[0][
                1:-1]
            x2 = float(box_element.find(xmax_tag).text)
            pattern = re.compile('<ymax>', re.IGNORECASE)
            ymax_tag = pattern.findall(str(ET.tostringlist(box_element)))[0][
                1:-1]
            y2 = float(box_element.find(ymax_tag).text)
            x1 = max(0, x1)
            y1 = max(0, y1)
            if im_w > 0.5 and im_h > 0.5:
                x2 = min(im_w - 1, x2)
                y2 = min(im_h - 1, y2)
            xml_info['annotations'].append({
                'bbox': [[x1, y1], [x2, y2], [x1, y2], [x2, y1]],
                'category': cname,
            })
        return xml_info

    def parse_json(self, img_dir, xml_dir, file_list=None):
        image_id = -1
        object_id = -1
        self.generate_label_list(xml_dir)
        for img_file in os.listdir(img_dir):
            if file_list is not None and img_file not in file_list:
                continue
            img_name_part = osp.splitext(img_file)[0]
            xml_file = osp.join(xml_dir, img_name_part + ".xml")
            if not osp.exists(xml_file):
                os.remove(osp.join(img_dir, img_file))
                continue
            image_id = image_id + 1
            xml_info = self.parse_xml(xml_file)
            img_info = self.generate_images_field(xml_info['im_info'],
                                                  osp.join(img_dir, img_file),
                                                  image_id)
            self.images_list.append(img_info)
            annos = xml_info['annotations']
            for anno in annos:
                object_id = object_id + 1
                label = anno["category"]
                if label not in self.labels_list:
                    self.categories_list.append(\
                        self.generate_categories_field(label, self.labels_list))
                    self.labels_list.append(label)
                    self.label_to_num[label] = len(self.labels_list)
                self.annotations_list.append(
                    self.generate_rectangle_anns_field(anno[
                        'bbox'], label, image_id, object_id,
                                                       self.label_to_num))

    def convert(self, image_dir, json_dir, dataset_save_dir):
        """转换。
        Args:
            image_dir (str): 图像文件存放的路径。
            json_dir (str): 与每张图像对应的json文件的存放路径。
            dataset_save_dir (str): 转换后数据集存放路径。
        """
        assert osp.exists(image_dir), "he image folder does not exist!"
        assert osp.exists(json_dir), "The json folder does not exist!"
        assert osp.exists(dataset_save_dir), "The save folder does not exist!"
        # Convert the image files.
        new_image_dir = osp.join(dataset_save_dir, "JPEGImages")
        if osp.exists(new_image_dir):
            shutil.rmtree(new_image_dir)
        os.makedirs(new_image_dir)
        for img_name in os.listdir(image_dir):
            if is_pic(img_name):
                shutil.copyfile(
                    osp.join(image_dir, img_name),
                    osp.join(new_image_dir, img_name))
        # Convert the json files.
        xml_dir_dir = os.path.abspath(
            os.path.join(os.path.dirname(json_dir), os.path.pardir))
        for part in ['train', 'val', 'test']:
            part_list_file = osp.join(xml_dir_dir, '{}_list.txt'.format(part))
            if osp.exists(part_list_file):
                file_list = list()
                with open(part_list_file, 'r') as f:
                    while True:
                        line = f.readline()
                        if not line:
                            break
                        if len(line.strip().split()) > 2:
                            raise Exception(
                                "A space is defined as the separator, but it exists in image or label name {}."
                                .format(line))
                        img_file = osp.join(
                            image_dir, osp.split(line.strip().split()[0])[-1])
                        xml_file = osp.join(
                            json_dir, osp.split(line.strip().split()[1])[-1])
                        img_file = path_normalization(img_file)
                        xml_file = path_normalization(xml_file)
                        if not is_pic(img_file):
                            continue
                        if not osp.isfile(xml_file):
                            continue
                        if not osp.exists(img_file):
                            raise IOError('The image file {} is not exist!'.
                                          format(img_file))
                        file_list.append(osp.split(img_file)[-1])
                self.parse_json(new_image_dir, json_dir, file_list)
                coco_data = {}
                coco_data["images"] = self.images_list
                coco_data["categories"] = self.categories_list
                coco_data["annotations"] = self.annotations_list
                json_path = osp.join(dataset_save_dir, "{}.json".format(part))
                json.dump(
                    coco_data, open(json_path, "w"), indent=4, cls=MyEncoder)
                logging.info("xml files in {} are converted to the MSCOCO format stored in {}".format(\
                    osp.join(xml_dir_dir, '{}_list.txt'.format(part)), osp.join(dataset_save_dir, "{}.json".format(part))))
                self.images_list = []
                self.annotations_list = []
        self.parse_json(new_image_dir, json_dir)
        coco_data = {}
        coco_data["images"] = self.images_list
        coco_data["categories"] = self.categories_list
        coco_data["annotations"] = self.annotations_list
        json_path = osp.join(dataset_save_dir, "annotations.json")
        json.dump(coco_data, open(json_path, "w"), indent=4, cls=MyEncoder)
