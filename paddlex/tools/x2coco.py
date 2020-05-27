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
import numpy as np
import PIL.ImageDraw
from .base import MyEncoder, is_pic, get_encoding
        
        
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
    
    def generate_rectangle_anns_field(self, points, label, image_id, object_id, label_to_num):
        annotation = {}
        seg_points = np.asarray(points).copy()
        seg_points[1, :] = np.asarray(points)[2, :]
        seg_points[2, :] = np.asarray(points)[1, :]
        annotation["segmentation"] = [list(seg_points.flatten())]
        annotation["iscrowd"] = 0
        annotation["image_id"] = image_id + 1
        annotation["bbox"] = list(
            map(float, [
                points[0][0], points[0][1], points[1][0] - points[0][0], points[1][
                    1] - points[0][1]
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
        json.dump(
            coco_data,
            open(json_path, "w"),
            indent=4,
            cls=MyEncoder)
    
    
class LabelMe2COCO(X2COCO):
    """将使用LabelMe标注的数据集转换为COCO数据集。
    """
    def __init__(self):
        super(LabelMe2COCO, self).__init__()
        
    def generate_images_field(self, json_info, image_id):
        image = {}
        image["height"] = json_info["imageHeight"]
        image["width"] = json_info["imageWidth"]
        image["id"] = image_id + 1
        image["file_name"] = json_info["imagePath"].split("/")[-1]
        return image
    
    def generate_polygon_anns_field(self, height, width, 
                                    points, label, image_id, 
                                    object_id, label_to_num):
        annotation = {}
        annotation["segmentation"] = [list(np.asarray(points).flatten())]
        annotation["iscrowd"] = 0
        annotation["image_id"] = image_id + 1
        annotation["bbox"] = list(map(float, self.get_bbox(height, width, points)))
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
                os.remove(os.remove(osp.join(image_dir, img_file)))
                continue
            image_id = image_id + 1
            with open(json_file, mode='r', \
                              encoding=get_encoding(json_file)) as j:
                json_info = json.load(j)
                img_info = self.generate_images_field(json_info, image_id)
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
                            self.generate_polygon_anns_field(json_info["imageHeight"], json_info[
                                "imageWidth"], points, label, image_id,
                                                object_id, label_to_num))
                    if p_type == "rectangle":
                        points.append([points[0][0], points[1][1]])
                        points.append([points[1][0], points[0][1]])
                        self.annotations_list.append(
                            self.generate_rectangle_anns_field(points, label, image_id,
                                                  object_id, label_to_num))
                        
    
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
        image["file_name"] = osp.split(img_path)[-1]
        return image
    
    def generate_polygon_anns_field(self, points, segmentation, 
                                    label, image_id, object_id,
                                    label_to_num):
        annotation = {}
        annotation["segmentation"] = segmentation
        annotation["iscrowd"] = 1 if len(segmentation) > 1 else 0
        annotation["image_id"] = image_id + 1
        annotation["bbox"] = list(map(float, [
                points[0][0], points[0][1], points[1][0] - points[0][0], points[1][
                    1] - points[0][1]
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
                os.remove(os.remove(osp.join(image_dir, img_file)))
                continue
            image_id = image_id + 1
            with open(json_file, mode='r', \
                              encoding=get_encoding(json_file)) as j:
                json_info = json.load(j)
                img_info = self.generate_images_field(osp.join(img_dir, img_file), image_id)
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
                            self.generate_rectangle_anns_field(points, label, image_id,
                                                  object_id, label_to_num))
                    else:
                        mask_dict = {}
                        mask_dict['size'] = [img_info["height"], img_info["width"]]
                        mask_dict['counts'] = shapes['mask'].encode()
                        mask = decode(mask_dict)
                        contours, hierarchy = cv2.findContours(
                                (mask).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        segmentation = []
                        for contour in contours:
                            contour_list = contour.flatten().tolist()
                            if len(contour_list) > 4:
                                segmentation.append(contour_list)
                        self.annotations_list.append(
                            self.generate_polygon_anns_field(points, segmentation, label, image_id, object_id,
                                                label_to_num))
