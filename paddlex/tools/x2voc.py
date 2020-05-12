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
from .base import MyEncoder, is_pic, get_encoding

class X2VOC(object):
    def __init__(self):
        pass
    
    def convert(self, image_dir, json_dir, dataset_save_dir):
        """转换。
        Args:
            image_dir (str): 图像文件存放的路径。
            json_dir (str): 与每张图像对应的json文件的存放路径。
            dataset_save_dir (str): 转换后数据集存放路径。
        """
        assert osp.exists(image_dir), "The image folder does not exist!"
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
        xml_dir = osp.join(dataset_save_dir, "Annotations")
        if osp.exists(xml_dir):
            shutil.rmtree(xml_dir)
        os.makedirs(xml_dir)
        self.json2xml(new_image_dir, json_dir, xml_dir)
        
        
class LabelMe2VOC(X2VOC):
    """将使用LabelMe标注的数据集转换为VOC数据集。
    """
    def __init__(self):
        pass
    
    def json2xml(self, image_dir, json_dir, xml_dir):
        import xml.dom.minidom as minidom
        for img_name in os.listdir(image_dir):
            img_name_part = osp.splitext(img_name)[0]
            json_file = osp.join(json_dir, img_name_part + ".json")
            if not osp.exists(json_file):
                os.remove(os.remove(osp.join(image_dir, img_name)))
                continue
            xml_doc = minidom.Document() 
            root = xml_doc.createElement("annotation") 
            xml_doc.appendChild(root)
            node_folder = xml_doc.createElement("folder")
            node_folder.appendChild(xml_doc.createTextNode("JPEGImages"))
            root.appendChild(node_folder)
            node_filename = xml_doc.createElement("filename")
            node_filename.appendChild(xml_doc.createTextNode(img_name))
            root.appendChild(node_filename)
            with open(json_file, mode="r", \
                              encoding=get_encoding(json_file)) as j:
                json_info = json.load(j)
                h = json_info["imageHeight"]
                w = json_info["imageWidth"]
                node_size = xml_doc.createElement("size")
                node_width = xml_doc.createElement("width")
                node_width.appendChild(xml_doc.createTextNode(str(w)))
                node_size.appendChild(node_width)
                node_height = xml_doc.createElement("height")
                node_height.appendChild(xml_doc.createTextNode(str(h)))
                node_size.appendChild(node_height)
                node_depth = xml_doc.createElement("depth")
                node_depth.appendChild(xml_doc.createTextNode(str(3)))
                node_size.appendChild(node_depth)
                root.appendChild(node_size)
                for shape in json_info["shapes"]:
                    if shape["shape_type"] != "rectangle":
                        continue
                    label = shape["label"]
                    (xmin, ymin), (xmax, ymax) = shape["points"]
                    xmin, xmax = sorted([xmin, xmax])
                    ymin, ymax = sorted([ymin, ymax])
                    node_obj = xml_doc.createElement("object")
                    node_name = xml_doc.createElement("name")
                    node_name.appendChild(xml_doc.createTextNode(label))
                    node_obj.appendChild(node_name)
                    node_diff = xml_doc.createElement("difficult")
                    node_diff.appendChild(xml_doc.createTextNode(str(0)))
                    node_obj.appendChild(node_diff)
                    node_box = xml_doc.createElement("bndbox")
                    node_xmin = xml_doc.createElement("xmin")
                    node_xmin.appendChild(xml_doc.createTextNode(str(xmin)))
                    node_box.appendChild(node_xmin)
                    node_ymin = xml_doc.createElement("ymin")
                    node_ymin.appendChild(xml_doc.createTextNode(str(ymin)))
                    node_box.appendChild(node_ymin)
                    node_xmax = xml_doc.createElement("xmax")
                    node_xmax.appendChild(xml_doc.createTextNode(str(xmax)))
                    node_box.appendChild(node_xmax)
                    node_ymax = xml_doc.createElement("ymax")
                    node_ymax.appendChild(xml_doc.createTextNode(str(ymax)))
                    node_box.appendChild(node_ymax)
                    node_obj.appendChild(node_box)
                    root.appendChild(node_obj)
            with open(osp.join(xml_dir, img_name_part + ".xml"), 'w') as fxml:
                xml_doc.writexml(fxml, indent='\t', addindent='\t', newl='\n', encoding="utf-8")
                    
                    
class EasyData2VOC(X2VOC):
    """将使用EasyData标注的分割数据集转换为VOC数据集。
    """
    def __init__(self):
        pass
    
    def json2xml(self, image_dir, json_dir, xml_dir):
        import xml.dom.minidom as minidom
        for img_name in os.listdir(image_dir):
            img_name_part = osp.splitext(img_name)[0]
            json_file = osp.join(json_dir, img_name_part + ".json")
            if not osp.exists(json_file):
                os.remove(os.remove(osp.join(image_dir, img_name)))
                continue
            xml_doc = minidom.Document() 
            root = xml_doc.createElement("annotation") 
            xml_doc.appendChild(root)
            node_folder = xml_doc.createElement("folder")
            node_folder.appendChild(xml_doc.createTextNode("JPEGImages"))
            root.appendChild(node_folder)
            node_filename = xml_doc.createElement("filename")
            node_filename.appendChild(xml_doc.createTextNode(img_name))
            root.appendChild(node_filename)
            img = cv2.imread(osp.join(image_dir, img_name))
            h = img.shape[0]
            w = img.shape[1]
            node_size = xml_doc.createElement("size")
            node_width = xml_doc.createElement("width")
            node_width.appendChild(xml_doc.createTextNode(str(w)))
            node_size.appendChild(node_width)
            node_height = xml_doc.createElement("height")
            node_height.appendChild(xml_doc.createTextNode(str(h)))
            node_size.appendChild(node_height)
            node_depth = xml_doc.createElement("depth")
            node_depth.appendChild(xml_doc.createTextNode(str(3)))
            node_size.appendChild(node_depth)
            root.appendChild(node_size)
            with open(json_file, mode="r", \
                              encoding=get_encoding(json_file)) as j:
                json_info = json.load(j)
                for shape in json_info["labels"]:
                    label = shape["name"]
                    xmin = shape["x1"]
                    ymin = shape["y1"]
                    xmax = shape["x2"]
                    ymax = shape["y2"]
                    node_obj = xml_doc.createElement("object")
                    node_name = xml_doc.createElement("name")
                    node_name.appendChild(xml_doc.createTextNode(label))
                    node_obj.appendChild(node_name)
                    node_diff = xml_doc.createElement("difficult")
                    node_diff.appendChild(xml_doc.createTextNode(str(0)))
                    node_obj.appendChild(node_diff)
                    node_box = xml_doc.createElement("bndbox")
                    node_xmin = xml_doc.createElement("xmin")
                    node_xmin.appendChild(xml_doc.createTextNode(str(xmin)))
                    node_box.appendChild(node_xmin)
                    node_ymin = xml_doc.createElement("ymin")
                    node_ymin.appendChild(xml_doc.createTextNode(str(ymin)))
                    node_box.appendChild(node_ymin)
                    node_xmax = xml_doc.createElement("xmax")
                    node_xmax.appendChild(xml_doc.createTextNode(str(xmax)))
                    node_box.appendChild(node_xmax)
                    node_ymax = xml_doc.createElement("ymax")
                    node_ymax.appendChild(xml_doc.createTextNode(str(ymax)))
                    node_box.appendChild(node_ymax)
                    node_obj.appendChild(node_box)
                    root.appendChild(node_obj)
            with open(osp.join(xml_dir, img_name_part + ".xml"), 'w') as fxml:
                xml_doc.writexml(fxml, indent='\t', addindent='\t', newl='\n', encoding="utf-8")                    
                    
