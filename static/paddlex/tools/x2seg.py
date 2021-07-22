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
import uuid
import json
import os
import os.path as osp
import shutil
import numpy as np
import PIL.Image
from .base import MyEncoder, is_pic, get_encoding
import math

class X2Seg(object):
    def __init__(self):
        self.labels2ids = {'_background_': 0}
        
    def shapes_to_label(self, img_shape, shapes, label_name_to_value):
        # 该函数基于https://github.com/wkentaro/labelme/blob/master/labelme/utils/shape.py实现。
        def shape_to_mask(img_shape, points, shape_type=None,
                  line_width=10, point_size=5):
            mask = np.zeros(img_shape[:2], dtype=np.uint8)
            mask = PIL.Image.fromarray(mask)
            draw = PIL.ImageDraw.Draw(mask)
            xy = [tuple(point) for point in points]
            if shape_type == 'circle':
                assert len(xy) == 2, 'Shape of shape_type=circle must have 2 points'
                (cx, cy), (px, py) = xy
                d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
                draw.ellipse([cx - d, cy - d, cx + d, cy + d], outline=1, fill=1)
            elif shape_type == 'rectangle':
                assert len(xy) == 2, 'Shape of shape_type=rectangle must have 2 points'
                draw.rectangle(xy, outline=1, fill=1)
            elif shape_type == 'line':
                assert len(xy) == 2, 'Shape of shape_type=line must have 2 points'
                draw.line(xy=xy, fill=1, width=line_width)
            elif shape_type == 'linestrip':
                draw.line(xy=xy, fill=1, width=line_width)
            elif shape_type == 'point':
                assert len(xy) == 1, 'Shape of shape_type=point must have 1 points'
                cx, cy = xy[0]
                r = point_size
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=1, fill=1)
            else:
                assert len(xy) > 2, 'Polygon must have points more than 2'
                draw.polygon(xy=xy, outline=1, fill=1)
            mask = np.array(mask, dtype=bool)
            return mask
        cls = np.zeros(img_shape[:2], dtype=np.int32)
        ins = np.zeros_like(cls)
        instances = []
        for shape in shapes:
            points = shape['points']
            label = shape['label']
            group_id = shape.get('group_id')
            if group_id is None:
                group_id = uuid.uuid1()
            shape_type = shape.get('shape_type', None)

            cls_name = label
            instance = (cls_name, group_id)

            if instance not in instances:
                instances.append(instance)
            ins_id = instances.index(instance) + 1
            cls_id = label_name_to_value[cls_name]
            mask = shape_to_mask(img_shape[:2], points, shape_type)
            cls[mask] = cls_id
            ins[mask] = ins_id
        return cls, ins
    
    def get_color_map_list(self, num_classes):
        color_map = num_classes * [0, 0, 0]
        for i in range(0, num_classes):
            j = 0
            lab = i
            while lab:
                color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
                color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
                color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
                j += 1
                lab >>= 3
        return color_map
    
    def convert(self, image_dir, json_dir, dataset_save_dir):
        """转换。
        Args:
            image_dir (str): 图像文件存放的路径。
            json_dir (str): 与每张图像对应的json文件的存放路径。
            dataset_save_dir (str): 转换后数据集存放路径。
        """
        assert osp.exists(image_dir), "The image folder does not exist!"
        assert osp.exists(json_dir), "The json folder does not exist!"
        if not osp.exists(dataset_save_dir):
            os.makedirs(dataset_save_dir)
        # Convert the image files.
        new_image_dir = osp.join(dataset_save_dir, "JPEGImages")
        if osp.exists(new_image_dir):
            raise Exception("The directory {} is already exist, please remove the directory first".format(new_image_dir))
        os.makedirs(new_image_dir)
        for img_name in os.listdir(image_dir):
            if is_pic(img_name):
                shutil.copyfile(
                            osp.join(image_dir, img_name),
                            osp.join(new_image_dir, img_name))
        # Convert the json files.
        png_dir = osp.join(dataset_save_dir, "Annotations")
        if osp.exists(png_dir):
            shutil.rmtree(png_dir)
        os.makedirs(png_dir)
        self.get_labels2ids(new_image_dir, json_dir)
        self.json2png(new_image_dir, json_dir, png_dir)
        # Generate the labels.txt
        ids2labels = {v : k for k, v in self.labels2ids.items()}
        with open(osp.join(dataset_save_dir, 'labels.txt'), 'w') as fw:
            for i in range(len(ids2labels)):
                fw.write(ids2labels[i] + '\n')
        

class JingLing2Seg(X2Seg):
    """将使用标注精灵标注的数据集转换为Seg数据集。
    """
    def __init__(self):
        super(JingLing2Seg, self).__init__() 
        
    def get_labels2ids(self, image_dir, json_dir):
        for img_name in os.listdir(image_dir):
            img_name_part = osp.splitext(img_name)[0]
            json_file = osp.join(json_dir, img_name_part + ".json")
            if not osp.exists(json_file):
                os.remove(osp.join(image_dir, img_name))
                continue
            with open(json_file, mode="r", \
                              encoding=get_encoding(json_file)) as j:
                json_info = json.load(j)
                if 'outputs' in json_info:
                    for output in json_info['outputs']['object']:
                        cls_name = output['name']
                        if cls_name not in self.labels2ids:
                            self.labels2ids[cls_name] =  len(self.labels2ids)
    
    def json2png(self, image_dir, json_dir, png_dir):
        color_map = self.get_color_map_list(256)
        for img_name in os.listdir(image_dir):
            img_name_part = osp.splitext(img_name)[0]
            json_file = osp.join(json_dir, img_name_part + ".json")
            if not osp.exists(json_file):
                os.remove(osp.join(image_dir, img_name))
                continue
            with open(json_file, mode="r", \
                              encoding=get_encoding(json_file)) as j:
                json_info = json.load(j)
                data_shapes = []
                if 'outputs' in json_info:
                    for output in json_info['outputs']['object']:
                        if 'polygon' in output.keys():
                            polygon = output['polygon']
                            name = output['name']
                            points = []
                            for i in range(1, int(len(polygon) / 2) + 1):
                                points.append(
                                    [polygon['x' + str(i)], polygon['y' + str(i)]])
                            shape = {
                                'label': name,
                                'points': points,
                                'shape_type': 'polygon'
                            }
                            data_shapes.append(shape)
                if 'size' not in json_info:
                    continue
            img_shape = (json_info['size']['height'], 
                         json_info['size']['width'],
                         json_info['size']['depth'])
            lbl, _ = self.shapes_to_label(
                img_shape=img_shape,
                shapes=data_shapes,
                label_name_to_value=self.labels2ids,
            )
            out_png_file = osp.join(png_dir, img_name_part + '.png')
            if lbl.min() >= 0 and lbl.max() <= 255:
                lbl_pil = PIL.Image.fromarray(lbl.astype(np.uint8), mode='P')
                lbl_pil.putpalette(color_map)
                lbl_pil.save(out_png_file)
            else:
                raise ValueError(
                    '[%s] Cannot save the pixel-wise class label as PNG. '
                    'Please consider using the .npy format.' % out_png_file)
                
                
class LabelMe2Seg(X2Seg):
    """将使用LabelMe标注的数据集转换为Seg数据集。
    """
    def __init__(self):
        super(LabelMe2Seg, self).__init__()
    
    def get_labels2ids(self, image_dir, json_dir):
        for img_name in os.listdir(image_dir):
            img_name_part = osp.splitext(img_name)[0]
            json_file = osp.join(json_dir, img_name_part + ".json")
            if not osp.exists(json_file):
                os.remove(osp.join(image_dir, img_name))
                continue
            with open(json_file, mode="r", \
                              encoding=get_encoding(json_file)) as j:
                json_info = json.load(j)
                for shape in json_info['shapes']:
                    cls_name = shape['label']
                    if cls_name not in self.labels2ids:
                        self.labels2ids[cls_name] =  len(self.labels2ids)
                     
    def json2png(self, image_dir, json_dir, png_dir):
        color_map = self.get_color_map_list(256)
        for img_name in os.listdir(image_dir):
            img_name_part = osp.splitext(img_name)[0]
            json_file = osp.join(json_dir, img_name_part + ".json")
            if not osp.exists(json_file):
                os.remove(osp.join(image_dir, img_name))
                continue
            img_file = osp.join(image_dir, img_name)
            img = np.asarray(PIL.Image.open(img_file))
            with open(json_file, mode="r", \
                              encoding=get_encoding(json_file)) as j:
                json_info = json.load(j)
            lbl, _ = self.shapes_to_label(
                img_shape=img.shape,
                shapes=json_info['shapes'],
                label_name_to_value=self.labels2ids,
            )
            out_png_file = osp.join(png_dir, img_name_part + '.png')
            if lbl.min() >= 0 and lbl.max() <= 255:
                lbl_pil = PIL.Image.fromarray(lbl.astype(np.uint8), mode='P')
                lbl_pil.putpalette(color_map)
                lbl_pil.save(out_png_file)
            else:
                raise ValueError(
                    '[%s] Cannot save the pixel-wise class label as PNG. '
                    'Please consider using the .npy format.' % out_png_file)
                
                            
class EasyData2Seg(X2Seg):
    """将使用EasyData标注的分割数据集转换为Seg数据集。
    """
    def __init__(self):
        super(EasyData2Seg, self).__init__()
    
    def get_labels2ids(self, image_dir, json_dir):
        for img_name in os.listdir(image_dir):
            img_name_part = osp.splitext(img_name)[0]
            json_file = osp.join(json_dir, img_name_part + ".json")
            if not osp.exists(json_file):
                os.remove(osp.join(image_dir, img_name))
                continue
            with open(json_file, mode="r", \
                              encoding=get_encoding(json_file)) as j:
                json_info = json.load(j)
                for shape in json_info["labels"]:
                    cls_name = shape['name']
                    if cls_name not in self.labels2ids:
                        self.labels2ids[cls_name] =  len(self.labels2ids)
                        
    def mask2polygon(self, mask, label):
        contours, hierarchy = cv2.findContours(
            (mask).astype(np.uint8), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        segmentation = []
        for contour in contours:
            contour_list = contour.flatten().tolist()
            if len(contour_list) > 4:
                points = []
                for i in range(0, len(contour_list), 2):
                    points.append(
                                [contour_list[i], contour_list[i + 1]])
                shape = {
                    'label': label,
                    'points': points,
                    'shape_type': 'polygon'
                }
                segmentation.append(shape)
        return segmentation
    
    def json2png(self, image_dir, json_dir, png_dir):
        from pycocotools.mask import decode
        color_map = self.get_color_map_list(256)
        for img_name in os.listdir(image_dir):
            img_name_part = osp.splitext(img_name)[0]
            json_file = osp.join(json_dir, img_name_part + ".json")
            if not osp.exists(json_file):
                os.remove(osp.join(image_dir, img_name))
                continue
            img_file = osp.join(image_dir, img_name)
            img = np.asarray(PIL.Image.open(img_file))
            img_h = img.shape[0]
            img_w = img.shape[1]
            with open(json_file, mode="r", \
                              encoding=get_encoding(json_file)) as j:
                json_info = json.load(j)
                data_shapes = []
                for shape in json_info['labels']:
                    mask_dict = {}
                    mask_dict['size'] = [img_h, img_w]
                    mask_dict['counts'] = shape['mask'].encode()
                    mask = decode(mask_dict)
                    polygon = self.mask2polygon(mask, shape["name"])
                    data_shapes.extend(polygon)
            lbl, _ = self.shapes_to_label(
                img_shape=img.shape,
                shapes=data_shapes,
                label_name_to_value=self.labels2ids,
            )
            out_png_file = osp.join(png_dir, img_name_part + '.png')
            if lbl.min() >= 0 and lbl.max() <= 255:
                lbl_pil = PIL.Image.fromarray(lbl.astype(np.uint8), mode='P')
                lbl_pil.putpalette(color_map)
                lbl_pil.save(out_png_file)
            else:
                raise ValueError(
                    '[%s] Cannot save the pixel-wise class label as PNG. '
                    'Please consider using the .npy format.' % out_png_file)
            


