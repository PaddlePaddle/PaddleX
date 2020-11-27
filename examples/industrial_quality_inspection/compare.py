# coding: utf8
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

# 环境变量配置，用于控制是否使用GPU
# 说明文档：https://paddlex.readthedocs.io/zh_CN/develop/appendix/parameters.html#gpu
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import os.path as osp
import cv2
import re
import xml.etree.ElementTree as ET
import paddlex as pdx

data_dir = 'aluminum_inspection/'
file_list = 'aluminum_inspection/val_list.txt'
model_dir = 'output/faster_rcnn_r50_vd_dcn/best_model/'
save_dir = './visualize/compare'
# 设置置信度阈值
score_threshold = 0.1

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

model = pdx.load_model(model_dir)

with open(file_list, 'r') as fr:
    while True:
        line = fr.readline()
        if not line:
            break
        img_file, xml_file = [osp.join(data_dir, x) \
                for x in line.strip().split()[:2]]
        if not osp.exists(img_file):
            continue
        if not osp.exists(xml_file):
            continue

        res = model.predict(img_file)
        det_vis = pdx.det.visualize(
            img_file, res, threshold=score_threshold, save_dir=None)

        tree = ET.parse(xml_file)
        pattern = re.compile('<object>', re.IGNORECASE)
        obj_match = pattern.findall(str(ET.tostringlist(tree.getroot())))
        if len(obj_match) == 0:
            continue
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
        gt_bbox = []
        gt_class = []
        for i, obj in enumerate(objs):
            pattern = re.compile('<name>', re.IGNORECASE)
            name_tag = pattern.findall(str(ET.tostringlist(obj)))[0][1:-1]
            cname = obj.find(name_tag).text.strip()
            gt_class.append(cname)
            pattern = re.compile('<difficult>', re.IGNORECASE)
            diff_tag = pattern.findall(str(ET.tostringlist(obj)))[0][1:-1]
            try:
                _difficult = int(obj.find(diff_tag).text)
            except Exception:
                _difficult = 0
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
            gt_bbox.append([x1, y1, x2, y2])
        gts = []
        for bbox, name in zip(gt_bbox, gt_class):
            x1, y1, x2, y2 = bbox
            w = x2 - x1 + 1
            h = y2 - y1 + 1
            gt = {
                'category_id': 0,
                'category': name,
                'bbox': [x1, y1, w, h],
                'score': 1
            }
            gts.append(gt)
        gt_vis = pdx.det.visualize(
            img_file, gts, threshold=score_threshold, save_dir=None)
        vis = cv2.hconcat([gt_vis, det_vis])
        cv2.imwrite(os.path.join(save_dir, os.path.split(img_file)[-1]), vis)
        print('The comparison has been made for {}'.format(img_file))

print(
    "The visualized ground-truths and predictions are saved in {}. Ground-truth is on the left, prediciton is on the right".
    format(save_dir))
