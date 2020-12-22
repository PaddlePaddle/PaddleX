# copytrue (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import math
import xml.etree.ElementTree as ET
from PIL import Image


def resize_img(img):
    """ 调整图片尺寸

    Args:
        img: 图片信息
    """
    h, w = img.shape[:2]
    min_size = 580

    if w >= h and w > min_size:
        new_w = min_size
        new_h = new_w * h / w
    elif h >= w and h > min_size:
        new_h = min_size
        new_w = new_h * w / h
    else:
        new_h = h
        new_w = w
    new_img = cv2.resize(
        img, (int(new_w), int(new_h)), interpolation=cv2.INTER_CUBIC)

    scale_value = new_w / w
    return new_img, scale_value


def plot_det_label(image, anno, labels):
    """ 目标检测类型生成标注图

    Args:
        image: 图片路径
        anno: 图片标注
        labels: 图片所属数据集的类别信息
    """
    catid2color = {}
    img = cv2.imread(image)
    img, scale_value = resize_img(img)
    tree = ET.parse(anno)
    objs = tree.findall('object')
    color_map = get_color_map_list(len(labels) + 1)
    for i, obj in enumerate(objs):
        cname = obj.find('name').text
        catid = labels.index(cname)
        if cname not in labels:
            continue
        xmin = int(float(obj.find('bndbox').find('xmin').text) * scale_value)
        ymin = int(float(obj.find('bndbox').find('ymin').text) * scale_value)
        xmax = int(float(obj.find('bndbox').find('xmax').text) * scale_value)
        ymax = int(float(obj.find('bndbox').find('ymax').text) * scale_value)

        if catid not in catid2color:
            catid2color[catid] = color_map[catid + 1]
        color = tuple(catid2color[catid])
        img = draw_rectangle_and_cname(img, xmin, ymin, xmax, ymax, cname,
                                       color)
    return img


def plot_seg_label(anno):
    """ 语义分割类型生成标注图

    Args:
        anno: 图片标注
    """
    label = pil_imread(anno)
    pse_label = gray2pseudo(label)
    return pse_label


def plot_insseg_label(image, anno, labels, alpha=0.7):
    """ 实例分割类型生成标注图

    Args:
        image: 图片路径
        anno: 图片标注
        labels: 图片所属数据集的类别信息
    """
    anno = np.load(anno, allow_pickle=True).tolist()
    catid2color = dict()
    img = cv2.imread(image)
    img, scale_value = resize_img(img)
    color_map = get_color_map_list(len(labels) + 1)
    img_h = anno['h']
    img_w = anno['w']
    gt_class = anno['gt_class']
    gt_bbox = anno['gt_bbox']
    gt_poly = anno['gt_poly']
    num_bbox = gt_bbox.shape[0]
    num_mask = len(gt_poly)
    # 描绘mask信息
    img_array = np.array(img).astype('float32')
    for i in range(num_mask):
        cname = gt_class[i]
        catid = labels.index(cname)
        if cname not in labels:
            continue
        if catid not in catid2color:
            catid2color[catid] = color_map[catid + 1]
        color = np.array(catid2color[catid]).astype('float32')

        import pycocotools.mask as mask_util
        for x in range(len(gt_poly[i])):
            for y in range(len(gt_poly[i][x])):
                gt_poly[i][x][y] = int(float(gt_poly[i][x][y]) * scale_value)
        poly = gt_poly[i]
        rles = mask_util.frPyObjects(poly,
                                     int(float(img_h) * scale_value),
                                     int(float(img_w) * scale_value))
        rle = mask_util.merge(rles)
        mask = mask_util.decode(rle) * 255
        idx = np.nonzero(mask)
        img_array[idx[0], idx[1], :] *= 1.0 - alpha
        img_array[idx[0], idx[1], :] += alpha * color
    img = img_array.astype('uint8')

    for i in range(num_bbox):
        cname = gt_class[i]
        catid = labels.index(cname)
        if cname not in labels:
            continue
        if catid not in catid2color:
            catid2color[catid] = color_map[catid]
        color = tuple(catid2color[catid])
        xmin, ymin, xmax, ymax = gt_bbox[i]

        img = draw_rectangle_and_cname(img,
                                       int(float(xmin) * scale_value),
                                       int(float(ymin) * scale_value),
                                       int(float(xmax) * scale_value),
                                       int(float(ymax) * scale_value), cname,
                                       color)

    return img


def draw_rectangle_and_cname(img, xmin, ymin, xmax, ymax, cname, color):
    """ 根据提供的标注信息，给图片描绘框体和类别显示

    Args:
        img: 图片路径
        xmin: 检测框最小的x坐标
        ymin: 检测框最小的y坐标
        xmax: 检测框最大的x坐标
        ymax: 检测框最大的y坐标
        cname: 类别信息
        color: 类别与颜色的对应信息
    """
    # 描绘检测框
    line_width = math.ceil(2 * max(img.shape[0:2]) / 600)
    cv2.rectangle(
        img,
        pt1=(xmin, ymin),
        pt2=(xmax, ymax),
        color=color,
        thickness=line_width)

    # 计算并描绘类别信息
    text_thickness = math.ceil(2 * max(img.shape[0:2]) / 1200)
    fontscale = math.ceil(0.5 * max(img.shape[0:2]) / 600)
    tw, th = cv2.getTextSize(
        cname, 0, fontScale=fontscale, thickness=text_thickness)[0]
    cv2.rectangle(
        img,
        pt1=(xmin + 1, ymin - th),
        pt2=(xmin + int(0.7 * tw) + 1, ymin),
        color=color,
        thickness=-1)
    cv2.putText(
        img,
        cname, (int(xmin) + 3, int(ymin) - 5),
        0,
        0.6 * fontscale, (255, 255, 255),
        lineType=cv2.LINE_AA,
        thickness=text_thickness)
    return img


def pil_imread(file_path):
    """ 将图片读成np格式数据

    Args:
        file_path: 图片路径
    """
    img = Image.open(file_path)
    return np.asarray(img)


def get_color_map_list(num_classes):
    """ 为类别信息生成对应的颜色列表

    Args:
        num_classes: 类别数量
    """
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
    color_map = [color_map[i:i + 3] for i in range(0, len(color_map), 3)]
    return color_map


def gray2pseudo(gray_image):
    """ 将分割的结果映射到图片

    Args:
        gray_image: 灰度图
    """
    color_map = get_color_map_list(256)
    color_map = np.array(color_map).astype("uint8")
    # 用OpenCV进行色彩映射
    c1 = cv2.LUT(gray_image, color_map[:, 0])
    c2 = cv2.LUT(gray_image, color_map[:, 1])
    c3 = cv2.LUT(gray_image, color_map[:, 2])
    pseudo_img = np.dstack((c1, c2, c3))
    return pseudo_img
