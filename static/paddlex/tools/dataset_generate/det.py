# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
import os
import os.path as osp
import random
import cv2
import time
import numpy as np
import xml.etree.ElementTree as ET
import paddlex.utils.logging as logging


def write_xml(im_info, label_info, anno_dir):
    im_fname = im_info['file_name']
    im_h, im_w, im_c = im_info['image_shape']
    is_crowd = label_info['is_crowd']
    gt_class = label_info['gt_class']
    gt_bbox = label_info['gt_bbox']
    gt_score = label_info['gt_score']
    gt_poly = label_info['gt_poly']
    difficult = label_info['difficult']
    import xml.dom.minidom as minidom
    xml_doc = minidom.Document()
    root = xml_doc.createElement("annotation")
    xml_doc.appendChild(root)
    node_filename = xml_doc.createElement("filename")
    node_filename.appendChild(xml_doc.createTextNode(im_fname))
    root.appendChild(node_filename)
    node_size = xml_doc.createElement("size")
    node_width = xml_doc.createElement("width")
    node_width.appendChild(xml_doc.createTextNode(str(im_w)))
    node_size.appendChild(node_width)
    node_height = xml_doc.createElement("height")
    node_height.appendChild(xml_doc.createTextNode(str(im_h)))
    node_size.appendChild(node_height)
    node_depth = xml_doc.createElement("depth")
    node_depth.appendChild(xml_doc.createTextNode(str(im_c)))
    node_size.appendChild(node_depth)
    root.appendChild(node_size)
    for i in range(len(label_info['gt_class'])):
        node_obj = xml_doc.createElement("object")
        node_name = xml_doc.createElement("name")
        label = gt_class[i]
        node_name.appendChild(xml_doc.createTextNode(label))
        node_obj.appendChild(node_name)
        node_diff = xml_doc.createElement("difficult")
        node_diff.appendChild(xml_doc.createTextNode(str(difficult[i][0])))
        node_obj.appendChild(node_diff)
        node_box = xml_doc.createElement("bndbox")
        node_xmin = xml_doc.createElement("xmin")
        node_xmin.appendChild(xml_doc.createTextNode(str(gt_bbox[i][0])))
        node_box.appendChild(node_xmin)
        node_ymin = xml_doc.createElement("ymin")
        node_ymin.appendChild(xml_doc.createTextNode(str(gt_bbox[i][1])))
        node_box.appendChild(node_ymin)
        node_xmax = xml_doc.createElement("xmax")
        node_xmax.appendChild(xml_doc.createTextNode(str(gt_bbox[i][2])))
        node_box.appendChild(node_xmax)
        node_ymax = xml_doc.createElement("ymax")
        node_ymax.appendChild(xml_doc.createTextNode(str(gt_bbox[i][3])))
        node_box.appendChild(node_ymax)
        node_obj.appendChild(node_box)
        root.appendChild(node_obj)
    img_name_part = im_fname.split('.')[0]
    with open(osp.join(anno_dir, img_name_part + ".xml"), 'w') as fxml:
        xml_doc.writexml(
            fxml, indent='\t', addindent='\t', newl='\n', encoding="utf-8")


def paste_objects(templates, background, save_dir='dataset_clone'):
    """将目标物体粘贴在背景图片上生成新的图片，并加入到数据集中

    Args:
        templates (list|tuple)：可以将多张图像上的目标物体同时粘贴在同一个背景图片上，
            因此templates是一个列表，其中每个元素是一个dict，表示一张图片的目标物体。
            一张图片的目标物体有`image`和`annos`两个关键字，`image`的键值是图像的路径，
            或者是解码后的排列格式为（H, W, C）且类型为uint8且为BGR格式的数组。
            图像上可以有多个目标物体，因此`annos`的键值是一个列表，列表中每个元素是一个dict，
            表示一个目标物体的信息。该dict包含`polygon`和`category`两个关键字，
            其中`polygon`表示目标物体的边缘坐标，例如[[0, 0], [0, 1], [1, 1], [1, 0]]，
            `category`表示目标物体的类别，例如'dog'。
        background (dict): 背景图片可以有真值，因此background是一个dict，包含`image`和`annos`
            两个关键字，`image`的键值是背景图像的路径，或者是解码后的排列格式为（H, W, C）
            且类型为uint8且为BGR格式的数组。若背景图片上没有真值，则`annos`的键值是空列表[]，
            若有，则`annos`的键值是由多个dict组成的列表，每个dict表示一个物体的信息，
            包含`bbox`和`category`两个关键字，`bbox`的键值是物体框左上角和右下角的坐标，即
            [x1, y1, x2, y2]，`category`表示目标物体的类别，例如'dog'。
        save_dir (str)：新图片及其标注文件的存储目录。默认值为`dataset_clone`。

    """
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    image_dir = osp.join(save_dir, 'JPEGImages_clone')
    anno_dir = osp.join(save_dir, 'Annotations_clone')
    json_path = osp.join(save_dir, "annotations.json")
    if not osp.exists(image_dir):
        os.makedirs(image_dir)
    if not osp.exists(anno_dir):
        os.makedirs(anno_dir)

    num_objs = len(background['annos'])
    for temp in templates:
        num_objs += len(temp['annos'])

    gt_bbox = np.zeros((num_objs, 4), dtype=np.float32)
    gt_class = list()
    gt_score = np.ones((num_objs, 1), dtype=np.float32)
    is_crowd = np.zeros((num_objs, 1), dtype=np.int32)
    difficult = np.zeros((num_objs, 1), dtype=np.int32)
    i = -1
    for i, back_anno in enumerate(background['annos']):
        gt_bbox[i] = back_anno['bbox']
        gt_class.append(back_anno['category'])

    back_im = background['image']
    if isinstance(back_im, np.ndarray):
        if len(back_im.shape) != 3:
            raise Exception(
                "background image should be 3-dimensions, but now is {}-dimensions".
                format(len(back_im.shape)))
    else:
        try:
            back_im = cv2.imread(back_im, cv2.IMREAD_UNCHANGED)
        except:
            raise TypeError('Can\'t read The image file {}!'.format(back_im))
    back_annos = background['annos']
    im_h, im_w, im_c = back_im.shape
    for temp in templates:
        temp_im = temp['image']
        if isinstance(temp_im, np.ndarray):
            if len(temp_im.shape) != 3:
                raise Exception(
                    "template image should be 3-dimensions, but now is {}-dimensions".
                    format(len(temp_im.shape)))
        else:
            try:
                temp_im = cv2.imread(temp_im, cv2.IMREAD_UNCHANGED)
            except:
                raise TypeError('Can\'t read The image file {}!'.format(
                    temp_im))
        if im_c != temp_im.shape[-1]:
            raise Exception(
                "The channels of template({}) and background({}) images are not same. Objects cannot be pasted normally! Please check your images.".
                format(temp_im.shape[-1], im_c))
        temp_annos = temp['annos']
        for temp_anno in temp_annos:
            temp_mask = np.zeros(temp_im.shape, temp_im.dtype)
            temp_poly = np.array(temp_anno['polygon'], np.int32)
            temp_category = temp_anno['category']
            cv2.fillPoly(temp_mask, [temp_poly], (255, 255, 255))
            x_list = [temp_poly[i][0] for i in range(len(temp_poly))]
            y_list = [temp_poly[i][1] for i in range(len(temp_poly))]
            temp_poly_w = max(x_list) - min(x_list)
            temp_poly_h = max(y_list) - min(y_list)
            found = False
            while not found:
                center_x = random.randint(1, im_w - 1)
                center_y = random.randint(1, im_h - 1)
                if center_x < temp_poly_w / 2 or center_x > im_w - temp_poly_w / 2 - 1 or \
                   center_y < temp_poly_h / 2 or center_y > im_h - temp_poly_h / 2 - 1:
                    found = False
                    continue
                if len(back_annos) == 0:
                    found = True
                for back_anno in back_annos:
                    x1, y1, x2, y2 = back_anno['bbox']
                    if center_x > x1 and center_x < x2 and center_y > y1 and center_y < y2:
                        found = False
                        continue
                    found = True
            center = (center_x, center_y)
            back_im = cv2.seamlessClone(temp_im, back_im, temp_mask, center,
                                        cv2.MIXED_CLONE)
            i += 1
            x1 = center[0] - temp_poly_w / 2
            x2 = center[0] + temp_poly_w / 2
            y1 = center[1] - temp_poly_h / 2
            y2 = center[1] + temp_poly_h / 2
            gt_bbox[i] = [x1, y1, x2, y2]
            gt_class.append(temp_category)

    im_fname = str(int(time.time() * 1000)) + '.jpg'
    im_info = {
        'file_name': im_fname,
        'image_shape': [im_h, im_w, im_c],
    }
    label_info = {
        'is_crowd': is_crowd,
        'gt_class': gt_class,
        'gt_bbox': gt_bbox,
        'gt_score': gt_score,
        'difficult': difficult,
        'gt_poly': [],
    }
    cv2.imwrite(osp.join(image_dir, im_fname), back_im.astype('uint8'))
    write_xml(im_info, label_info, anno_dir)
    logging.info("Gegerated image is saved in {}".format(image_dir))
    logging.info("Generated Annotation is saved as xml files in {}".format(
        anno_dir))
