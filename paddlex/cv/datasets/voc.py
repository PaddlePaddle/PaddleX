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
import copy
import os
import os.path as osp
import random
import re
import numpy as np
import cv2
import json
from collections import OrderedDict
import xml.etree.ElementTree as ET
import paddlex.utils.logging as logging
from paddlex.utils import path_normalization
from .dataset import Dataset
from .dataset import is_pic
from .dataset import get_encoding


class VOCDetection(Dataset):
    """读取PascalVOC格式的检测数据集，并对样本进行相应的处理。

    Args:
        data_dir (str): 数据集所在的目录路径。
        file_list (str): 描述数据集图片文件和对应标注文件的文件路径（文本内每行路径为相对data_dir的相对路）。
        label_list (str): 描述数据集包含的类别信息文件路径。
        transforms (paddlex.det.transforms): 数据集中每个样本的预处理/增强算子。
        num_workers (int|str): 数据集中样本在预处理过程中的线程或进程数。默认为'auto'。当设为'auto'时，根据
            系统的实际CPU核数设置`num_workers`: 如果CPU核数的一半大于8，则`num_workers`为8，否则为CPU核数的
            一半。
        buffer_size (int): 数据集中样本在预处理过程中队列的缓存长度，以样本数为单位。默认为100。
        parallel_method (str): 数据集中样本在预处理过程中并行处理的方式，支持'thread'
            线程和'process'进程两种方式。默认为'process'（Windows和Mac下会强制使用thread，该参数无效）。
        shuffle (bool): 是否需要对数据集中样本打乱顺序。默认为False。
    """

    def __init__(self,
                 data_dir,
                 file_list,
                 label_list,
                 transforms=None,
                 num_workers='auto',
                 buffer_size=100,
                 parallel_method='process',
                 shuffle=False):
        from pycocotools.coco import COCO
        super(VOCDetection, self).__init__(
            transforms=transforms,
            num_workers=num_workers,
            buffer_size=buffer_size,
            parallel_method=parallel_method,
            shuffle=shuffle)
        self.file_list = list()
        self.labels = list()
        self._epoch = 0

        annotations = {}
        annotations['images'] = []
        annotations['categories'] = []
        annotations['annotations'] = []

        self.cname2cid = OrderedDict()
        self.cid2cname = OrderedDict()
        label_id = 1
        with open(label_list, 'r', encoding=get_encoding(label_list)) as fr:
            for line in fr.readlines():
                self.cname2cid[line.strip()] = label_id
                self.cid2cname[label_id] = line.strip()
                label_id += 1
                self.labels.append(line.strip())
        logging.info("Starting to read file list from dataset...")
        for k, v in self.cname2cid.items():
            annotations['categories'].append({
                'supercategory': 'component',
                'id': v,
                'name': k
            })
        ct = 0
        self.ann_ct = 0
        with open(file_list, 'r', encoding=get_encoding(file_list)) as fr:
            while True:
                line = fr.readline()
                if not line:
                    break
                if len(line.strip().split()) > 2:
                    raise Exception(
                        "A space is defined as the separator, but it exists in image or label name {}."
                        .format(line))
                img_file, xml_file = [osp.join(data_dir, x) \
                        for x in line.strip().split()[:2]]
                img_file = path_normalization(img_file)
                xml_file = path_normalization(xml_file)
                if not is_pic(img_file):
                    continue
                if not osp.isfile(xml_file):
                    continue
                if not osp.exists(img_file):
                    logging.warning('The image file {} is not exist!'.format(
                        img_file))
                    continue
                if not osp.exists(xml_file):
                    logging.warning('The annotation file {} is not exist!'.
                                    format(xml_file))
                    continue
                tree = ET.parse(xml_file)
                if tree.find('id') is None:
                    im_id = np.array([ct])
                else:
                    ct = int(tree.find('id').text)
                    im_id = np.array([int(tree.find('id').text)])
                pattern = re.compile('<object>', re.IGNORECASE)
                obj_match = pattern.findall(
                    str(ET.tostringlist(tree.getroot())))
                if len(obj_match) == 0:
                    continue
                obj_tag = obj_match[0][1:-1]
                objs = tree.findall(obj_tag)
                pattern = re.compile('<size>', re.IGNORECASE)
                size_tag = pattern.findall(
                    str(ET.tostringlist(tree.getroot())))[0][1:-1]
                size_element = tree.find(size_tag)
                pattern = re.compile('<width>', re.IGNORECASE)
                width_tag = pattern.findall(
                    str(ET.tostringlist(size_element)))[0][1:-1]
                im_w = float(size_element.find(width_tag).text)
                pattern = re.compile('<height>', re.IGNORECASE)
                height_tag = pattern.findall(
                    str(ET.tostringlist(size_element)))[0][1:-1]
                im_h = float(size_element.find(height_tag).text)
                gt_bbox = np.zeros((len(objs), 4), dtype=np.float32)
                gt_class = np.zeros((len(objs), 1), dtype=np.int32)
                gt_score = np.ones((len(objs), 1), dtype=np.float32)
                is_crowd = np.zeros((len(objs), 1), dtype=np.int32)
                difficult = np.zeros((len(objs), 1), dtype=np.int32)
                for i, obj in enumerate(objs):
                    pattern = re.compile('<name>', re.IGNORECASE)
                    name_tag = pattern.findall(str(ET.tostringlist(obj)))[0][
                        1:-1]
                    cname = obj.find(name_tag).text.strip()
                    gt_class[i][0] = self.cname2cid[cname]
                    pattern = re.compile('<difficult>', re.IGNORECASE)
                    diff_tag = pattern.findall(str(ET.tostringlist(obj)))[0][
                        1:-1]
                    try:
                        _difficult = int(obj.find(diff_tag).text)
                    except Exception:
                        _difficult = 0
                    pattern = re.compile('<bndbox>', re.IGNORECASE)
                    box_tag = pattern.findall(str(ET.tostringlist(obj)))[0][1:
                                                                            -1]
                    box_element = obj.find(box_tag)
                    pattern = re.compile('<xmin>', re.IGNORECASE)
                    xmin_tag = pattern.findall(
                        str(ET.tostringlist(box_element)))[0][1:-1]
                    x1 = float(box_element.find(xmin_tag).text)
                    pattern = re.compile('<ymin>', re.IGNORECASE)
                    ymin_tag = pattern.findall(
                        str(ET.tostringlist(box_element)))[0][1:-1]
                    y1 = float(box_element.find(ymin_tag).text)
                    pattern = re.compile('<xmax>', re.IGNORECASE)
                    xmax_tag = pattern.findall(
                        str(ET.tostringlist(box_element)))[0][1:-1]
                    x2 = float(box_element.find(xmax_tag).text)
                    pattern = re.compile('<ymax>', re.IGNORECASE)
                    ymax_tag = pattern.findall(
                        str(ET.tostringlist(box_element)))[0][1:-1]
                    y2 = float(box_element.find(ymax_tag).text)
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    if im_w > 0.5 and im_h > 0.5:
                        x2 = min(im_w - 1, x2)
                        y2 = min(im_h - 1, y2)
                    gt_bbox[i] = [x1, y1, x2, y2]
                    is_crowd[i][0] = 0
                    difficult[i][0] = _difficult
                    annotations['annotations'].append({
                        'iscrowd': 0,
                        'image_id': int(im_id[0]),
                        'bbox': [x1, y1, x2 - x1 + 1, y2 - y1 + 1],
                        'area': float((x2 - x1 + 1) * (y2 - y1 + 1)),
                        'category_id': self.cname2cid[cname],
                        'id': self.ann_ct,
                        'difficult': _difficult
                    })
                    self.ann_ct += 1

                im_info = {
                    'im_id': im_id,
                    'image_shape': np.array([im_h, im_w]).astype('int32'),
                }
                label_info = {
                    'is_crowd': is_crowd,
                    'gt_class': gt_class,
                    'gt_bbox': gt_bbox,
                    'gt_score': gt_score,
                    'gt_poly': [],
                    'difficult': difficult
                }
                voc_rec = (im_info, label_info)
                if len(objs) != 0:
                    self.file_list.append([img_file, voc_rec])
                    ct += 1
                    annotations['images'].append({
                        'height': im_h,
                        'width': im_w,
                        'id': int(im_id[0]),
                        'file_name': osp.split(img_file)[1]
                    })

        if not len(self.file_list) > 0:
            raise Exception('not found any voc record in %s' % (file_list))
        logging.info("{} samples in file {}".format(
            len(self.file_list), file_list))
        self.num_samples = len(self.file_list)
        self.coco_gt = COCO()
        self.coco_gt.dataset = annotations
        self.coco_gt.createIndex()

    def add_negative_samples(self, image_dir):
        """将背景图片加入训练

        Args:
            image_dir (str)：背景图片所在的文件夹目录。

        """
        import cv2
        if not osp.exists(image_dir):
            raise Exception("{} background images directory does not exist.".
                            format(image_dir))
        image_list = os.listdir(image_dir)
        max_img_id = max(self.coco_gt.getImgIds())
        for image in image_list:
            if not is_pic(image):
                continue
            # False ground truth
            gt_bbox = np.array([[0, 0, 1e-05, 1e-05]], dtype=np.float32)
            gt_class = np.array([[0]], dtype=np.int32)
            gt_score = np.ones((1, 1), dtype=np.float32)
            is_crowd = np.array([[0]], dtype=np.int32)
            difficult = np.zeros((1, 1), dtype=np.int32)
            gt_poly = [[[0, 0, 0, 1e-05, 1e-05, 1e-05, 1e-05, 0]]]

            max_img_id += 1
            im_fname = osp.join(image_dir, image)
            img_data = cv2.imread(im_fname, cv2.IMREAD_UNCHANGED)
            im_h, im_w, im_c = img_data.shape
            im_info = {
                'im_id': np.array([max_img_id]).astype('int32'),
                'image_shape': np.array([im_h, im_w]).astype('int32'),
            }
            label_info = {
                'is_crowd': is_crowd,
                'gt_class': gt_class,
                'gt_bbox': gt_bbox,
                'gt_score': gt_score,
                'difficult': difficult,
                'gt_poly': gt_poly
            }
            coco_rec = (im_info, label_info)
            self.file_list.append([im_fname, coco_rec])
        self.num_samples = len(self.file_list)

    def generate_image(self, templates, background, save_dir='dataset_clone'):
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
        logging.info("Gegerated images will be saved in {}".format(image_dir))
        logging.info(
            "Annotation of generated images will be saved as xml files in {}".
            format(anno_dir))
        logging.info(
            "Annotation of images (loaded before and generated now) will be saved as a COCO json file {}".
            format(json_path))
        if not osp.exists(image_dir):
            os.makedirs(image_dir)
        if not osp.exists(anno_dir):
            os.makedirs(anno_dir)

        num_objs = len(background['annos'])
        for temp in templates:
            num_objs += len(temp['annos'])

        gt_bbox = np.zeros((num_objs, 4), dtype=np.float32)
        gt_class = np.zeros((num_objs, 1), dtype=np.int32)
        gt_score = np.ones((num_objs, 1), dtype=np.float32)
        is_crowd = np.zeros((num_objs, 1), dtype=np.int32)
        difficult = np.zeros((num_objs, 1), dtype=np.int32)
        i = -1
        for i, back_anno in enumerate(background['annos']):
            gt_bbox[i] = back_anno['bbox']
            gt_class[i][0] = self.cname2cid[back_anno['category']]

        max_img_id = max(self.coco_gt.getImgIds())
        max_img_id += 1

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
                raise TypeError('Can\'t read The image file {}!'.format(
                    back_im))
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
                        category = back_anno['category']
                        if center_x > x1 and center_x < x2 and center_y > y1 and center_y < y2:
                            found = False
                            continue
                        found = True
                center = (center_x, center_y)
                back_im = cv2.seamlessClone(temp_im, back_im, temp_mask,
                                            center, cv2.MIXED_CLONE)
                i += 1
                x1 = center[0] - temp_poly_w / 2
                x2 = center[0] + temp_poly_w / 2
                y1 = center[1] - temp_poly_h / 2
                y2 = center[1] + temp_poly_h / 2
                gt_bbox[i] = [x1, y1, x2, y2]
                gt_class[i][0] = self.cname2cid[temp_category]
                self.ann_ct += 1
                self.coco_gt.dataset['annotations'].append({
                    'iscrowd': 0,
                    'image_id': max_img_id,
                    'bbox': [x1, y1, x2 - x1 + 1, y2 - y1 + 1],
                    'area': float((x2 - x1 + 1) * (y2 - y1 + 1)),
                    'category_id': self.cname2cid[temp_category],
                    'id': self.ann_ct,
                    'difficult': 0,
                })
        im_info = {
            'im_id': np.array([max_img_id]).astype('int32'),
            'image_shape': np.array([im_h, im_w]).astype('int32'),
        }
        label_info = {
            'is_crowd': is_crowd,
            'gt_class': gt_class,
            'gt_bbox': gt_bbox,
            'gt_score': gt_score,
            'difficult': difficult,
            'gt_poly': [],
        }
        self.coco_gt.dataset['images'].append({
            'height': im_h,
            'width': im_w,
            'id': max_img_id,
            'file_name': "clone_{:06d}.jpg".format(max_img_id)
        })
        coco_rec = (im_info, label_info)
        im_fname = osp.join(image_dir, "clone_{:06d}.jpg".format(max_img_id))
        cv2.imwrite(im_fname, back_im.astype('uint8'))
        self._write_xml(im_fname, im_h, im_w, im_c, label_info, anno_dir)

        self.file_list.append([im_fname, coco_rec])
        self.num_samples = len(self.file_list)
        self._write_json(self.coco_gt.dataset, save_dir)

    def iterator(self):
        self._epoch += 1
        self._pos = 0
        files = copy.deepcopy(self.file_list)
        if self.shuffle:
            random.shuffle(files)
        files = files[:self.num_samples]
        self.num_samples = len(files)
        for f in files:
            records = f[1]
            im_info = copy.deepcopy(records[0])
            label_info = copy.deepcopy(records[1])
            im_info['epoch'] = self._epoch
            if self.num_samples > 1:
                mix_idx = random.randint(1, self.num_samples - 1)
                mix_pos = (mix_idx + self._pos) % self.num_samples
            else:
                mix_pos = 0
            im_info['mixup'] = [
                files[mix_pos][0], copy.deepcopy(files[mix_pos][1][0]),
                copy.deepcopy(files[mix_pos][1][1])
            ]
            self._pos += 1
            sample = [f[0], im_info, label_info]
            yield sample

    def _write_xml(self, im_fname, im_h, im_w, im_c, label_info, anno_dir):
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
        for i in range(label_info['gt_class'].shape[0]):
            node_obj = xml_doc.createElement("object")
            node_name = xml_doc.createElement("name")
            label = self.cid2cname[gt_class[i][0]]
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
        img_name_part = osp.split(im_fname)[-1].split('.')[0]
        with open(osp.join(anno_dir, img_name_part + ".xml"), 'w') as fxml:
            xml_doc.writexml(
                fxml, indent='\t', addindent='\t', newl='\n', encoding="utf-8")

    def _write_json(self, coco_gt, save_dir):
        from paddlex.tools.base import MyEncoder
        json_path = osp.join(save_dir, "annotations.json")
        f = open(json_path, "w")
        json.dump(coco_gt, f, indent=4, cls=MyEncoder)
        f.close()
