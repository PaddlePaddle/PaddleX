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
import shutil
from enum import Enum
import traceback
import chardet
from PIL import Image
import numpy as np
import json
from ..utils import set_folder_status, get_folder_status, DatasetStatus


def copy_directory(src, dst, files=None):
    """从src目录copy文件至dst目录，
       注意:拷贝前会先清空dst中的所有文件

    Args:
        src: 源目录路径
        dst: 目标目录路径
        files: 需要拷贝的文件列表（src的相对路径）
    """
    set_folder_status(dst, DatasetStatus.XCOPYING, os.getpid())
    if files is None:
        files = list_files(src)
    try:
        message = '{} {}'.format(os.getpid(), len(files))
        set_folder_status(dst, DatasetStatus.XCOPYING, message)
        if not osp.samefile(src, dst):
            for i, f in enumerate(files):
                items = osp.split(f)
                if len(items) > 2:
                    continue
                if len(items) == 2:
                    if not osp.isdir(osp.join(dst, items[0])):
                        if osp.exists(osp.join(dst, items[0])):
                            os.remove(osp.join(dst, items[0]))
                        os.makedirs(osp.join(dst, items[0]))
                shutil.copy(osp.join(src, f), osp.join(dst, f))
        set_folder_status(dst, DatasetStatus.XCOPYDONE)
    except Exception as e:
        error_info = traceback.format_exc()
        set_folder_status(dst, DatasetStatus.XCOPYFAIL, error_info)


def is_pic(filename):
    """ 判断文件是否为图片格式

    Args:
        filename: 文件路径
    """
    suffixes = {'JPEG', 'jpeg', 'JPG', 'jpg', 'BMP', 'bmp', 'PNG', 'png'}
    suffix = filename.strip().split('.')[-1]
    if suffix not in suffixes:
        return False
    return True


def replace_ext(filename, new_ext):
    """ 替换文件后缀

    Args:
        filename: 文件路径
        new_ext: 需要替换的新的后缀
    """
    items = filename.split(".")
    items[-1] = new_ext
    new_filename = ".".join(items)
    return new_filename


def get_encoding(filename):
    """ 获取文件编码方式

    Args:
        filename: 文件路径
    """
    f = open(filename, 'rb')
    data = f.read()
    file_encoding = chardet.detect(data).get('encoding')
    return file_encoding


def pil_imread(file_path):
    """ 获取分割标注图片信息

    Args:
        filename: 文件路径
    """
    im = Image.open(file_path)
    return np.asarray(im)


def check_list_txt(list_txts):
    """ 检查切分信息文件的格式

    Args:
        list_txts: 包含切分信息文件路径的list
    """
    for list_txt in list_txts:
        if not osp.exists(list_txt):
            continue
        with open(list_txt) as f:
            for line in f:
                items = line.strip().split()
                if len(items) != 2:
                    raise Exception('{} 格式错误. 列表应包含两列，由空格分离。'.format(list_txt))


def read_seg_ann(pngfile):
    """ 解析语义分割的标注png图片

    Args:
        pngfile: 包含标注信息的png图片路径
    """
    grt = pil_imread(pngfile)
    labels = list(np.unique(grt))
    if 255 in labels:
        labels.remove(255)
    return labels, grt.shape


def read_coco_ann(img_id, coco, cid2cname, catid2clsid):
    img_anno = coco.loadImgs(img_id)[0]
    im_w = float(img_anno['width'])
    im_h = float(img_anno['height'])

    ins_anno_ids = coco.getAnnIds(imgIds=img_id, iscrowd=0)
    instances = coco.loadAnns(ins_anno_ids)

    bboxes = []
    for inst in instances:
        x, y, box_w, box_h = inst['bbox']
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(im_w - 1, x1 + max(0, box_w - 1))
        y2 = min(im_h - 1, y1 + max(0, box_h - 1))
        if inst['area'] > 0 and x2 >= x1 and y2 >= y1:
            inst['clean_bbox'] = [x1, y1, x2, y2]
            bboxes.append(inst)
        else:
            raise Exception("标注文件存在错误")
    num_bbox = len(bboxes)

    gt_bbox = np.zeros((num_bbox, 4), dtype=np.float32)
    gt_class = [""] * num_bbox
    gt_poly = [None] * num_bbox

    for i, box in enumerate(bboxes):
        catid = box['category_id']
        gt_class[i] = cid2cname[catid2clsid[catid]]
        gt_bbox[i, :] = box['clean_bbox']
        # is_crowd[i][0] = box['iscrowd']
        if 'segmentation' in box:
            gt_poly[i] = box['segmentation']

    anno_dict = {
        'h': im_h,
        'w': im_w,
        'gt_class': gt_class,
        'gt_bbox': gt_bbox,
        'gt_poly': gt_poly,
    }
    return anno_dict


def get_npy_from_coco_json(coco, npy_path, files):
    """ 从实例分割标注的json文件中，获取每张图片的信息，并存为npy格式

    Args:
        coco: 从json文件中解析出的标注信息
        npy_path: npy文件保存的地址
        files: 需要生成npy文件的目录
    """
    img_ids = coco.getImgIds()
    cat_ids = coco.getCatIds()
    anno_ids = coco.getAnnIds()
    catid2clsid = dict({catid: i for i, catid in enumerate(cat_ids)})
    cid2cname = dict({
        clsid: coco.loadCats(catid)[0]['name']
        for catid, clsid in catid2clsid.items()
    })
    iname2id = dict()
    for img_id in img_ids:
        img_name = osp.split(coco.loadImgs(img_id)[0]["file_name"])[-1]
        iname2id[img_name] = img_id

    if not osp.exists(npy_path):
        os.makedirs(npy_path)

    for img in files:
        img_id = iname2id[osp.split(img)[-1]]
        anno_dict = read_coco_ann(img_id, coco, cid2cname, catid2clsid)

        img_name = osp.split(img)[-1]
        npy_name = replace_ext(img_name, "npy")
        np.save(osp.join(npy_path, npy_name), anno_dict)


def get_label_count(label_info):
    """ 根据存储的label_info字段，计算label_count字段

    Args:
        label_info: 存储的label_info
    """
    label_count = dict()
    for key in sorted(label_info):
        label_count[key] = len(label_info[key])
    return label_count


class MyEncoder(json.JSONEncoder):
    # 调整json文件存储形式
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)
