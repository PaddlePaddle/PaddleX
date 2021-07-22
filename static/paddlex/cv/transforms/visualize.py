# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import os
import os.path as osp
import cv2
from PIL import Image
import numpy as np
import math
from .imgaug_support import execute_imgaug
from .cls_transforms import ClsTransform
from .det_transforms import DetTransform
from .seg_transforms import SegTransform
import paddlex as pdx
from paddlex.cv.models.utils.visualize import get_color_map_list


def _draw_rectangle_and_cname(img, xmin, ymin, xmax, ymax, cname, color):
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
    return img


def cls_compose(im, label=None, transforms=None, vdl_writer=None, step=0):
    """
        Args:
            im (str/np.ndarray): 图像路径/图像np.ndarray数据。
            label (int): 每张图像所对应的类别序号。
            vdl_writer (visualdl.LogWriter): VisualDL存储器，日志信息将保存在其中。
                当为None时，不对日志进行保存。默认为None。
            step (int): 数据预处理的轮数，当vdl_writer不为None时有效。默认为0。

        Returns:
            tuple: 根据网络所需字段所组成的tuple；
                字段由transforms中的最后一个数据预处理操作决定。
        """
    if isinstance(im, np.ndarray):
        if len(im.shape) != 3:
            raise Exception(
                "im should be 3-dimension, but now is {}-dimensions".format(
                    len(im.shape)))
    else:
        try:
            im = cv2.imread(im).astype('float32')
        except:
            raise TypeError('Can\'t read The image file {}!'.format(im))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if vdl_writer is not None:
        vdl_writer.add_image(
            tag='0. OriginalImage/' + str(step), img=im, step=0)
    op_id = 1
    for op in transforms:
        if isinstance(op, ClsTransform):
            if vdl_writer is not None and hasattr(op, 'prob'):
                op.prob = 1.0
            outputs = op(im, label)
            im = outputs[0]
            if len(outputs) == 2:
                label = outputs[1]
            if isinstance(op, pdx.cv.transforms.cls_transforms.Normalize):
                continue
        else:
            import imgaug.augmenters as iaa
            if isinstance(op, iaa.Augmenter):
                im = execute_imgaug(op, im)
            outputs = (im, )
            if label is not None:
                outputs = (im, label)
        if vdl_writer is not None:
            tag = str(op_id) + '. ' + op.__class__.__name__ + '/' + str(step)
            vdl_writer.add_image(tag=tag, img=im, step=0)
        op_id += 1


def det_compose(im,
                im_info=None,
                label_info=None,
                transforms=None,
                vdl_writer=None,
                step=0,
                labels=[],
                catid2color=None):
    def decode_image(im_file, im_info, label_info):
        if im_info is None:
            im_info = dict()
        if isinstance(im_file, np.ndarray):
            if len(im_file.shape) != 3:
                raise Exception(
                    "im should be 3-dimensions, but now is {}-dimensions".
                    format(len(im_file.shape)))
            im = im_file
        else:
            try:
                im = cv2.imread(im_file).astype('float32')
            except:
                raise TypeError('Can\'t read The image file {}!'.format(
                    im_file))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        # make default im_info with [h, w, 1]
        im_info['im_resize_info'] = np.array(
            [im.shape[0], im.shape[1], 1.], dtype=np.float32)
        im_info['image_shape'] = np.array([im.shape[0],
                                           im.shape[1]]).astype('int32')
        use_mixup = False
        for t in transforms:
            if type(t).__name__ == 'MixupImage':
                use_mixup = True
            if not use_mixup:
                if 'mixup' in im_info:
                    del im_info['mixup']
        # decode mixup image
        if 'mixup' in im_info:
            im_info['mixup'] = \
              decode_image(im_info['mixup'][0],
                           im_info['mixup'][1],
                           im_info['mixup'][2])
        if label_info is None:
            return (im, im_info)
        else:
            return (im, im_info, label_info)

    outputs = decode_image(im, im_info, label_info)
    im = outputs[0]
    im_info = outputs[1]
    if len(outputs) == 3:
        label_info = outputs[2]
    if vdl_writer is not None:
        vdl_writer.add_image(
            tag='0. OriginalImage/' + str(step), img=im, step=0)
    op_id = 1
    bboxes = label_info['gt_bbox']
    transforms = [None] + transforms
    for op in transforms:
        if im is None:
            return None
        if isinstance(op, DetTransform) or op is None:
            if vdl_writer is not None and hasattr(op, 'prob'):
                op.prob = 1.0
            if op is not None:
                outputs = op(im, im_info, label_info)
            else:
                outputs = (im, im_info, label_info)
            im = outputs[0]
            vdl_im = im
            if vdl_writer is not None:
                if isinstance(op,
                              pdx.cv.transforms.det_transforms.ResizeByShort):
                    scale = outputs[1]['im_resize_info'][2]
                    bboxes = bboxes * scale
                elif isinstance(op, pdx.cv.transforms.det_transforms.Resize):
                    h = outputs[1]['image_shape'][0]
                    w = outputs[1]['image_shape'][1]
                    target_size = op.target_size
                    if isinstance(target_size, int):
                        h_scale = float(target_size) / h
                        w_scale = float(target_size) / w
                    else:
                        h_scale = float(target_size[0]) / h
                        w_scale = float(target_size[1]) / w
                    bboxes[:, 0] = bboxes[:, 0] * w_scale
                    bboxes[:, 1] = bboxes[:, 1] * h_scale
                    bboxes[:, 2] = bboxes[:, 2] * w_scale
                    bboxes[:, 3] = bboxes[:, 3] * h_scale
                else:
                    bboxes = outputs[2]['gt_bbox']
                if not isinstance(op, (
                        pdx.cv.transforms.det_transforms.RandomHorizontalFlip,
                        pdx.cv.transforms.det_transforms.Padding)):
                    for i in range(bboxes.shape[0]):
                        bbox = bboxes[i]
                        cname = labels[outputs[2]['gt_class'][i][0] - 1]
                        vdl_im = _draw_rectangle_and_cname(
                            vdl_im,
                            int(bbox[0]),
                            int(bbox[1]),
                            int(bbox[2]),
                            int(bbox[3]), cname,
                            catid2color[outputs[2]['gt_class'][i][0] - 1])
                if isinstance(op, pdx.cv.transforms.det_transforms.Normalize):
                    continue
        else:
            im = execute_imgaug(op, im)
            if label_info is not None:
                outputs = (im, im_info, label_info)
            else:
                outputs = (im, im_info)
            vdl_im = im
        if vdl_writer is not None:
            tag = str(op_id) + '. ' + op.__class__.__name__ + '/' + str(step)
            if op is None:
                tag = str(op_id) + '. OriginalImageWithGTBox/' + str(step)
            vdl_writer.add_image(tag=tag, img=vdl_im, step=0)
        op_id += 1


def seg_compose(im,
                im_info=None,
                label=None,
                transforms=None,
                vdl_writer=None,
                step=0):
    if im_info is None:
        im_info = list()
    if isinstance(im, np.ndarray):
        if len(im.shape) != 3:
            raise Exception(
                "im should be 3-dimensions, but now is {}-dimensions".format(
                    len(im.shape)))
    else:
        try:
            im = cv2.imread(im)
        except:
            raise ValueError('Can\'t read The image file {}!'.format(im))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im.astype('float32')
    h, w, c = im.shape
    if label is not None:
        if not isinstance(label, np.ndarray):
            label = np.asarray(Image.open(label))
    if vdl_writer is not None:
        for i in range(0, c, 3):
            if c > 3:
                tag = '0. OriginalImage/{}_{}'.format(str(step), str(i // 3))
            else:
                tag = '0. OriginalImage/{}'.format(str(step))
            vdl_writer.add_image(tag=tag, img=im[:, :, i:i + 3], step=0)
    op_id = 1
    for op in transforms:
        if isinstance(op, SegTransform):
            outputs = op(im, im_info, label)
            im = outputs[0]
            if len(outputs) >= 2:
                im_info = outputs[1]
            if len(outputs) == 3:
                label = outputs[2]
            if isinstance(op, pdx.cv.transforms.seg_transforms.Normalize):
                continue
        else:
            im = execute_imgaug(op, im)
            if label is not None:
                outputs = (im, im_info, label)
            else:
                outputs = (im, im_info)
        if vdl_writer is not None:
            for i in range(0, c, 3):
                if c > 3:
                    tag = str(
                        op_id) + '. ' + op.__class__.__name__ + '/' + str(
                            step) + '_' + str(i // 3)
                else:
                    tag = str(
                        op_id) + '. ' + op.__class__.__name__ + '/' + str(step)
                vdl_writer.add_image(tag=tag, img=im[:, :, i:i + 3], step=0)
        op_id += 1


def visualize(dataset, img_count=3, save_dir='vdl_output'):
    '''对数据预处理/增强中间结果进行可视化。
    可使用VisualDL查看中间结果：
    1. VisualDL启动方式: visualdl --logdir vdl_output --port 8001
    2. 浏览器打开 https://0.0.0.0:8001即可，
        其中0.0.0.0为本机访问，如为远程服务, 改成相应机器IP

    Args:
        dataset (paddlex.datasets): 数据集读取器。
        img_count (int): 需要进行数据预处理/增强的图像数目。默认为3。
        save_dir (str): 日志保存的路径。默认为'vdl_output'。
    '''
    if dataset.num_samples < img_count:
        img_count = dataset.num_samples
    transforms = dataset.transforms
    if not osp.isdir(save_dir):
        if osp.exists(save_dir):
            os.remove(save_dir)
        os.makedirs(save_dir)
    from visualdl import LogWriter
    vdl_save_dir = osp.join(save_dir, 'image_transforms')
    vdl_writer = LogWriter(vdl_save_dir)
    for i, data in enumerate(dataset.iterator()):
        if i == img_count:
            break
        data.append(transforms.transforms)
        data.append(vdl_writer)
        data.append(i)
        if isinstance(transforms, ClsTransform):
            cls_compose(*data)
        elif isinstance(transforms, DetTransform):
            labels = dataset.labels
            color_map = get_color_map_list(len(labels) + 1)
            catid2color = {}
            for catid in range(len(labels)):
                catid2color[catid] = color_map[catid + 1]
            data.append(labels)
            data.append(catid2color)
            det_compose(*data)
        elif isinstance(transforms, SegTransform):
            seg_compose(*data)
        else:
            raise Exception('The transform must the subclass of \
                    ClsTransform or DetTransform or SegTransform!')
