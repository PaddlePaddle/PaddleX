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

import numpy as np
import random
import math
import cv2
import scipy


def bbox_area(src_bbox):
    if src_bbox[2] < src_bbox[0] or src_bbox[3] < src_bbox[1]:
        return 0.
    else:
        width = src_bbox[2] - src_bbox[0]
        height = src_bbox[3] - src_bbox[1]
        return width * height


def jaccard_overlap(sample_bbox, object_bbox):
    if sample_bbox[0] >= object_bbox[2] or \
        sample_bbox[2] <= object_bbox[0] or \
        sample_bbox[1] >= object_bbox[3] or \
        sample_bbox[3] <= object_bbox[1]:
        return 0
    intersect_xmin = max(sample_bbox[0], object_bbox[0])
    intersect_ymin = max(sample_bbox[1], object_bbox[1])
    intersect_xmax = min(sample_bbox[2], object_bbox[2])
    intersect_ymax = min(sample_bbox[3], object_bbox[3])
    intersect_size = (intersect_xmax - intersect_xmin) * (
        intersect_ymax - intersect_ymin)
    sample_bbox_size = bbox_area(sample_bbox)
    object_bbox_size = bbox_area(object_bbox)
    overlap = intersect_size / (
        sample_bbox_size + object_bbox_size - intersect_size)
    return overlap


def iou_matrix(a, b):
    tl_i = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    br_i = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(br_i - tl_i, axis=2) * (tl_i < br_i).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    area_o = (area_a[:, np.newaxis] + area_b - area_i)
    return area_i / (area_o + 1e-10)


def crop_box_with_center_constraint(box, crop):
    cropped_box = box.copy()

    cropped_box[:, :2] = np.maximum(box[:, :2], crop[:2])
    cropped_box[:, 2:] = np.minimum(box[:, 2:], crop[2:])
    cropped_box[:, :2] -= crop[:2]
    cropped_box[:, 2:] -= crop[:2]

    centers = (box[:, :2] + box[:, 2:]) / 2
    valid = np.logical_and(crop[:2] <= centers, centers < crop[2:]).all(axis=1)
    valid = np.logical_and(
        valid, (cropped_box[:, :2] < cropped_box[:, 2:]).all(axis=1))

    return cropped_box, np.where(valid)[0]


def is_poly(segm):
    if not isinstance(segm, (list, dict)):
        raise Exception("Invalid segm type: {}".format(type(segm)))
    return isinstance(segm, list)


def crop_image(img, crop):
    x1, y1, x2, y2 = crop
    return img[y1:y2, x1:x2, :]


def crop_segms(segms, valid_ids, crop, height, width):
    def _crop_poly(segm, crop):
        xmin, ymin, xmax, ymax = crop
        crop_coord = [xmin, ymin, xmin, ymax, xmax, ymax, xmax, ymin]
        crop_p = np.array(crop_coord).reshape(4, 2)
        crop_p = Polygon(crop_p)

        crop_segm = list()
        for poly in segm:
            poly = np.array(poly).reshape(len(poly) // 2, 2)
            polygon = Polygon(poly)
            if not polygon.is_valid:
                exterior = polygon.exterior
                multi_lines = exterior.intersection(exterior)
                polygons = shapely.ops.polygonize(multi_lines)
                polygon = MultiPolygon(polygons)
            multi_polygon = list()
            if isinstance(polygon, MultiPolygon):
                multi_polygon = copy.deepcopy(polygon)
            else:
                multi_polygon.append(copy.deepcopy(polygon))
            for per_polygon in multi_polygon:
                inter = per_polygon.intersection(crop_p)
                if not inter:
                    continue
                if isinstance(inter, (MultiPolygon, GeometryCollection)):
                    for part in inter:
                        if not isinstance(part, Polygon):
                            continue
                        part = np.squeeze(
                            np.array(part.exterior.coords[:-1]).reshape(1, -1))
                        part[0::2] -= xmin
                        part[1::2] -= ymin
                        crop_segm.append(part.tolist())
                elif isinstance(inter, Polygon):
                    crop_poly = np.squeeze(
                        np.array(inter.exterior.coords[:-1]).reshape(1, -1))
                    crop_poly[0::2] -= xmin
                    crop_poly[1::2] -= ymin
                    crop_segm.append(crop_poly.tolist())
                else:
                    continue
        return crop_segm

    def _crop_rle(rle, crop, height, width):
        if 'counts' in rle and type(rle['counts']) == list:
            rle = mask_util.frPyObjects(rle, height, width)
        mask = mask_util.decode(rle)
        mask = mask[crop[1]:crop[3], crop[0]:crop[2]]
        rle = mask_util.encode(np.array(mask, order='F', dtype=np.uint8))
        return rle

    crop_segms = []
    for id in valid_ids:
        segm = segms[id]
        if is_poly(segm):
            import copy
            import shapely.ops
            import logging
            from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
            logging.getLogger("shapely").setLevel(logging.WARNING)
            # Polygon format
            crop_segms.append(_crop_poly(segm, crop))
        else:
            # RLE format
            import pycocotools.mask as mask_util
            crop_segms.append(_crop_rle(segm, crop, height, width))
    return crop_segms


def expand_segms(segms, x, y, height, width, ratio):
    def _expand_poly(poly, x, y):
        expanded_poly = np.array(poly)
        expanded_poly[0::2] += x
        expanded_poly[1::2] += y
        return expanded_poly.tolist()

    def _expand_rle(rle, x, y, height, width, ratio):
        if 'counts' in rle and type(rle['counts']) == list:
            rle = mask_util.frPyObjects(rle, height, width)
        mask = mask_util.decode(rle)
        expanded_mask = np.full((int(height * ratio), int(width * ratio)),
                                0).astype(mask.dtype)
        expanded_mask[y:y + height, x:x + width] = mask
        rle = mask_util.encode(
            np.array(expanded_mask, order='F', dtype=np.uint8))
        return rle

    expanded_segms = []
    for segm in segms:
        if is_poly(segm):
            # Polygon format
            expanded_segms.append([_expand_poly(poly, x, y) for poly in segm])
        else:
            # RLE format
            import pycocotools.mask as mask_util
            expanded_segms.append(
                _expand_rle(segm, x, y, height, width, ratio))
    return expanded_segms


def box_horizontal_flip(bboxes, width):
    oldx1 = bboxes[:, 0].copy()
    oldx2 = bboxes[:, 2].copy()
    bboxes[:, 0] = width - oldx2 - 1
    bboxes[:, 2] = width - oldx1 - 1
    if bboxes.shape[0] != 0 and (bboxes[:, 2] < bboxes[:, 0]).all():
        raise ValueError(
            "RandomHorizontalFlip: invalid box, x2 should be greater than x1")
    return bboxes


def segms_horizontal_flip(segms, height, width):
    def _flip_poly(poly, width):
        flipped_poly = np.array(poly)
        flipped_poly[0::2] = width - np.array(poly[0::2]) - 1
        return flipped_poly.tolist()

    def _flip_rle(rle, height, width):
        if 'counts' in rle and type(rle['counts']) == list:
            rle = mask_util.frPyObjects([rle], height, width)
        mask = mask_util.decode(rle)
        mask = mask[:, ::-1]
        rle = mask_util.encode(np.array(mask, order='F', dtype=np.uint8))
        return rle

    flipped_segms = []
    for segm in segms:
        if is_poly(segm):
            # Polygon format
            flipped_segms.append([_flip_poly(poly, width) for poly in segm])
        else:
            # RLE format
            import pycocotools.mask as mask_util
            flipped_segms.append(_flip_rle(segm, height, width))
    return flipped_segms


class GenerateYoloTarget(object):
    """生成YOLOv3的ground truth（真实标注框）在不同特征层的位置转换信息。
       该transform只在YOLOv3计算细粒度loss时使用。
       
       Args:
           anchors (list|tuple): anchor框的宽度和高度。
           anchor_masks (list|tuple): 在计算损失时，使用anchor的mask索引。
           num_classes (int): 类别数。默认为80。
           iou_thresh (float): iou阈值，当anchor和真实标注框的iou大于该阈值时，计入target。默认为1.0。
    """
    
    def __init__(self, 
                 anchors, 
                 anchor_masks,
                 num_classes=80,
                 iou_thresh=1.):
        super(GenerateYoloTarget, self).__init__()
        self.anchors = anchors
        self.anchor_masks = anchor_masks
        self.num_classes = num_classes
        self.iou_thresh = iou_thresh
        
    def __call__(self, batch_data):
        """
        Args:
            batch_data (list): 由与图像相关的各种信息组成的batch数据。

        Returns:
            list: 由与图像相关的各种信息组成的batch数据。
                  其中，每个数据新添加的字段为：
                           - target0 (np.ndarray): YOLOv3的ground truth在特征层0的位置转换信息，
                                   形状为(特征层0的anchor数量, 6+类别数, 特征层0的h, 特征层0的w)。
                           - target1 (np.ndarray): YOLOv3的ground truth在特征层1的位置转换信息，
                                   形状为(特征层1的anchor数量, 6+类别数, 特征层1的h, 特征层1的w)。
                           - ...
                           -targetn (np.ndarray): YOLOv3的ground truth在特征层n的位置转换信息，
                                   形状为(特征层n的anchor数量, 6+类别数, 特征层n的h, 特征层n的w)。
                    n的是大小由anchor_masks的长度决定。
        """
        im = batch_data[0][0]
        h = im.shape[1]
        w = im.shape[2]
        an_hw = np.array(self.anchors) / np.array([[w, h]])
        for data_id, data in enumerate(batch_data):
            gt_bbox = data[1]
            gt_class = data[2]
            gt_score = data[3]
            im_shape = data[4]
            origin_h = float(im_shape[0])
            origin_w = float(im_shape[1])
            data_list = list(data)
            for i, mask in enumerate(self.anchor_masks):
                downsample_ratio = 32 // pow(2, i)
                grid_h = int(h / downsample_ratio)
                grid_w = int(w / downsample_ratio)
                target = np.zeros(
                    (len(mask), 6 + self.num_classes, grid_h, grid_w),
                    dtype=np.float32)
                for b in range(gt_bbox.shape[0]):
                    gx = gt_bbox[b, 0] / float(origin_w)
                    gy = gt_bbox[b, 1] / float(origin_h)
                    gw = gt_bbox[b, 2] / float(origin_w)
                    gh = gt_bbox[b, 3] / float(origin_h)
                    cls = gt_class[b]
                    score = gt_score[b]
                    if gw <= 0. or gh <= 0. or score <= 0.:
                        continue
                    # find best match anchor index
                    best_iou = 0.
                    best_idx = -1
                    for an_idx in range(an_hw.shape[0]):
                        iou = jaccard_overlap(
                            [0., 0., gw, gh],
                            [0., 0., an_hw[an_idx, 0], an_hw[an_idx, 1]])
                        if iou > best_iou:
                            best_iou = iou
                            best_idx = an_idx
                    gi = int(gx * grid_w)
                    gj = int(gy * grid_h)
                    # gtbox should be regresed in this layes if best match 
                    # anchor index in anchor mask of this layer
                    if best_idx in mask:
                        best_n = mask.index(best_idx)
                        # x, y, w, h, scale
                        target[best_n, 0, gj, gi] = gx * grid_w - gi
                        target[best_n, 1, gj, gi] = gy * grid_h - gj
                        target[best_n, 2, gj, gi] = np.log(
                            gw * w / self.anchors[best_idx][0])
                        target[best_n, 3, gj, gi] = np.log(
                            gh * h / self.anchors[best_idx][1])
                        target[best_n, 4, gj, gi] = 2.0 - gw * gh
                        # objectness record gt_score
                        target[best_n, 5, gj, gi] = score
                        # classification
                        target[best_n, 6 + cls, gj, gi] = 1.
                    # For non-matched anchors, calculate the target if the iou 
                    # between anchor and gt is larger than iou_thresh
                    if self.iou_thresh < 1:
                        for idx, mask_i in enumerate(mask):
                            if mask_i == best_idx: continue
                            iou = jaccard_overlap(
                                [0., 0., gw, gh],
                                [0., 0., an_hw[mask_i, 0], an_hw[mask_i, 1]])
                            if iou > self.iou_thresh:
                                # x, y, w, h, scale
                                target[idx, 0, gj, gi] = gx * grid_w - gi
                                target[idx, 1, gj, gi] = gy * grid_h - gj
                                target[idx, 2, gj, gi] = np.log(
                                    gw * w / self.anchors[mask_i][0])
                                target[idx, 3, gj, gi] = np.log(
                                    gh * h / self.anchors[mask_i][1])
                                target[idx, 4, gj, gi] = 2.0 - gw * gh
                                # objectness record gt_score
                                target[idx, 5, gj, gi] = score
                                # classification
                                target[idx, 6 + cls, gj, gi] = 1.
                data_list.append(target)
            batch_data[data_id] = tuple(data_list)
        return batch_data   
