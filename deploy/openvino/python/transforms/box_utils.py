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
