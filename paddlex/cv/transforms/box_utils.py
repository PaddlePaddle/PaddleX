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


def data_anchor_sampling(bbox_labels, image_width, image_height, scale_array,
                         resize_width):
    num_gt = len(bbox_labels)
    # np.random.randint range: [low, high)
    rand_idx = np.random.randint(0, num_gt) if num_gt != 0 else 0

    if num_gt != 0:
        norm_xmin = bbox_labels[rand_idx][0]
        norm_ymin = bbox_labels[rand_idx][1]
        norm_xmax = bbox_labels[rand_idx][2]
        norm_ymax = bbox_labels[rand_idx][3]

        xmin = norm_xmin * image_width
        ymin = norm_ymin * image_height
        wid = image_width * (norm_xmax - norm_xmin)
        hei = image_height * (norm_ymax - norm_ymin)
        range_size = 0

        area = wid * hei
        for scale_ind in range(0, len(scale_array) - 1):
            if area > scale_array[scale_ind] ** 2 and area < \
                    scale_array[scale_ind + 1] ** 2:
                range_size = scale_ind + 1
                break

        if area > scale_array[len(scale_array) - 2]**2:
            range_size = len(scale_array) - 2

        scale_choose = 0.0
        if range_size == 0:
            rand_idx_size = 0
        else:
            # np.random.randint range: [low, high)
            rng_rand_size = np.random.randint(0, range_size + 1)
            rand_idx_size = rng_rand_size % (range_size + 1)

        if rand_idx_size == range_size:
            min_resize_val = scale_array[rand_idx_size] / 2.0
            max_resize_val = min(2.0 * scale_array[rand_idx_size],
                                 2 * math.sqrt(wid * hei))
            scale_choose = random.uniform(min_resize_val, max_resize_val)
        else:
            min_resize_val = scale_array[rand_idx_size] / 2.0
            max_resize_val = 2.0 * scale_array[rand_idx_size]
            scale_choose = random.uniform(min_resize_val, max_resize_val)

        sample_bbox_size = wid * resize_width / scale_choose

        w_off_orig = 0.0
        h_off_orig = 0.0
        if sample_bbox_size < max(image_height, image_width):
            if wid <= sample_bbox_size:
                w_off_orig = np.random.uniform(xmin + wid - sample_bbox_size,
                                               xmin)
            else:
                w_off_orig = np.random.uniform(xmin,
                                               xmin + wid - sample_bbox_size)

            if hei <= sample_bbox_size:
                h_off_orig = np.random.uniform(ymin + hei - sample_bbox_size,
                                               ymin)
            else:
                h_off_orig = np.random.uniform(ymin,
                                               ymin + hei - sample_bbox_size)

        else:
            w_off_orig = np.random.uniform(image_width - sample_bbox_size, 0.0)
            h_off_orig = np.random.uniform(image_height - sample_bbox_size, 0.0)

        w_off_orig = math.floor(w_off_orig)
        h_off_orig = math.floor(h_off_orig)

        # Figure out top left coordinates.
        w_off = float(w_off_orig / image_width)
        h_off = float(h_off_orig / image_height)

        sampled_bbox = [
            w_off, h_off, w_off + float(sample_bbox_size / image_width),
            h_off + float(sample_bbox_size / image_height)
        ]
        return sampled_bbox
    else:
        return 0
    
    
def bbox_area_sampling(bboxes, labels, scores, target_size, min_size):
    new_bboxes = []
    new_labels = []
    new_scores = []
    for i, bbox in enumerate(bboxes):
        w = float((bbox[2] - bbox[0]) * target_size)
        h = float((bbox[3] - bbox[1]) * target_size)
        if w * h < float(min_size * min_size):
            continue
        else:
            new_bboxes.append(bbox)
            new_labels.append(labels[i])
            if scores is not None and scores.size != 0:
                new_scores.append(scores[i])
    bboxes = np.array(new_bboxes)
    labels = np.array(new_labels)
    scores = np.array(new_scores)
    return bboxes, labels, scores


def satisfy_sample_constraint_coverage(sampler, sample_bbox, gt_bboxes):
    if sampler[6] == 0 and sampler[7] == 0:
        has_jaccard_overlap = False
    else:
        has_jaccard_overlap = True
    if sampler[8] == 0 and sampler[9] == 0:
        has_object_coverage = False
    else:
        has_object_coverage = True

    if not has_jaccard_overlap and not has_object_coverage:
        return True
    found = False
    for i in range(len(gt_bboxes)):
        object_bbox = [
            gt_bboxes[i][0], gt_bboxes[i][1], gt_bboxes[i][2], gt_bboxes[i][3]
        ]
        if has_jaccard_overlap:
            overlap = jaccard_overlap(sample_bbox, object_bbox)
            if sampler[6] != 0 and \
                    overlap < sampler[6]:
                continue
            if sampler[7] != 0 and \
                    overlap > sampler[7]:
                continue
            found = True
        if has_object_coverage:
            object_coverage = bbox_coverage(object_bbox, sample_bbox)
            if sampler[8] != 0 and \
                    object_coverage < sampler[8]:
                continue
            if sampler[9] != 0 and \
                    object_coverage > sampler[9]:
                continue
            found = True
        if found:
            return True
    return found


def filter_and_process(sample_bbox, bboxes, labels, scores=None):
    new_bboxes = []
    new_labels = []
    new_scores = []
    for i in range(len(bboxes)):
        new_bbox = [0, 0, 0, 0]
        obj_bbox = [bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3]]
        if not meet_emit_constraint(obj_bbox, sample_bbox):
            continue
        if not is_overlap(obj_bbox, sample_bbox):
            continue
        sample_width = sample_bbox[2] - sample_bbox[0]
        sample_height = sample_bbox[3] - sample_bbox[1]
        new_bbox[0] = (obj_bbox[0] - sample_bbox[0]) / sample_width
        new_bbox[1] = (obj_bbox[1] - sample_bbox[1]) / sample_height
        new_bbox[2] = (obj_bbox[2] - sample_bbox[0]) / sample_width
        new_bbox[3] = (obj_bbox[3] - sample_bbox[1]) / sample_height
        new_bbox = clip_bbox(new_bbox)
        if bbox_area(new_bbox) > 0:
            new_bboxes.append(new_bbox)
            new_labels.append([labels[i][0]])
            if scores is not None:
                new_scores.append([scores[i][0]])
    bboxes = np.array(new_bboxes)
    labels = np.array(new_labels)
    scores = np.array(new_scores)
    return bboxes, labels, scores


def crop_image_sampling(img, sample_bbox, image_width, image_height,
                        target_size):
    # no clipping here
    xmin = int(sample_bbox[0] * image_width)
    xmax = int(sample_bbox[2] * image_width)
    ymin = int(sample_bbox[1] * image_height)
    ymax = int(sample_bbox[3] * image_height)

    w_off = xmin
    h_off = ymin
    width = xmax - xmin
    height = ymax - ymin
    cross_xmin = max(0.0, float(w_off))
    cross_ymin = max(0.0, float(h_off))
    cross_xmax = min(float(w_off + width - 1.0), float(image_width))
    cross_ymax = min(float(h_off + height - 1.0), float(image_height))
    cross_width = cross_xmax - cross_xmin
    cross_height = cross_ymax - cross_ymin

    roi_xmin = 0 if w_off >= 0 else abs(w_off)
    roi_ymin = 0 if h_off >= 0 else abs(h_off)
    roi_width = cross_width
    roi_height = cross_height

    roi_y1 = int(roi_ymin)
    roi_y2 = int(roi_ymin + roi_height)
    roi_x1 = int(roi_xmin)
    roi_x2 = int(roi_xmin + roi_width)

    cross_y1 = int(cross_ymin)
    cross_y2 = int(cross_ymin + cross_height)
    cross_x1 = int(cross_xmin)
    cross_x2 = int(cross_xmin + cross_width)

    sample_img = np.zeros((height, width, 3))
    sample_img[roi_y1: roi_y2, roi_x1: roi_x2] = \
        img[cross_y1: cross_y2, cross_x1: cross_x2]

    sample_img = cv2.resize(
        sample_img, (target_size, target_size), interpolation=cv2.INTER_AREA)

    return sample_img


def generate_sample_bbox_square(sampler, image_width, image_height):
    scale = np.random.uniform(sampler[2], sampler[3])
    aspect_ratio = np.random.uniform(sampler[4], sampler[5])
    aspect_ratio = max(aspect_ratio, (scale**2.0))
    aspect_ratio = min(aspect_ratio, 1 / (scale**2.0))
    bbox_width = scale * (aspect_ratio**0.5)
    bbox_height = scale / (aspect_ratio**0.5)
    if image_height < image_width:
        bbox_width = bbox_height * image_height / image_width
    else:
        bbox_height = bbox_width * image_width / image_height
    xmin_bound = 1 - bbox_width
    ymin_bound = 1 - bbox_height
    xmin = np.random.uniform(0, xmin_bound)
    ymin = np.random.uniform(0, ymin_bound)
    xmax = xmin + bbox_width
    ymax = ymin + bbox_height
    sampled_bbox = [xmin, ymin, xmax, ymax]
    return sampled_bbox