# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import cv2
import numpy as np

import shapely.ops
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
import copy


def normalize(im, mean, std, min_value=[0, 0, 0], max_value=[255, 255, 255]):
    # Rescaling (min-max normalization)
    range_value = np.asarray(
        [1. / (max_value[i] - min_value[i]) for i in range(len(max_value))],
        dtype=np.float32)
    im = (im - np.asarray(min_value, dtype=np.float32)) * range_value

    # Standardization (Z-score Normalization)
    im -= mean
    im /= std
    return im


def permute(im, to_bgr=False):
    im = np.swapaxes(im, 1, 2)
    im = np.swapaxes(im, 1, 0)
    if to_bgr:
        im = im[[2, 1, 0], :, :]
    return im


def center_crop(im, crop_size=224):
    height, width = im.shape[:2]
    w_start = (width - crop_size) // 2
    h_start = (height - crop_size) // 2
    w_end = w_start + crop_size
    h_end = h_start + crop_size
    im = im[h_start:h_end, w_start:w_end, ...]
    return im


def horizontal_flip(im):
    im = im[:, ::-1, ...]
    return im


def vertical_flip(im):
    im = im[::-1, :, ...]
    return im


def rgb2bgr(im):
    return im[:, :, ::-1]


def is_poly(poly):
    assert isinstance(poly, (list, dict)), \
        "Invalid poly type: {}".format(type(poly))
    return isinstance(poly, list)


def horizontal_flip_poly(poly, width):
    flipped_poly = np.array(poly)
    flipped_poly[0::2] = width - np.array(poly[0::2])
    return flipped_poly.tolist()


def horizontal_flip_rle(rle, height, width):
    import pycocotools.mask as mask_util
    if 'counts' in rle and type(rle['counts']) == list:
        rle = mask_util.frPyObjects(rle, height, width)
    mask = mask_util.decode(rle)
    mask = mask[:, ::-1]
    rle = mask_util.encode(np.array(mask, order='F', dtype=np.uint8))
    return rle


def vertical_flip_poly(poly, height):
    flipped_poly = np.array(poly)
    flipped_poly[1::2] = height - np.array(poly[1::2])
    return flipped_poly.tolist()


def vertical_flip_rle(rle, height, width):
    import pycocotools.mask as mask_util
    if 'counts' in rle and type(rle['counts']) == list:
        rle = mask_util.frPyObjects(rle, height, width)
    mask = mask_util.decode(rle)
    mask = mask[::-1, :]
    rle = mask_util.encode(np.array(mask, order='F', dtype=np.uint8))
    return rle


def crop_poly(segm, crop):
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


def crop_rle(rle, crop, height, width):
    import pycocotools.mask as mask_util
    if 'counts' in rle and type(rle['counts']) == list:
        rle = mask_util.frPyObjects(rle, height, width)
    mask = mask_util.decode(rle)
    mask = mask[crop[1]:crop[3], crop[0]:crop[2]]
    rle = mask_util.encode(np.array(mask, order='F', dtype=np.uint8))
    return rle


def expand_poly(poly, x, y):
    expanded_poly = np.array(poly)
    expanded_poly[0::2] += x
    expanded_poly[1::2] += y
    return expanded_poly.tolist()


def expand_rle(rle, x, y, height, width, h, w):
    import pycocotools.mask as mask_util
    if 'counts' in rle and type(rle['counts']) == list:
        rle = mask_util.frPyObjects(rle, height, width)
    mask = mask_util.decode(rle)
    expanded_mask = np.full((h, w), 0).astype(mask.dtype)
    expanded_mask[y:y + height, x:x + width] = mask
    rle = mask_util.encode(np.array(expanded_mask, order='F', dtype=np.uint8))
    return rle


def resize_poly(poly, im_scale_x, im_scale_y):
    resized_poly = np.array(poly, dtype=np.float32)
    resized_poly[0::2] *= im_scale_x
    resized_poly[1::2] *= im_scale_y
    return resized_poly.tolist()


def resize_rle(rle, im_h, im_w, im_scale_x, im_scale_y, interp):
    import pycocotools.mask as mask_util
    if 'counts' in rle and type(rle['counts']) == list:
        rle = mask_util.frPyObjects(rle, im_h, im_w)
    mask = mask_util.decode(rle)
    mask = cv2.resize(
        mask, None, None, fx=im_scale_x, fy=im_scale_y, interpolation=interp)
    rle = mask_util.encode(np.array(mask, order='F', dtype=np.uint8))
    return rle
