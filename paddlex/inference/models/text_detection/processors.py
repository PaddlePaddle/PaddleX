# copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.
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

from typing import List, Tuple, Union
import os
import sys
import cv2
import copy
import math
import pyclipper
import numpy as np
from numpy.linalg import norm
from PIL import Image
from shapely.geometry import Polygon

from ...utils.io import ImageReader
from ....utils import logging
from ...utils.benchmark import benchmark


@benchmark.timeit
class DetResizeForTest:
    """DetResizeForTest"""

    def __init__(self, **kwargs):
        super().__init__()
        self.resize_type = 0
        self.keep_ratio = False
        if "image_shape" in kwargs:
            self.image_shape = kwargs["image_shape"]
            self.resize_type = 1
            if "keep_ratio" in kwargs:
                self.keep_ratio = kwargs["keep_ratio"]
        elif "limit_side_len" in kwargs:
            self.limit_side_len = kwargs["limit_side_len"]
            self.limit_type = kwargs.get("limit_type", "min")
        elif "resize_long" in kwargs:
            self.resize_type = 2
            self.resize_long = kwargs.get("resize_long", 960)
        else:
            self.limit_side_len = 736
            self.limit_type = "min"

    def __call__(
        self,
        imgs,
        limit_side_len: Union[int, None] = None,
        limit_type: Union[str, None] = None,
    ):
        """apply"""
        resize_imgs, img_shapes = [], []
        for ori_img in imgs:
            img, shape = self.resize(ori_img, limit_side_len, limit_type)
            resize_imgs.append(img)
            img_shapes.append(shape)
        return resize_imgs, img_shapes

    def resize(
        self, img, limit_side_len: Union[int, None], limit_type: Union[str, None]
    ):
        src_h, src_w, _ = img.shape
        if sum([src_h, src_w]) < 64:
            img = self.image_padding(img)

        if self.resize_type == 0:
            # img, shape = self.resize_image_type0(img)
            img, [ratio_h, ratio_w] = self.resize_image_type0(
                img, limit_side_len, limit_type
            )
        elif self.resize_type == 2:
            img, [ratio_h, ratio_w] = self.resize_image_type2(img)
        else:
            # img, shape = self.resize_image_type1(img)
            img, [ratio_h, ratio_w] = self.resize_image_type1(img)
        return img, np.array([src_h, src_w, ratio_h, ratio_w])

    def image_padding(self, im, value=0):
        """padding image"""
        h, w, c = im.shape
        im_pad = np.zeros((max(32, h), max(32, w), c), np.uint8) + value
        im_pad[:h, :w, :] = im
        return im_pad

    def resize_image_type1(self, img):
        """resize the image"""
        resize_h, resize_w = self.image_shape
        ori_h, ori_w = img.shape[:2]  # (h, w, c)
        if self.keep_ratio is True:
            resize_w = ori_w * resize_h / ori_h
            N = math.ceil(resize_w / 32)
            resize_w = N * 32
        ratio_h = float(resize_h) / ori_h
        ratio_w = float(resize_w) / ori_w
        img = cv2.resize(img, (int(resize_w), int(resize_h)))
        # return img, np.array([ori_h, ori_w])
        return img, [ratio_h, ratio_w]

    def resize_image_type0(
        self, img, limit_side_len: Union[int, None], limit_type: Union[str, None]
    ):
        """
        resize image to a size multiple of 32 which is required by the network
        args:
            img(array): array with shape [h, w, c]
        return(tuple):
            img, (ratio_h, ratio_w)
        """
        limit_side_len = limit_side_len or self.limit_side_len
        limit_type = limit_type or self.limit_type
        h, w, c = img.shape

        # limit the max side
        if limit_type == "max":
            if max(h, w) > limit_side_len:
                if h > w:
                    ratio = float(limit_side_len) / h
                else:
                    ratio = float(limit_side_len) / w
            else:
                ratio = 1.0
        elif limit_type == "min":
            if min(h, w) < limit_side_len:
                if h < w:
                    ratio = float(limit_side_len) / h
                else:
                    ratio = float(limit_side_len) / w
            else:
                ratio = 1.0
        elif limit_type == "resize_long":
            ratio = float(limit_side_len) / max(h, w)
        else:
            raise Exception("not support limit type, image ")
        resize_h = int(h * ratio)
        resize_w = int(w * ratio)

        resize_h = max(int(round(resize_h / 32) * 32), 32)
        resize_w = max(int(round(resize_w / 32) * 32), 32)

        try:
            if int(resize_w) <= 0 or int(resize_h) <= 0:
                return None, (None, None)
            img = cv2.resize(img, (int(resize_w), int(resize_h)))
        except:
            logging.info(img.shape, resize_w, resize_h)
            sys.exit(0)
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)
        return img, [ratio_h, ratio_w]

    def resize_image_type2(self, img):
        """resize image size"""
        h, w, _ = img.shape

        resize_w = w
        resize_h = h

        if resize_h > resize_w:
            ratio = float(self.resize_long) / resize_h
        else:
            ratio = float(self.resize_long) / resize_w

        resize_h = int(resize_h * ratio)
        resize_w = int(resize_w * ratio)

        max_stride = 128
        resize_h = (resize_h + max_stride - 1) // max_stride * max_stride
        resize_w = (resize_w + max_stride - 1) // max_stride * max_stride
        img = cv2.resize(img, (int(resize_w), int(resize_h)))
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)

        return img, [ratio_h, ratio_w]


@benchmark.timeit
class NormalizeImage:
    """normalize image such as substract mean, divide std"""

    def __init__(self, scale=None, mean=None, std=None, order="chw", **kwargs):
        super().__init__()
        if isinstance(scale, str):
            scale = eval(scale)
        self.scale = np.float32(scale if scale is not None else 1.0 / 255.0)
        mean = mean if mean is not None else [0.485, 0.456, 0.406]
        std = std if std is not None else [0.229, 0.224, 0.225]

        shape = (3, 1, 1) if order == "chw" else (1, 1, 3)
        self.mean = np.array(mean).reshape(shape).astype("float32")
        self.std = np.array(std).reshape(shape).astype("float32")

    def __call__(self, imgs):
        """apply"""

        def norm(img):
            return (img.astype("float32") * self.scale - self.mean) / self.std

        return [norm(img) for img in imgs]


@benchmark.timeit
class DBPostProcess:
    """
    The post process for Differentiable Binarization (DB).
    """

    def __init__(
        self,
        thresh=0.3,
        box_thresh=0.7,
        max_candidates=1000,
        unclip_ratio=2.0,
        use_dilation=False,
        score_mode="fast",
        box_type="quad",
        **kwargs
    ):
        super().__init__()
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = 3
        self.score_mode = score_mode
        self.box_type = box_type
        assert score_mode in [
            "slow",
            "fast",
        ], "Score mode must be in [slow, fast] but got: {}".format(score_mode)
        self.use_dilation = use_dilation

    def polygons_from_bitmap(
        self,
        pred,
        _bitmap,
        dest_width,
        dest_height,
        box_thresh,
        unclip_ratio,
    ):
        """_bitmap: single map with shape (1, H, W), whose values are binarized as {0, 1}"""

        bitmap = _bitmap
        height, width = bitmap.shape

        boxes = []
        scores = []

        contours, _ = cv2.findContours(
            (bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours[: self.max_candidates]:
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape((-1, 2))
            if points.shape[0] < 4:
                continue

            score = self.box_score_fast(pred, points.reshape(-1, 2))
            if box_thresh > score:
                continue

            if points.shape[0] > 2:
                box = self.unclip(points, unclip_ratio)
                if len(box) > 1:
                    continue
            else:
                continue
            box = box.reshape(-1, 2)

            if len(box) > 0:
                _, sside = self.get_mini_boxes(box.reshape((-1, 1, 2)))
                if sside < self.min_size + 2:
                    continue
            else:
                continue

            box = np.array(box)
            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height
            )
            boxes.append(box)
            scores.append(score)
        return boxes, scores

    def boxes_from_bitmap(
        self,
        pred,
        _bitmap,
        dest_width,
        dest_height,
        box_thresh,
        unclip_ratio,
    ):
        """_bitmap: single map with shape (1, H, W), whose values are binarized as {0, 1}"""

        bitmap = _bitmap
        height, width = bitmap.shape

        outs = cv2.findContours(
            (bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        if len(outs) == 3:
            img, contours, _ = outs[0], outs[1], outs[2]
        elif len(outs) == 2:
            contours, _ = outs[0], outs[1]

        num_contours = min(len(contours), self.max_candidates)

        boxes = []
        scores = []
        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            if self.score_mode == "fast":
                score = self.box_score_fast(pred, points.reshape(-1, 2))
            else:
                score = self.box_score_slow(pred, contour)
            if box_thresh > score:
                continue

            box = self.unclip(points, unclip_ratio).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)

            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height
            )
            boxes.append(box.astype(np.int16))
            scores.append(score)
        return np.array(boxes, dtype=np.int16), scores

    def unclip(self, box, unclip_ratio):
        """unclip"""
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        try:
            expanded = np.array(offset.Execute(distance))
        except ValueError:
            expanded = np.array(offset.Execute(distance)[0])
        return expanded

    def get_mini_boxes(self, contour):
        """get mini boxes"""
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box):
        """box_score_fast: use bbox mean score as the mean score"""
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype("int"), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype("int"), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype("int"), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype("int"), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin : ymax + 1, xmin : xmax + 1], mask)[0]

    def box_score_slow(self, bitmap, contour):
        """box_score_slow: use polygon mean score as the mean score"""
        h, w = bitmap.shape[:2]
        contour = contour.copy()
        contour = np.reshape(contour, (-1, 2))

        xmin = np.clip(np.min(contour[:, 0]), 0, w - 1)
        xmax = np.clip(np.max(contour[:, 0]), 0, w - 1)
        ymin = np.clip(np.min(contour[:, 1]), 0, h - 1)
        ymax = np.clip(np.max(contour[:, 1]), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)

        contour[:, 0] = contour[:, 0] - xmin
        contour[:, 1] = contour[:, 1] - ymin

        cv2.fillPoly(mask, contour.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin : ymax + 1, xmin : xmax + 1], mask)[0]

    def __call__(
        self,
        preds,
        img_shapes,
        thresh: Union[float, None] = None,
        box_thresh: Union[float, None] = None,
        unclip_ratio: Union[float, None] = None,
    ):
        """apply"""
        boxes, scores = [], []
        for pred, img_shape in zip(preds[0], img_shapes):
            box, score = self.process(
                pred,
                img_shape,
                thresh or self.thresh,
                box_thresh or self.box_thresh,
                unclip_ratio or self.unclip_ratio,
            )
            boxes.append(box)
            scores.append(score)
        return boxes, scores

    def process(
        self,
        pred,
        img_shape,
        thresh,
        box_thresh,
        unclip_ratio,
    ):
        pred = pred[0, :, :]
        segmentation = pred > thresh
        dilation_kernel = None if not self.use_dilation else np.array([[1, 1], [1, 1]])
        src_h, src_w, ratio_h, ratio_w = img_shape
        if dilation_kernel is not None:
            mask = cv2.dilate(
                np.array(segmentation).astype(np.uint8),
                dilation_kernel,
            )
        else:
            mask = segmentation
        if self.box_type == "poly":
            boxes, scores = self.polygons_from_bitmap(
                pred, mask, src_w, src_h, box_thresh, unclip_ratio
            )
        elif self.box_type == "quad":
            boxes, scores = self.boxes_from_bitmap(
                pred, mask, src_w, src_h, box_thresh, unclip_ratio
            )
        else:
            raise ValueError("box_type can only be one of ['quad', 'poly']")
        return boxes, scores
