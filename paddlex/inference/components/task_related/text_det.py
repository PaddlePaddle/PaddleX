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
from ..base import BaseComponent
from .seal_det_warp import AutoRectifier


__all__ = ["DetResizeForTest", "NormalizeImage", "DBPostProcess", "CropByPolys"]


class DetResizeForTest(BaseComponent):
    """DetResizeForTest"""

    INPUT_KEYS = ["img"]
    OUTPUT_KEYS = ["img", "img_shape"]
    DEAULT_INPUTS = {"img": "img"}
    DEAULT_OUTPUTS = {"img": "img", "img_shape": "img_shape"}

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

    def apply(self, img):
        """apply"""
        src_h, src_w, _ = img.shape
        if sum([src_h, src_w]) < 64:
            img = self.image_padding(img)

        if self.resize_type == 0:
            # img, shape = self.resize_image_type0(img)
            img, [ratio_h, ratio_w] = self.resize_image_type0(img)
        elif self.resize_type == 2:
            img, [ratio_h, ratio_w] = self.resize_image_type2(img)
        else:
            # img, shape = self.resize_image_type1(img)
            img, [ratio_h, ratio_w] = self.resize_image_type1(img)
        return {"img": img, "img_shape": np.array([src_h, src_w, ratio_h, ratio_w])}

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

    def resize_image_type0(self, img):
        """
        resize image to a size multiple of 32 which is required by the network
        args:
            img(array): array with shape [h, w, c]
        return(tuple):
            img, (ratio_h, ratio_w)
        """
        limit_side_len = self.limit_side_len
        h, w, c = img.shape

        # limit the max side
        if self.limit_type == "max":
            if max(h, w) > limit_side_len:
                if h > w:
                    ratio = float(limit_side_len) / h
                else:
                    ratio = float(limit_side_len) / w
            else:
                ratio = 1.0
        elif self.limit_type == "min":
            if min(h, w) < limit_side_len:
                if h < w:
                    ratio = float(limit_side_len) / h
                else:
                    ratio = float(limit_side_len) / w
            else:
                ratio = 1.0
        elif self.limit_type == "resize_long":
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


class NormalizeImage(BaseComponent):
    """normalize image such as substract mean, divide std"""

    INPUT_KEYS = ["img"]
    OUTPUT_KEYS = ["img"]
    DEAULT_INPUTS = {"img": "img"}
    DEAULT_OUTPUTS = {"img": "img"}

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

    def apply(self, img):
        """apply"""
        from PIL import Image

        if isinstance(img, Image.Image):
            img = np.array(img)
        assert isinstance(img, np.ndarray), "invalid input 'img' in NormalizeImage"
        img = (img.astype("float32") * self.scale - self.mean) / self.std
        return {"img": img}


class DBPostProcess(BaseComponent):
    """
    The post process for Differentiable Binarization (DB).
    """

    INPUT_KEYS = ["pred", "img_shape"]
    OUTPUT_KEYS = ["dt_polys", "dt_scores"]
    DEAULT_INPUTS = {"pred": "pred", "img_shape": "img_shape"}
    DEAULT_OUTPUTS = {"dt_polys": "dt_polys", "dt_scores": "dt_scores"}

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

        self.dilation_kernel = None if not use_dilation else np.array([[1, 1], [1, 1]])

    def polygons_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
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
            if self.box_thresh > score:
                continue

            if points.shape[0] > 2:
                box = self.unclip(points, self.unclip_ratio)
                if len(box) > 1:
                    continue
            else:
                continue
            box = box.reshape(-1, 2)

            _, sside = self.get_mini_boxes(box.reshape((-1, 1, 2)))
            if sside < self.min_size + 2:
                continue

            box = np.array(box)
            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height
            )
            boxes.append(box.tolist())
            scores.append(score)
        return boxes, scores

    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
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
            if self.box_thresh > score:
                continue

            box = self.unclip(points, self.unclip_ratio).reshape(-1, 1, 2)
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
        expanded = np.array(offset.Execute(distance))
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
        """box_score_slow: use polyon mean score as the mean score"""
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

    def apply(self, pred, img_shape):
        """apply"""
        pred = pred[0][0, :, :]
        segmentation = pred > self.thresh

        src_h, src_w, ratio_h, ratio_w = img_shape
        if self.dilation_kernel is not None:
            mask = cv2.dilate(
                np.array(segmentation).astype(np.uint8),
                self.dilation_kernel,
            )
        else:
            mask = segmentation
        if self.box_type == "poly":
            boxes, scores = self.polygons_from_bitmap(pred, mask, src_w, src_h)
        elif self.box_type == "quad":
            boxes, scores = self.boxes_from_bitmap(pred, mask, src_w, src_h)
        else:
            raise ValueError("box_type can only be one of ['quad', 'poly']")

        return {"dt_polys": boxes, "dt_scores": scores}


class CropByPolys(BaseComponent):
    """Crop Image by Polys"""

    INPUT_KEYS = ["img_path", "dt_polys"]
    OUTPUT_KEYS = ["img"]
    DEAULT_INPUTS = {"img_path": "img_path", "dt_polys": "dt_polys"}
    DEAULT_OUTPUTS = {"img": "img"}

    def __init__(self, det_box_type="quad"):
        super().__init__()
        self.det_box_type = det_box_type
        self._reader = ImageReader(backend="opencv")

    def apply(self, img_path, dt_polys):
        """apply"""
        img = self._reader.read(img_path)
        
        # TODO
        # dt_boxes = self.sorted_boxes(data[K.DT_POLYS])
        if self.det_box_type == "quad":
            dt_boxes = np.array(dt_polys)
            output_list = []
            for bno in range(len(dt_boxes)):
                dt_boxes = self.sorted_boxes(dt_polys)
                tmp_box = copy.deepcopy(dt_boxes[bno])
                img_crop = self.get_rotate_crop_image(img, tmp_box)
                output_list.append(
                {"img": img_crop, "img_size": [img_crop.shape[1], img_crop.shape[0]]}
            )
        elif self.det_box_type == "poly":
            output_list = []
            dt_boxes = dt_polys
            for bno in range(len(dt_boxes)):
                tmp_box = copy.deepcopy(dt_boxes[bno])
                img_crop = self.get_poly_rect_crop(img.copy(), tmp_box)
                output_list.append(
                {"img": img_crop, "img_size": [img_crop.shape[1], img_crop.shape[0]]}
                )
        else:
            raise NotImplementedError

        return output_list

    def sorted_boxes(self, dt_boxes):
        """
        Sort text boxes in order from top to bottom, left to right
        args:
            dt_boxes(array):detected text boxes with shape [4, 2]
        return:
            sorted boxes(array) with shape [4, 2]
        """
        dt_boxes = np.array(dt_boxes)
        num_boxes = dt_boxes.shape[0]
        sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
        _boxes = list(sorted_boxes)

        for i in range(num_boxes - 1):
            for j in range(i, -1, -1):
                if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and (
                    _boxes[j + 1][0][0] < _boxes[j][0][0]
                ):
                    tmp = _boxes[j]
                    _boxes[j] = _boxes[j + 1]
                    _boxes[j + 1] = tmp
                else:
                    break
        return _boxes

    def get_rotate_crop_image(self, img, points):
        """
        img_height, img_width = img.shape[0:2]
        left = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 0]))
        top = int(np.min(points[:, 1]))
        bottom = int(np.max(points[:, 1]))
        img_crop = img[top:bottom, left:right, :].copy()
        points[:, 0] = points[:, 0] - left
        points[:, 1] = points[:, 1] - top
        """
        assert len(points) == 4, "shape of points must be 4*2"
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3]),
            )
        )
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2]),
            )
        )
        pts_std = np.float32(
            [
                [0, 0],
                [img_crop_width, 0],
                [img_crop_width, img_crop_height],
                [0, img_crop_height],
            ]
        )
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M,
            (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC,
        )
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img

    def reorder_poly_edge(self, points):
        """Get the respective points composing head edge, tail edge, top
        sideline and bottom sideline.

        Args:
            points (ndarray): The points composing a text polygon.

        Returns:
            head_edge (ndarray): The two points composing the head edge of text
                polygon.
            tail_edge (ndarray): The two points composing the tail edge of text
                polygon.
            top_sideline (ndarray): The points composing top curved sideline of
                text polygon.
            bot_sideline (ndarray): The points composing bottom curved sideline
                of text polygon.
        """

        assert points.ndim == 2
        assert points.shape[0] >= 4
        assert points.shape[1] == 2

        orientation_thr=2.0             # 一个经验超参数

        head_inds, tail_inds = self.find_head_tail(points, orientation_thr)
        head_edge, tail_edge = points[head_inds], points[tail_inds]


        pad_points = np.vstack([points, points])
        if tail_inds[1] < 1:
            tail_inds[1] = len(points)
        sideline1 = pad_points[head_inds[1]:tail_inds[1]]
        sideline2 = pad_points[tail_inds[1]:(head_inds[1] + len(points))]
        return head_edge, tail_edge, sideline1, sideline2

    def vector_slope(self, vec):
        assert len(vec) == 2
        return abs(vec[1] / (vec[0] + 1e-8)) 

    def find_head_tail(self, points, orientation_thr):
        """Find the head edge and tail edge of a text polygon.

        Args:
            points (ndarray): The points composing a text polygon.
            orientation_thr (float): The threshold for distinguishing between
                head edge and tail edge among the horizontal and vertical edges
                of a quadrangle.

        Returns:
            head_inds (list): The indexes of two points composing head edge.
            tail_inds (list): The indexes of two points composing tail edge.
        """

        assert points.ndim == 2
        assert points.shape[0] >= 4
        assert points.shape[1] == 2
        assert isinstance(orientation_thr, float)

        if len(points) > 4:
            pad_points = np.vstack([points, points[0]])
            edge_vec = pad_points[1:] - pad_points[:-1]

            theta_sum = []
            adjacent_vec_theta = []
            for i, edge_vec1 in enumerate(edge_vec):
                adjacent_ind = [x % len(edge_vec) for x in [i - 1, i + 1]]
                adjacent_edge_vec = edge_vec[adjacent_ind]
                temp_theta_sum = np.sum(
                    self.vector_angle(edge_vec1, adjacent_edge_vec))
                temp_adjacent_theta = self.vector_angle(adjacent_edge_vec[0],
                                                        adjacent_edge_vec[1])
                theta_sum.append(temp_theta_sum)
                adjacent_vec_theta.append(temp_adjacent_theta)
            theta_sum_score = np.array(theta_sum) / np.pi
            adjacent_theta_score = np.array(adjacent_vec_theta) / np.pi
            poly_center = np.mean(points, axis=0)
            edge_dist = np.maximum(
                norm(
                    pad_points[1:] - poly_center, axis=-1),
                norm(
                    pad_points[:-1] - poly_center, axis=-1))
            dist_score = edge_dist / np.max(edge_dist)
            position_score = np.zeros(len(edge_vec))
            score = 0.5 * theta_sum_score + 0.15 * adjacent_theta_score
            score += 0.35 * dist_score
            if len(points) % 2 == 0:
                position_score[(len(score) // 2 - 1)] += 1
                position_score[-1] += 1
            score += 0.1 * position_score
            pad_score = np.concatenate([score, score])
            score_matrix = np.zeros((len(score), len(score) - 3))
            x = np.arange(len(score) - 3) / float(len(score) - 4)
            gaussian = 1. / (np.sqrt(2. * np.pi) * 0.5) * np.exp(-np.power(
                (x - 0.5) / 0.5, 2.) / 2)
            gaussian = gaussian / np.max(gaussian)
            for i in range(len(score)):
                score_matrix[i, :] = score[i] + pad_score[(i + 2):(i + len(
                    score) - 1)] * gaussian * 0.3

            head_start, tail_increment = np.unravel_index(score_matrix.argmax(),
                                                            score_matrix.shape)
            tail_start = (head_start + tail_increment + 2) % len(points)
            head_end = (head_start + 1) % len(points)
            tail_end = (tail_start + 1) % len(points)

            if head_end > tail_end:
                head_start, tail_start = tail_start, head_start
                head_end, tail_end = tail_end, head_end
            head_inds = [head_start, head_end]
            tail_inds = [tail_start, tail_end]
        else:
            if vector_slope(points[1] - points[0]) + vector_slope(points[
                    3] - points[2]) < vector_slope(points[2] - points[
                        1]) + vector_slope(points[0] - points[3]):
                horizontal_edge_inds = [[0, 1], [2, 3]]
                vertical_edge_inds = [[3, 0], [1, 2]]
            else:
                horizontal_edge_inds = [[3, 0], [1, 2]]
                vertical_edge_inds = [[0, 1], [2, 3]]

            vertical_len_sum = norm(points[vertical_edge_inds[0][0]] - points[
                vertical_edge_inds[0][1]]) + norm(points[vertical_edge_inds[1][
                    0]] - points[vertical_edge_inds[1][1]])
            horizontal_len_sum = norm(points[horizontal_edge_inds[0][
                0]] - points[horizontal_edge_inds[0][1]]) + norm(points[
                    horizontal_edge_inds[1][0]] - points[horizontal_edge_inds[1]
                                                            [1]])

            if vertical_len_sum > horizontal_len_sum * orientation_thr:
                head_inds = horizontal_edge_inds[0]
                tail_inds = horizontal_edge_inds[1]
            else:
                head_inds = vertical_edge_inds[0]
                tail_inds = vertical_edge_inds[1]

        return head_inds, tail_inds

    def vector_angle(self, vec1, vec2):
        if vec1.ndim > 1:
            unit_vec1 = vec1 / (norm(vec1, axis=-1) + 1e-8).reshape((-1, 1))
        else:
            unit_vec1 = vec1 / (norm(vec1, axis=-1) + 1e-8)
        if vec2.ndim > 1:
            unit_vec2 = vec2 / (norm(vec2, axis=-1) + 1e-8).reshape((-1, 1))
        else:
            unit_vec2 = vec2 / (norm(vec2, axis=-1) + 1e-8)
        return np.arccos(np.clip(np.sum(unit_vec1 * unit_vec2, axis=-1), -1.0, 1.0))


    def get_minarea_rect(self, img, points):
        bounding_box = cv2.minAreaRect(points)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_a, index_b, index_c, index_d = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_a = 0
            index_d = 1
        else:
            index_a = 1
            index_d = 0
        if points[3][1] > points[2][1]:
            index_b = 2
            index_c = 3
        else:
            index_b = 3
            index_c = 2

        box = [points[index_a], points[index_b], points[index_c], points[index_d]]
        crop_img = self.get_rotate_crop_image(img, np.array(box))
        return crop_img, box

    def sample_points_on_bbox_bp(self, line, n=50):
        """Resample n points on a line.

        Args:
            line (ndarray): The points composing a line.
            n (int): The resampled points number.

        Returns:
            resampled_line (ndarray): The points composing the resampled line.
        """
        from numpy.linalg import norm
        # 断言检查输入参数的有效性
        assert line.ndim == 2
        assert line.shape[0] >= 2
        assert line.shape[1] == 2
        assert isinstance(n, int)
        assert n > 0

        length_list = [
            norm(line[i + 1] - line[i]) for i in range(len(line) - 1)
        ]
        total_length = sum(length_list)
        length_cumsum = np.cumsum([0.0] + length_list)
        delta_length = total_length / (float(n) + 1e-8)
        current_edge_ind = 0
        resampled_line = [line[0]]

        for i in range(1, n):
            current_line_len = i * delta_length
            while current_edge_ind + 1 < len(
                    length_cumsum) and current_line_len >= length_cumsum[
                        current_edge_ind + 1]:
                current_edge_ind += 1
            current_edge_end_shift = current_line_len - length_cumsum[
                current_edge_ind]
            if current_edge_ind >= len(length_list):
                break
            end_shift_ratio = current_edge_end_shift / length_list[
                current_edge_ind]
            current_point = line[current_edge_ind] + (line[current_edge_ind + 1]
                                                    - line[current_edge_ind]
                                                    ) * end_shift_ratio
            resampled_line.append(current_point)
        resampled_line.append(line[-1])
        resampled_line = np.array(resampled_line)
        return resampled_line

    def sample_points_on_bbox(self, line, n=50):
        """Resample n points on a line.

        Args:
            line (ndarray): The points composing a line.
            n (int): The resampled points number.

        Returns:
            resampled_line (ndarray): The points composing the resampled line.
        """
        assert line.ndim == 2
        assert line.shape[0] >= 2
        assert line.shape[1] == 2
        assert isinstance(n, int)
        assert n > 0

        length_list = [
            norm(line[i + 1] - line[i]) for i in range(len(line) - 1)
        ]
        total_length = sum(length_list)
        mean_length = total_length / (len(length_list) + 1e-8)
        group = [[0]]
        for i in range(len(length_list)):
            point_id = i+1
            if length_list[i] < 0.9 * mean_length:
                for g in group:
                    if i in g:
                        g.append(point_id)
                        break
            else:
                g = [point_id]
                group.append(g)

        top_tail_len = norm(line[0] - line[-1])
        if top_tail_len < 0.9 * mean_length:
            group[0].extend(g)
            group.remove(g)
        mean_positions = []  
        for indices in group:  
            x_sum = 0  
            y_sum = 0  
            for index in indices:  
                x, y = line[index]  
                x_sum += x  
                y_sum += y  
            num_points = len(indices)  
            mean_x = x_sum / num_points  
            mean_y = y_sum / num_points  
            mean_positions.append((mean_x, mean_y)) 
        resampled_line = np.array(mean_positions)
        return resampled_line

    def get_poly_rect_crop(self, img, points):
        '''
            修改该函数，实现使用polygon，对不规则、弯曲文本的矫正以及crop
            args： img: 图片 ndarrary格式
            points： polygon格式的多点坐标 N*2 shape， ndarray格式
            return： 矫正后的图片 ndarray格式
        '''
        points = np.array(points).astype(np.int32).reshape(-1, 2)
        temp_crop_img, temp_box = self.get_minarea_rect(img, points)
        # 计算最小外接矩形与polygon的IoU
        def get_union(pD, pG):
            return Polygon(pD).union(Polygon(pG)).area

        def get_intersection_over_union(pD, pG):
            return get_intersection(pD, pG) / (get_union(pD, pG)+ 1e-10)

        def get_intersection(pD, pG):
            return Polygon(pD).intersection(Polygon(pG)).area

        cal_IoU = get_intersection_over_union(points, temp_box)

        if cal_IoU >= 0.7:
            points = self.sample_points_on_bbox_bp(points, 31)
            return temp_crop_img

        points_sample = self.sample_points_on_bbox(points)
        points_sample = points_sample.astype(np.int32)
        head_edge, tail_edge, top_line, bot_line = self.reorder_poly_edge(points_sample)

        resample_top_line = self.sample_points_on_bbox_bp(top_line, 15)
        resample_bot_line = self.sample_points_on_bbox_bp(bot_line, 15)

        sideline_mean_shift = np.mean(
            resample_top_line, axis=0) - np.mean(
                resample_bot_line, axis=0)
        if sideline_mean_shift[1] > 0:
            resample_bot_line, resample_top_line = resample_top_line, resample_bot_line
        rectifier = AutoRectifier()
        new_points = np.concatenate([resample_top_line, resample_bot_line])
        new_points_list = list(new_points.astype(np.float32).reshape(1, -1).tolist())

        if len(img.shape) == 2:
            img = np.stack((img,)*3, axis=-1)
        img_crop, image = rectifier.run(img, new_points_list, mode='homography')
        return img_crop[0]
