# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import copy
from typing import List, Tuple

import numpy as np
from numpy.linalg import norm

from .....utils.deps import class_requires_deps, is_dep_available
from .base_operator import BaseOperator
from .seal_det_warp import AutoRectifier

if is_dep_available("opencv-contrib-python"):
    import cv2
if is_dep_available("shapely"):
    from shapely.geometry import Polygon


class CropByBoxes(BaseOperator):
    """Crop Image by Boxes"""

    entities = "CropByBoxes"

    def __init__(self) -> None:
        """Initializes the class."""
        super().__init__()

    def __call__(self, img: np.ndarray, boxes: List[dict]) -> List[dict]:
        """
        Process the input image and bounding boxes to produce a list of cropped images
        with their corresponding bounding box coordinates and labels.

        Args:
            img (np.ndarray): The input image as a NumPy array.
            boxes (list[dict]): A list of dictionaries, each containing bounding box
                information including 'cls_id' (class ID), 'coordinate' (bounding box
                coordinates as a list or tuple, left, top, right, bottom),
                and optionally 'label' (label text).

        Returns:
            list[dict]: A list of dictionaries, each containing a cropped image ('img'),
                the original bounding box coordinates ('box'), and the label ('label').
        """
        output_list = []
        for bbox_info in boxes:
            label_id = bbox_info["cls_id"]
            box = bbox_info["coordinate"]
            label = bbox_info.get("label", label_id)
            xmin, ymin, xmax, ymax = [int(i) for i in box]
            img_crop = img[ymin:ymax, xmin:xmax].copy()
            output_list.append({"img": img_crop, "box": box, "label": label})
        return output_list


@class_requires_deps("opencv-contrib-python", "shapely")
class CropByPolys(BaseOperator):
    """Crop Image by Polys"""

    entities = "CropByPolys"

    def __init__(self, det_box_type: str = "quad") -> None:
        """
        Initializes the operator with a default detection box type.

        Args:
            det_box_type (str, optional): The type of detection box, quad or poly. Defaults to "quad".
        """
        super().__init__()
        self.det_box_type = det_box_type

    def __call__(self, img: np.ndarray, dt_polys: List[list]) -> List[dict]:
        """
        Call method to crop images based on detection boxes.

        Args:
            img (nd.ndarray): The input image.
            dt_polys (list[list]): List of detection polygons.

        Returns:
            list[dict]: A list of dictionaries containing cropped images and their sizes.

        Raises:
            NotImplementedError: If det_box_type is not 'quad' or 'poly'.
        """

        if self.det_box_type == "quad":
            dt_boxes = np.array(dt_polys)
            output_list = []
            for bno in range(len(dt_boxes)):
                tmp_box = copy.deepcopy(dt_boxes[bno])
                img_crop = self.get_minarea_rect_crop(img, tmp_box)
                output_list.append(img_crop)
        elif self.det_box_type == "poly":
            output_list = []
            dt_boxes = dt_polys
            for bno in range(len(dt_boxes)):
                tmp_box = copy.deepcopy(dt_boxes[bno])
                img_crop = self.get_poly_rect_crop(img.copy(), tmp_box)
                output_list.append(img_crop)
        else:
            raise NotImplementedError

        return output_list

    def get_minarea_rect_crop(self, img: np.ndarray, points: np.ndarray) -> np.ndarray:
        """
        Get the minimum area rectangle crop from the given image and points.

        Args:
            img (np.ndarray): The input image.
            points (np.ndarray): A list of points defining the shape to be cropped.

        Returns:
            np.ndarray: The cropped image with the minimum area rectangle.
        """
        bounding_box = cv2.minAreaRect(np.array(points).astype(np.int32))
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
        return crop_img

    def get_rotate_crop_image(self, img: np.ndarray, points: list) -> np.ndarray:
        """
        Crop and rotate the input image based on the given four points to form a perspective-transformed image.

        Args:
            img (np.ndarray): The input image array.
            points (list): A list of four 2D points defining the crop region in the image.

        Returns:
            np.ndarray: The transformed image array.
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

    def reorder_poly_edge(
        self, points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

        orientation_thr = 2.0  # 一个经验超参数

        head_inds, tail_inds = self.find_head_tail(points, orientation_thr)
        head_edge, tail_edge = points[head_inds], points[tail_inds]

        pad_points = np.vstack([points, points])
        if tail_inds[1] < 1:
            tail_inds[1] = len(points)
        sideline1 = pad_points[head_inds[1] : tail_inds[1]]
        sideline2 = pad_points[tail_inds[1] : (head_inds[1] + len(points))]
        return head_edge, tail_edge, sideline1, sideline2

    def vector_slope(self, vec: list) -> float:
        """
        Calculate the slope of a vector in 2D space.

        Args:
            vec (list): A list of two elements representing the coordinates of the vector.

        Returns:
            float: The slope of the vector.

        Raises:
            AssertionError: If the length of the vector is not equal to 2.
        """
        assert len(vec) == 2
        return abs(vec[1] / (vec[0] + 1e-8))

    def find_head_tail(
        self, points: np.ndarray, orientation_thr: float
    ) -> Tuple[list, list]:
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
                temp_theta_sum = np.sum(self.vector_angle(edge_vec1, adjacent_edge_vec))
                temp_adjacent_theta = self.vector_angle(
                    adjacent_edge_vec[0], adjacent_edge_vec[1]
                )
                theta_sum.append(temp_theta_sum)
                adjacent_vec_theta.append(temp_adjacent_theta)
            theta_sum_score = np.array(theta_sum) / np.pi
            adjacent_theta_score = np.array(adjacent_vec_theta) / np.pi
            poly_center = np.mean(points, axis=0)
            edge_dist = np.maximum(
                norm(pad_points[1:] - poly_center, axis=-1),
                norm(pad_points[:-1] - poly_center, axis=-1),
            )
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
            gaussian = (
                1.0
                / (np.sqrt(2.0 * np.pi) * 0.5)
                * np.exp(-np.power((x - 0.5) / 0.5, 2.0) / 2)
            )
            gaussian = gaussian / np.max(gaussian)
            for i in range(len(score)):
                score_matrix[i, :] = (
                    score[i]
                    + pad_score[(i + 2) : (i + len(score) - 1)] * gaussian * 0.3
                )

            head_start, tail_increment = np.unravel_index(
                score_matrix.argmax(), score_matrix.shape
            )
            tail_start = (head_start + tail_increment + 2) % len(points)
            head_end = (head_start + 1) % len(points)
            tail_end = (tail_start + 1) % len(points)

            if head_end > tail_end:
                head_start, tail_start = tail_start, head_start
                head_end, tail_end = tail_end, head_end
            head_inds = [head_start, head_end]
            tail_inds = [tail_start, tail_end]
        else:
            if self.vector_slope(points[1] - points[0]) + self.vector_slope(
                points[3] - points[2]
            ) < self.vector_slope(points[2] - points[1]) + self.vector_slope(
                points[0] - points[3]
            ):
                horizontal_edge_inds = [[0, 1], [2, 3]]
                vertical_edge_inds = [[3, 0], [1, 2]]
            else:
                horizontal_edge_inds = [[3, 0], [1, 2]]
                vertical_edge_inds = [[0, 1], [2, 3]]

            vertical_len_sum = norm(
                points[vertical_edge_inds[0][0]] - points[vertical_edge_inds[0][1]]
            ) + norm(
                points[vertical_edge_inds[1][0]] - points[vertical_edge_inds[1][1]]
            )
            horizontal_len_sum = norm(
                points[horizontal_edge_inds[0][0]] - points[horizontal_edge_inds[0][1]]
            ) + norm(
                points[horizontal_edge_inds[1][0]] - points[horizontal_edge_inds[1][1]]
            )

            if vertical_len_sum > horizontal_len_sum * orientation_thr:
                head_inds = horizontal_edge_inds[0]
                tail_inds = horizontal_edge_inds[1]
            else:
                head_inds = vertical_edge_inds[0]
                tail_inds = vertical_edge_inds[1]

        return head_inds, tail_inds

    def vector_angle(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate the angle between two vectors.

        Args:
            vec1 (ndarray): The first vector.
            vec2 (ndarray): The second vector.

        Returns:
            float: The angle between the two vectors in radians.
        """
        if vec1.ndim > 1:
            unit_vec1 = vec1 / (norm(vec1, axis=-1) + 1e-8).reshape((-1, 1))
        else:
            unit_vec1 = vec1 / (norm(vec1, axis=-1) + 1e-8)
        if vec2.ndim > 1:
            unit_vec2 = vec2 / (norm(vec2, axis=-1) + 1e-8).reshape((-1, 1))
        else:
            unit_vec2 = vec2 / (norm(vec2, axis=-1) + 1e-8)
        return np.arccos(np.clip(np.sum(unit_vec1 * unit_vec2, axis=-1), -1.0, 1.0))

    def get_minarea_rect(
        self, img: np.ndarray, points: np.ndarray
    ) -> Tuple[np.ndarray, list]:
        """
        Get the minimum area rectangle for the given points and crop the image accordingly.

        Args:
            img (np.ndarray): The input image.
            points (np.ndarray): The points to compute the minimum area rectangle for.

        Returns:
            tuple[np.ndarray, list]: The cropped image,
            and the list of points in the order of the bounding box.
        """
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

        length_list = [norm(line[i + 1] - line[i]) for i in range(len(line) - 1)]
        total_length = sum(length_list)
        length_cumsum = np.cumsum([0.0] + length_list)
        delta_length = total_length / (float(n) + 1e-8)
        current_edge_ind = 0
        resampled_line = [line[0]]

        for i in range(1, n):
            current_line_len = i * delta_length
            while (
                current_edge_ind + 1 < len(length_cumsum)
                and current_line_len >= length_cumsum[current_edge_ind + 1]
            ):
                current_edge_ind += 1
            current_edge_end_shift = current_line_len - length_cumsum[current_edge_ind]
            if current_edge_ind >= len(length_list):
                break
            end_shift_ratio = current_edge_end_shift / length_list[current_edge_ind]
            current_point = (
                line[current_edge_ind]
                + (line[current_edge_ind + 1] - line[current_edge_ind])
                * end_shift_ratio
            )
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

        length_list = [norm(line[i + 1] - line[i]) for i in range(len(line) - 1)]
        total_length = sum(length_list)
        mean_length = total_length / (len(length_list) + 1e-8)
        group = [[0]]
        for i in range(len(length_list)):
            point_id = i + 1
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
        """
        修改该函数，实现使用polygon，对不规则、弯曲文本的矫正以及crop
        args： img: 图片 ndarrary格式
        points： polygon格式的多点坐标 N*2 shape， ndarray格式
        return： 矫正后的图片 ndarray格式
        """
        points = np.array(points).astype(np.int32).reshape(-1, 2)
        temp_crop_img, temp_box = self.get_minarea_rect(img, points)

        # 计算最小外接矩形与polygon的IoU
        def get_union(pD, pG):
            return Polygon(pD).union(Polygon(pG)).area

        def get_intersection_over_union(pD, pG):
            return get_intersection(pD, pG) / (get_union(pD, pG) + 1e-10)

        def get_intersection(pD, pG):
            return Polygon(pD).intersection(Polygon(pG)).area

        if not Polygon(points).is_valid:
            return temp_crop_img
        cal_IoU = get_intersection_over_union(points, temp_box)

        if cal_IoU >= 0.7:
            points = self.sample_points_on_bbox_bp(points, 31)
            return temp_crop_img

        points_sample = self.sample_points_on_bbox(points)
        points_sample = points_sample.astype(np.int32)
        head_edge, tail_edge, top_line, bot_line = self.reorder_poly_edge(points_sample)

        resample_top_line = self.sample_points_on_bbox_bp(top_line, 15)
        resample_bot_line = self.sample_points_on_bbox_bp(bot_line, 15)

        sideline_mean_shift = np.mean(resample_top_line, axis=0) - np.mean(
            resample_bot_line, axis=0
        )
        if sideline_mean_shift[1] > 0:
            resample_bot_line, resample_top_line = resample_top_line, resample_bot_line
        rectifier = AutoRectifier()
        new_points = np.concatenate([resample_top_line, resample_bot_line])
        new_points_list = list(new_points.astype(np.float32).reshape(1, -1).tolist())

        if len(img.shape) == 2:
            img = np.stack((img,) * 3, axis=-1)
        img_crop, image = rectifier.run(img, new_points_list, mode="homography")
        return np.array(img_crop[0], dtype=np.uint8)
