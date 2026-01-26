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

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from numpy import ndarray

from ....utils import logging
from ....utils.deps import function_requires_deps, is_dep_available
from ...utils.benchmark import benchmark
from ..object_detection.processors import check_containment, nms

if is_dep_available("opencv-contrib-python"):
    import cv2

Boxes = List[dict]
Number = Union[int, float]


SKIP_ORDER_LABELS = [
    "figure_title",
    "vision_footnote",
    "image",
    "chart",
    "table",
    "header",
    "header_image",
    "footer",
    "footer_image",
    "footnote",
    "aside_text",
]


def is_convex(p_prev, p_curr, p_next):
    """
    Calculate if the polygon is convex.
    """
    v1 = p_curr - p_prev
    v2 = p_next - p_curr
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    return cross < 0


def angle_between_vectors(v1, v2):
    """
    Calculate the angle between two vectors.
    """

    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    dot_prod = np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)
    angle_rad = np.arccos(dot_prod)
    return np.degrees(angle_rad)


def calc_new_point(p_curr, v1, v2, distance=20):
    """
    Calculate the new point based on the direction of two vectors.
    """
    dir_vec = v1 / np.linalg.norm(v1) + v2 / np.linalg.norm(v2)
    dir_vec = dir_vec / np.linalg.norm(dir_vec)
    p_new = p_curr + dir_vec * distance
    return p_new


def extract_custom_vertices(
    polygon, max_allowed_dist, sharp_angle_thresh=45, max_dist_ratio=0.3
):
    poly = np.array(polygon)
    n = len(poly)
    max_allowed_dist *= max_dist_ratio

    point_info = []
    for i in range(n):
        p_prev, p_curr, p_next = poly[(i - 1) % n], poly[i], poly[(i + 1) % n]
        v1, v2 = p_prev - p_curr, p_next - p_curr
        is_convex_point = is_convex(p_prev, p_curr, p_next)
        angle = angle_between_vectors(v1, v2)
        point_info.append(
            {
                "index": i,
                "is_convex": is_convex_point,
                "angle": angle,
                "v1": v1,
                "v2": v2,
            }
        )

    concave_indices = [i for i, info in enumerate(point_info) if not info["is_convex"]]
    preserve_concave = set()

    if concave_indices:
        groups = []
        current_group = [concave_indices[0]]

        for i in range(1, len(concave_indices)):
            if concave_indices[i] - concave_indices[i - 1] == 1 or (
                concave_indices[i - 1] == n - 1 and concave_indices[i] == 0
            ):
                current_group.append(concave_indices[i])
            else:
                if len(current_group) >= 2:
                    groups.extend(current_group)
                current_group = [concave_indices[i]]

        if len(current_group) >= 2:
            groups.extend(current_group)

        if (
            len(concave_indices) >= 2
            and concave_indices[0] == 0
            and concave_indices[-1] == n - 1
        ):
            if 0 in groups and n - 1 in groups:
                preserve_concave.update(groups)
        else:
            preserve_concave.update(groups)

    kept_points = [
        i
        for i, info in enumerate(point_info)
        if info["is_convex"] or (i in preserve_concave and info["angle"] >= 120)
    ]

    final_points = []
    for idx in range(len(kept_points)):
        current_idx = kept_points[idx]
        next_idx = kept_points[(idx + 1) % len(kept_points)]
        final_points.append(current_idx)

        dist = np.linalg.norm(poly[current_idx] - poly[next_idx])
        if dist > max_allowed_dist:
            intermediate = (
                list(range(current_idx + 1, next_idx))
                if next_idx > current_idx
                else list(range(current_idx + 1, n)) + list(range(0, next_idx))
            )

            if intermediate:
                num_needed = int(np.ceil(dist / max_allowed_dist)) - 1
                if len(intermediate) <= num_needed:
                    final_points.extend(intermediate)
                else:
                    step = len(intermediate) / num_needed
                    final_points.extend(
                        [intermediate[int(i * step)] for i in range(num_needed)]
                    )

    final_points = sorted(set(final_points))
    res = []

    for i in final_points:
        info = point_info[i]
        p_curr = poly[i]

        if info["is_convex"] and abs(info["angle"] - sharp_angle_thresh) < 1:
            v1_norm = info["v1"] / np.linalg.norm(info["v1"])
            v2_norm = info["v2"] / np.linalg.norm(info["v2"])
            dir_vec = v1_norm + v2_norm
            dir_vec /= np.linalg.norm(dir_vec)
            d = (np.linalg.norm(info["v1"]) + np.linalg.norm(info["v2"])) / 2
            res.append(tuple(p_curr + dir_vec * d))
        else:
            res.append(tuple(p_curr))

    return res


@function_requires_deps("opencv-contrib-python")
def mask2polygon(mask, max_allowed_dist, epsilon_ratio=0.004, extract_custom=True):
    """
    Postprocess mask by removing small noise.
    Args:
        mask (ndarray): The input mask of shape [H, W].
        epsilon_ratio (float): The ratio of epsilon.
    Returns:
        ndarray: The output mask after postprocessing.
    """
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not cnts:
        return None

    cnt = max(cnts, key=cv2.contourArea)
    epsilon = epsilon_ratio * cv2.arcLength(cnt, True)
    approx_cnt = cv2.approxPolyDP(cnt, epsilon, True)
    polygon_points = approx_cnt.squeeze()
    polygon_points = np.atleast_2d(polygon_points)
    if extract_custom:
        polygon_points = extract_custom_vertices(polygon_points, max_allowed_dist)

    return polygon_points


def extract_polygon_points_by_masks(boxes, masks, scale_ratio, layout_shape_mode):
    """
    修改后的提取函数：auto 模式下信任几何决策
    """
    scale_w, scale_h = scale_ratio[0] / 4, scale_ratio[1] / 4
    h_m, w_m = masks.shape[1:]
    polygon_points = []
    iou_threshold = 0.95

    max_box_w = max(boxes[:, 4] - boxes[:, 3])

    for i in range(len(boxes)):
        x_min, y_min, x_max, y_max = boxes[i, 2:6].astype(np.int32)
        box_w, box_h = x_max - x_min, y_max - y_min

        # default rect
        rect = np.array(
            [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
            dtype=np.float32,
        )

        if box_w <= 0 or box_h <= 0:
            polygon_points.append(rect)
            continue

        # crop mask
        x_s = np.clip(
            [int(round(x_min * scale_w)), int(round(x_max * scale_w))], 0, w_m
        )
        y_s = np.clip(
            [int(round(y_min * scale_h)), int(round(y_max * scale_h))], 0, h_m
        )

        cropped = masks[i, y_s[0] : y_s[1], x_s[0] : x_s[1]]
        if cropped.size == 0 or np.sum(cropped) == 0:
            polygon_points.append(rect)
            continue

        if layout_shape_mode == "rect":
            polygon_points.append(rect)
            continue

        # resize mask to match box size
        resized_mask = cv2.resize(
            cropped.astype(np.uint8), (box_w, box_h), interpolation=cv2.INTER_NEAREST
        )

        if box_w > max_box_w * 0.6:
            max_allowed_dist = box_w
        else:
            max_allowed_dist = max_box_w

        polygon = mask2polygon(resized_mask, max_allowed_dist)
        if polygon is not None and len(polygon) < 4:
            polygon_points.append(rect)
            continue
        if polygon is not None and len(polygon) > 0:
            polygon = polygon + np.array([x_min, y_min])
        if layout_shape_mode == "poly":
            polygon_points.append(polygon)
        elif layout_shape_mode == "quad":
            # convert polygon to quadrilateral
            quad = convert_polygon_to_quad(polygon)
            polygon_points.append(quad if quad is not None else rect)
        elif layout_shape_mode == "auto":
            iou_threshold = 0.8

            rect_list = rect.tolist()
            quad = convert_polygon_to_quad(polygon)
            if quad is not None:
                quad_list = quad.tolist()

                iou_quad = calculate_polygon_overlap_ratio(
                    rect_list,
                    quad_list,
                    mode="union",
                )
                if iou_quad >= 0.95:
                    # if quad is very similar to rect, use rect instead
                    quad = rect

                poly_list = (
                    polygon.tolist() if isinstance(polygon, np.ndarray) else polygon
                )

                iou_quad = calculate_polygon_overlap_ratio(
                    poly_list, quad_list, mode="union"
                )

                pre_poly = polygon_points[-1] if len(polygon_points) > 0 else None
                iou_pre = 0
                if pre_poly is not None:
                    iou_pre = calculate_polygon_overlap_ratio(
                        pre_poly.tolist(),
                        rect_list,
                        mode="small",
                    )

                if iou_quad >= iou_threshold and iou_pre < 0.01:
                    # if quad is similar to polygon, use quad
                    polygon_points.append(quad)
                    continue

            # if all ious are less than threshold, use polygon
            polygon_points.append(polygon)
        else:
            raise ValueError(
                "layout_shape_mode must be one of ['rect', 'poly', 'quad', 'auto']"
            )

    return polygon_points


def convert_polygon_to_quad(polygon):
    """
    Convert polygon to minimum bounding rectangle (quad).
    Args:
        polygon (ndarray): The polygon points of shape [N, 2].
    Returns:
        quad (ndarray): The 4-point quad, clockwise from top-left, or None if invalid.
    """
    if polygon is None or len(polygon) < 3:
        return None

    points = np.array(polygon, dtype=np.float32)
    if len(points.shape) == 1:
        points = points.reshape(-1, 2)

    min_rect = cv2.minAreaRect(points)
    quad = cv2.boxPoints(min_rect)

    center = quad.mean(axis=0)
    angles = np.arctan2(quad[:, 1] - center[1], quad[:, 0] - center[0])
    sorted_indices = np.argsort(angles)
    quad = quad[sorted_indices]
    sums = quad[:, 0] + quad[:, 1]
    top_left_idx = np.argmin(sums)
    quad = np.roll(quad, -top_left_idx, axis=0)

    return quad


def restructured_boxes(
    boxes: ndarray,
    labels: List[str],
    img_size: Tuple[int, int],
    polygon_points: ndarray = None,
) -> Boxes:
    """
    Restructure the given bounding boxes and labels based on the image size.

    Args:
        boxes (ndarray): A 2D array of bounding boxes with each box represented as [cls_id, score, xmin, ymin, xmax, ymax].
        labels (List[str]): A list of class labels corresponding to the class ids.
        img_size (Tuple[int, int]): A tuple representing the width and height of the image.
        polygon_points (ndarray): A 2D array of polygon points with each point represented as [x, y].
    Returns:
        Boxes: A list of dictionaries, each containing 'cls_id', 'label', 'score', and 'coordinate' keys.
    """
    box_list = []
    w, h = img_size

    for idx, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = box[2:]
        xmin = int(max(0, xmin))
        ymin = int(max(0, ymin))
        xmax = int(min(w, xmax))
        ymax = int(min(h, ymax))
        if xmax <= xmin or ymax <= ymin:
            continue
        res = {
            "cls_id": int(box[0]),
            "label": labels[int(box[0])],
            "score": float(box[1]),
            "coordinate": [xmin, ymin, xmax, ymax],
            "order": idx + 1,
        }
        if polygon_points is not None:
            polygon_point = polygon_points[idx]
            if polygon_point is None:
                continue
            res["polygon_points"] = polygon_point
        box_list.append(res)

    return box_list


def unclip_boxes(boxes, unclip_ratio=None):
    """
    Expand bounding boxes from (x1, y1, x2, y2) format using an unclipping ratio.

    Parameters:
    - boxes: np.ndarray of shape (N, 4), where each row is (x1, y1, x2, y2).
    - unclip_ratio: tuple of (width_ratio, height_ratio), optional.

    Returns:
    - expanded_boxes: np.ndarray of shape (N, 4), where each row is (x1, y1, x2, y2).
    """
    if unclip_ratio is None:
        return boxes

    if isinstance(unclip_ratio, dict):
        expanded_boxes = []
        for box in boxes:
            class_id, score, x1, y1, x2, y2 = box
            if class_id in unclip_ratio:
                width_ratio, height_ratio = unclip_ratio[class_id]

                width = x2 - x1
                height = y2 - y1

                new_w = width * width_ratio
                new_h = height * height_ratio
                center_x = x1 + width / 2
                center_y = y1 + height / 2

                new_x1 = center_x - new_w / 2
                new_y1 = center_y - new_h / 2
                new_x2 = center_x + new_w / 2
                new_y2 = center_y + new_h / 2

                expanded_boxes.append([class_id, score, new_x1, new_y1, new_x2, new_y2])
            else:
                expanded_boxes.append(box)
        return np.array(expanded_boxes)

    else:
        widths = boxes[:, 4] - boxes[:, 2]
        heights = boxes[:, 5] - boxes[:, 3]

        new_w = widths * unclip_ratio[0]
        new_h = heights * unclip_ratio[1]
        center_x = boxes[:, 2] + widths / 2
        center_y = boxes[:, 3] + heights / 2

        new_x1 = center_x - new_w / 2
        new_y1 = center_y - new_h / 2
        new_x2 = center_x + new_w / 2
        new_y2 = center_y + new_h / 2
        expanded_boxes = np.column_stack(
            (boxes[:, 0], boxes[:, 1], new_x1, new_y1, new_x2, new_y2)
        )
        return expanded_boxes


def make_valid(poly):
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def calculate_polygon_overlap_ratio(
    polygon1: List[Tuple[int, int]],
    polygon2: List[Tuple[int, int]],
    mode: str = "union",
) -> float:
    """
    Calculate the overlap ratio between two polygons.

    Args:
        polygon1 (List[Tuple[int, int]]): First polygon represented as a list of points.
        polygon2 (List[Tuple[int, int]]): Second polygon represented as a list of points.
        mode (str, optional): Overlap calculation mode. Defaults to "union".

    Returns:
        float: Overlap ratio value between 0 and 1.
    """
    try:
        from shapely.geometry import Polygon
    except ImportError:
        raise ImportError("Please install Shapely library.")
    poly1 = Polygon(polygon1)
    poly2 = Polygon(polygon2)
    poly1 = make_valid(poly1)
    poly2 = make_valid(poly2)
    intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    if mode == "union":
        return intersection / union
    elif mode == "small":
        small_area = min(poly1.area, poly2.area)
        return intersection / small_area
    elif mode == "large":
        large_area = max(poly1.area, poly2.area)
        return intersection / large_area
    else:
        raise ValueError(f"Unknown mode: {mode}")


def calculate_bbox_area(bbox):
    """Calculate bounding box area"""
    x1, y1, x2, y2 = map(float, bbox)
    area = abs((x2 - x1) * (y2 - y1))
    return area


def calculate_overlap_ratio(
    bbox1: Union[np.ndarray, list, tuple],
    bbox2: Union[np.ndarray, list, tuple],
    mode="union",
) -> float:
    """
    Calculate the overlap ratio between two bounding boxes using NumPy.

    Args:
        bbox1 (np.ndarray, list or tuple): The first bounding box, format [x_min, y_min, x_max, y_max]
        bbox2 (np.ndarray, list or tuple): The second bounding box, format [x_min, y_min, x_max, y_max]
        mode (str): The mode of calculation, either 'union', 'small', or 'large'.

    Returns:
        float: The overlap ratio value between the two bounding boxes
    """
    bbox1 = np.array(bbox1)
    bbox2 = np.array(bbox2)

    x_min_inter = np.maximum(bbox1[0], bbox2[0])
    y_min_inter = np.maximum(bbox1[1], bbox2[1])
    x_max_inter = np.minimum(bbox1[2], bbox2[2])
    y_max_inter = np.minimum(bbox1[3], bbox2[3])

    inter_width = np.maximum(0, x_max_inter - x_min_inter)
    inter_height = np.maximum(0, y_max_inter - y_min_inter)

    inter_area = inter_width * inter_height

    bbox1_area = calculate_bbox_area(bbox1)
    bbox2_area = calculate_bbox_area(bbox2)

    if mode == "union":
        ref_area = bbox1_area + bbox2_area - inter_area
    elif mode == "small":
        ref_area = np.minimum(bbox1_area, bbox2_area)
    elif mode == "large":
        ref_area = np.maximum(bbox1_area, bbox2_area)
    else:
        raise ValueError(
            f"Invalid mode {mode}, must be one of ['union', 'small', 'large']."
        )

    if ref_area == 0:
        return 0.0

    return inter_area / ref_area


def filter_boxes(
    src_boxes: Dict[str, List[Dict]], layout_shape_mode: str
) -> Dict[str, List[Dict]]:
    """
    Remove overlapping boxes from layout detection results based on a given overlap ratio.

    Args:
        boxes (Dict[str, List[Dict]]): Layout detection result dict containing a 'boxes' list.

    Returns:
        Dict[str, List[Dict]]: Filtered dict with overlapping boxes removed.
    """
    boxes = [box for box in src_boxes if box["label"] != "reference"]
    dropped_indexes = set()

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]["coordinate"]
        w, h = x2 - x1, y2 - y1
        if w < 6 or h < 6:
            dropped_indexes.add(i)
        for j in range(i + 1, len(boxes)):
            if i in dropped_indexes or j in dropped_indexes:
                continue
            overlap_ratio = calculate_overlap_ratio(
                boxes[i]["coordinate"], boxes[j]["coordinate"], "small"
            )
            if (
                boxes[i]["label"] == "inline_formula"
                or boxes[j]["label"] == "inline_formula"
            ):
                if overlap_ratio > 0.5:
                    if boxes[i]["label"] == "inline_formula":
                        dropped_indexes.add(i)
                    if boxes[j]["label"] == "inline_formula":
                        dropped_indexes.add(j)
                    continue
            if overlap_ratio > 0.7:
                if layout_shape_mode != "rect" and "polygon_points" in boxes[i]:
                    poly_overlap_ratio = calculate_polygon_overlap_ratio(
                        boxes[i]["polygon_points"], boxes[j]["polygon_points"], "small"
                    )
                    if poly_overlap_ratio < 0.7:
                        continue
                box_area_i = calculate_bbox_area(boxes[i]["coordinate"])
                box_area_j = calculate_bbox_area(boxes[j]["coordinate"])
                if (
                    boxes[i]["label"] == "image" or boxes[j]["label"] == "image"
                ) and boxes[i]["label"] != boxes[j]["label"]:
                    continue
                if box_area_i >= box_area_j:
                    dropped_indexes.add(j)
                else:
                    dropped_indexes.add(i)
    out_boxes = [box for idx, box in enumerate(boxes) if idx not in dropped_indexes]
    return out_boxes


def update_order_index(boxes: List[Dict], skip_order_labels: List[str]):
    """
    Update the 'order_index' field of each box in the provided list of boxes.

    Args:
        boxes (List[Dict]): A list of boxes, where each box is represented as a dictionary with an 'order_index' field.

    Returns:
        None. The  function updates the 'order_index' field of each box in the input list.
    """
    order_index = 1
    for box in boxes:
        label = box["label"]
        if label not in skip_order_labels:
            box["order"] = order_index
            order_index += 1
        else:
            box["order"] = None
    return boxes


def find_label_position(box, polygon_points, text_w, text_h, max_shift=50):
    try:
        from shapely.geometry import Polygon
    except ImportError:
        raise ImportError("Please install Shapely library.")
    poly = Polygon(polygon_points)
    min_x = min([p[0] for p in polygon_points])
    min_y = min([p[1] for p in polygon_points])
    for dy in range(max_shift):
        x1, y1 = min_x, min_y + dy
        x2, y2 = x1 + text_w, y1 + text_h
        label_rect = box(x1, y1, x2, y2)
        if poly.intersects(label_rect):
            return int(x1), int(y1)

    return int(min_x), int(min_y)


@benchmark.timeit
class LayoutAnalysisProcess:
    """Save Result Transform

    This class is responsible for post-processing detection results, including
    thresholding, non-maximum suppression (NMS), and restructuring the boxes
    based on the input type (normal or rotated object detection).
    """

    def __init__(
        self, labels: Optional[List[str]] = None, scale_size: Optional[List[int]] = None
    ) -> None:
        """Initialize the DetPostProcess class.

        Args:
            threshold (float, optional): The threshold to apply to the detection scores. Defaults to 0.5.
            labels (Optional[List[str]], optional): The list of labels for the detection categories. Defaults to None.
            layout_postprocess (bool, optional): Whether to apply layout post-processing. Defaults to False.
        """
        super().__init__()
        self.labels = labels
        self.scale_size = scale_size

    def apply(
        self,
        boxes: ndarray,
        img_size: Tuple[int, int],
        threshold: Union[float, dict],
        layout_nms: Optional[bool],
        layout_unclip_ratio: Optional[Union[float, Tuple[float, float], dict]],
        layout_merge_bboxes_mode: Optional[Union[str, dict]],
        masks: Optional[ndarray] = None,
        layout_shape_mode: Optional[str] = "auto",
    ) -> Boxes:
        """Apply post-processing to the detection boxes.

        Args:
            boxes (ndarray): The input detection boxes with scores.
            img_size (tuple): The original image size.

        Returns:
            Boxes: The post-processed detection boxes.
        """
        if layout_shape_mode == "rect":
            masks = None
        boxes[:, 2:6] = np.round(boxes[:, 2:6]).astype(int)
        if isinstance(threshold, float):
            expect_boxes = (boxes[:, 1] > threshold) & (boxes[:, 0] > -1)
            boxes = boxes[expect_boxes, :]
            if masks is not None:
                masks = masks[expect_boxes, ...]
        elif isinstance(threshold, dict):
            category_filtered_boxes = []
            if masks is not None:
                category_filtered_masks = []
            for cat_id in np.unique(boxes[:, 0]):
                category_boxes = boxes[boxes[:, 0] == cat_id]
                if masks is not None:
                    category_masks = masks[boxes[:, 0] == cat_id]
                category_threshold = threshold.get(int(cat_id), 0.5)
                selected_indices = (category_boxes[:, 1] > category_threshold) & (
                    category_boxes[:, 0] > -1
                )
                if masks is not None:
                    category_masks = category_masks[selected_indices]
                    category_filtered_masks.append(category_masks)
                category_filtered_boxes.append(category_boxes[selected_indices])
            boxes = (
                np.vstack(category_filtered_boxes)
                if category_filtered_boxes
                else np.array([])
            )
            if masks is not None:
                masks = (
                    np.concatenate(category_filtered_masks)
                    if category_filtered_masks
                    else np.array([])
                )

        if layout_nms:
            selected_indices = nms(boxes[:, :6], iou_same=0.6, iou_diff=0.98)
            boxes = np.array(boxes[selected_indices])
            if masks is not None:
                masks = [masks[i] for i in selected_indices]

        filter_large_image = True
        # boxes.shape[1] == 6 is object detection, 7 is new ordered object detection, 8 is ordered object detection
        if filter_large_image and len(boxes) > 1 and boxes.shape[1] in [6, 7, 8]:
            if img_size[0] > img_size[1]:
                area_thres = 0.82
            else:
                area_thres = 0.93
            image_index = self.labels.index("image") if "image" in self.labels else None
            img_area = img_size[0] * img_size[1]
            filtered_boxes = []
            filtered_masks = []
            for idx, box in enumerate(boxes):
                (
                    label_index,
                    score,
                    xmin,
                    ymin,
                    xmax,
                    ymax,
                ) = box[:6]
                if label_index == image_index:
                    xmin = max(0, xmin)
                    ymin = max(0, ymin)
                    xmax = min(img_size[0], xmax)
                    ymax = min(img_size[1], ymax)
                    box_area = (xmax - xmin) * (ymax - ymin)
                    if box_area <= area_thres * img_area:
                        filtered_boxes.append(box)
                        if masks is not None:
                            filtered_masks.append(masks[idx])
                else:
                    filtered_boxes.append(box)
                    if masks is not None:
                        filtered_masks.append(masks[idx])
            if len(filtered_boxes) == 0:
                filtered_boxes = boxes
                if masks is not None:
                    filtered_masks = masks
            boxes = np.array(filtered_boxes)
            if masks is not None:
                masks = filtered_masks

        if layout_merge_bboxes_mode:
            formula_index = (
                self.labels.index("formula") if "formula" in self.labels else None
            )
            if isinstance(layout_merge_bboxes_mode, str):
                assert layout_merge_bboxes_mode in [
                    "union",
                    "large",
                    "small",
                ], f"The value of `layout_merge_bboxes_mode` must be one of ['union', 'large', 'small'], but got {layout_merge_bboxes_mode}"

                if layout_merge_bboxes_mode == "union":
                    pass
                else:
                    contains_other, contained_by_other = check_containment(
                        boxes[:, :6], formula_index
                    )
                    if layout_merge_bboxes_mode == "large":
                        boxes = boxes[contained_by_other == 0]
                        if masks is not None:
                            masks = [
                                mask
                                for i, mask in enumerate(masks)
                                if contained_by_other[i] == 0
                            ]
                    elif layout_merge_bboxes_mode == "small":
                        boxes = boxes[(contains_other == 0) | (contained_by_other == 1)]
                        if masks is not None:
                            masks = [
                                mask
                                for i, mask in enumerate(masks)
                                if (contains_other[i] == 0)
                                | (contained_by_other[i] == 1)
                            ]
            elif isinstance(layout_merge_bboxes_mode, dict):
                keep_mask = np.ones(len(boxes), dtype=bool)
                for category_index, layout_mode in layout_merge_bboxes_mode.items():
                    assert layout_mode in [
                        "union",
                        "large",
                        "small",
                    ], f"The value of `layout_merge_bboxes_mode` must be one of ['union', 'large', 'small'], but got {layout_mode}"
                    if layout_mode == "union":
                        pass
                    else:
                        if layout_mode == "large":
                            contains_other, contained_by_other = check_containment(
                                boxes[:, :6],
                                formula_index,
                                category_index,
                                mode=layout_mode,
                            )
                            # Remove boxes that are contained by other boxes
                            keep_mask &= contained_by_other == 0
                        elif layout_mode == "small":
                            contains_other, contained_by_other = check_containment(
                                boxes[:, :6],
                                formula_index,
                                category_index,
                                mode=layout_mode,
                            )
                            # Keep boxes that do not contain others or are contained by others
                            keep_mask &= (contains_other == 0) | (
                                contained_by_other == 1
                            )
                boxes = boxes[keep_mask]
                if masks is not None:
                    masks = [mask for i, mask in enumerate(masks) if keep_mask[i]]

        if boxes.size == 0:
            return np.array([])

        if boxes.shape[1] == 8:
            # Sort boxes by their order
            sorted_idx = np.lexsort((-boxes[:, 7], boxes[:, 6]))
            sorted_boxes = boxes[sorted_idx]
            boxes = sorted_boxes[:, :6]
            if masks is not None:
                sorted_masks = [masks[i] for i in sorted_idx]
                masks = sorted_masks

        if boxes.shape[1] == 7:
            # Sort boxes by their order
            sorted_idx = np.argsort(boxes[:, 6])
            sorted_boxes = boxes[sorted_idx]
            boxes = sorted_boxes[:, :6]
            if masks is not None:
                sorted_masks = [masks[i] for i in sorted_idx]
                masks = sorted_masks

        polygon_points = None
        if masks is not None:
            scale_ratio = [h / s for h, s in zip(self.scale_size, img_size)]
            polygon_points = extract_polygon_points_by_masks(
                boxes, np.array(masks), scale_ratio, layout_shape_mode
            )

        if layout_unclip_ratio:
            if isinstance(layout_unclip_ratio, float):
                layout_unclip_ratio = (layout_unclip_ratio, layout_unclip_ratio)
            elif isinstance(layout_unclip_ratio, (tuple, list)):
                assert (
                    len(layout_unclip_ratio) == 2
                ), f"The length of `layout_unclip_ratio` should be 2."
            elif isinstance(layout_unclip_ratio, dict):
                pass
            else:
                raise ValueError(
                    f"The type of `layout_unclip_ratio` must be float, Tuple[float, float] or  Dict[int, Tuple[float, float]], but got {type(layout_unclip_ratio)}."
                )
            boxes = unclip_boxes(boxes, layout_unclip_ratio)

        if boxes.shape[1] == 6:
            """For Normal Object Detection"""
            boxes = restructured_boxes(boxes, self.labels, img_size, polygon_points)
        else:
            """Unexpected Input Box Shape"""
            raise ValueError(
                f"The shape of boxes should be 6 or 10, instead of {boxes.shape[1]}"
            )
        return boxes

    def __call__(
        self,
        batch_outputs: List[dict],
        datas: List[dict],
        threshold: Optional[Union[float, dict]] = None,
        layout_nms: Optional[bool] = None,
        layout_unclip_ratio: Optional[Union[float, Tuple[float, float]]] = None,
        layout_merge_bboxes_mode: Optional[str] = None,
        layout_shape_mode: Optional[str] = None,
        filter_overlap_boxes: Optional[bool] = None,
        skip_order_labels: Optional[List[str]] = None,
    ) -> List[Boxes]:
        """Apply the post-processing to a batch of outputs.

        Args:
            batch_outputs (List[dict]): The list of detection outputs.
            datas (List[dict]): The list of input data.

        Returns:
            List[Boxes]: The list of post-processed detection boxes.
        """
        outputs = []
        for idx, (data, output) in enumerate(zip(datas, batch_outputs)):
            if "masks" in output:
                masks = output["masks"]
            else:
                layout_shape_mode = "rect"
                if idx == 0 and layout_shape_mode not in ["rect", "auto"]:
                    logging.warning(
                        f"The model you are using does not support polygon output, but the layout_shape_mode is specified as {layout_shape_mode}, which will be set to 'rect'"
                    )
                masks = None
            boxes = self.apply(
                output["boxes"],
                data["ori_img_size"],
                threshold,
                layout_nms,
                layout_unclip_ratio,
                layout_merge_bboxes_mode,
                masks,
                layout_shape_mode,
            )
            if filter_overlap_boxes:
                boxes = filter_boxes(boxes, layout_shape_mode)
            skip_order_labels = (
                skip_order_labels
                if skip_order_labels is not None
                else SKIP_ORDER_LABELS
            )
            boxes = update_order_index(boxes, skip_order_labels)
            outputs.append(boxes)
        return outputs
