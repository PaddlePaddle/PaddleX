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
from .utils import compute_reading_order

if is_dep_available("opencv-contrib-python"):
    import cv2
if is_dep_available("shapely"):
    from shapely.geometry import Polygon

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


def extract_polygon_points_by_masks(
    boxes, masks, scale_ratio, layout_shape_mode, quad=None
):
    """
    修改后的提取函数：auto 模式下信任几何决策
    """
    scale_w, scale_h = scale_ratio[0] / 4, scale_ratio[1] / 4
    h_m, w_m = masks.shape[1:]
    polygon_points = []

    max_box_w = max(boxes[:, 4] - boxes[:, 3])

    for i in range(len(boxes)):
        x_min, y_min, x_max, y_max = boxes[i, 2:6].astype(np.int32)
        box_w, box_h = x_max - x_min, y_max - y_min
        rect = _rect_from_box(boxes[i, 2:6])

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

        # resize mask to match box size
        resized_mask = cv2.resize(
            cropped.astype(np.uint8), (box_w, box_h), interpolation=cv2.INTER_NEAREST
        )

        if box_w > max_box_w * 0.6:
            max_allowed_dist = box_w
        else:
            max_allowed_dist = max_box_w

        polygon = mask2polygon(resized_mask, max_allowed_dist)
        if polygon is not None and len(polygon) > 0:
            polygon = polygon + np.array([x_min, y_min])
        polygon_points.append(
            _normalize_layout_polygon(
                box=boxes[i, 2:6],
                polygon=polygon,
                layout_shape_mode=layout_shape_mode,
                previous_polygon=(
                    polygon_points[-1] if len(polygon_points) > 0 else None
                ),
                model_quad=quad[i].reshape(4, 2) if quad is not None else None,
            )
        )

    return polygon_points


def normalize_polygon_points_by_boxes(boxes, polygon_points, layout_shape_mode):
    normalized_points = []

    for polygon, box in zip(polygon_points, boxes):
        normalized_points.append(
            _normalize_layout_polygon(
                box=box[2:6],
                polygon=polygon,
                layout_shape_mode=layout_shape_mode,
                previous_polygon=(
                    normalized_points[-1] if len(normalized_points) > 0 else None
                ),
            )
        )

    return normalized_points


def _rect_from_box(box):
    x_min, y_min, x_max, y_max = np.asarray(box).astype(np.int32)
    return np.array(
        [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
        dtype=np.float32,
    )


def _normalize_layout_polygon(
    box,
    polygon,
    layout_shape_mode,
    previous_polygon=None,
    model_quad=None,
):
    rect = _rect_from_box(box)

    if polygon is None and model_quad is None:
        return rect

    if polygon is not None:
        polygon = np.asarray(polygon, dtype=np.float32)
        if polygon.ndim == 1:
            polygon = polygon.reshape(-1, 2)

    if polygon is not None and len(polygon) < 4 and model_quad is None:
        return rect

    if layout_shape_mode == "rect":
        return rect

    if layout_shape_mode == "poly":
        return polygon

    # The quad predicted by 4-point models takes precedence over the fitted one.
    quad = model_quad if model_quad is not None else convert_polygon_to_quad(polygon)
    if layout_shape_mode == "quad":
        return quad if quad is not None else rect

    if layout_shape_mode == "auto":
        rect_list = rect.tolist()
        if quad is not None:
            quad_list = quad.tolist()
            iou_rect_quad = calculate_polygon_overlap_ratio(
                rect_list, quad_list, mode="union"
            )
            if iou_rect_quad >= 0.95:
                return rect

            poly_list = polygon.tolist()
            iou_poly_quad = calculate_polygon_overlap_ratio(
                poly_list, quad_list, mode="union"
            )

            iou_pre = 0
            if previous_polygon is not None:
                iou_pre = calculate_polygon_overlap_ratio(
                    previous_polygon.tolist(),
                    rect_list,
                    mode="small",
                )

            if iou_poly_quad >= 0.8 and iou_pre < 0.01:
                return quad

        return polygon

    raise ValueError(
        "layout_shape_mode must be one of ['rect', 'poly', 'quad', 'auto']"
    )


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
    quad: ndarray = None,
) -> Boxes:
    """
    Restructure the given bounding boxes and labels based on the image size.

    Args:
        boxes (ndarray): A 2D array of bounding boxes with each box represented as [cls_id, score, xmin, ymin, xmax, ymax].
        labels (List[str]): A list of class labels corresponding to the class ids.
        img_size (Tuple[int, int]): A tuple representing the width and height of the image.
        polygon_points (ndarray): A 2D array of polygon points with each point represented as [x, y].
        quad (ndarray, optional): Corners [N, 8] predicted by 4-point models.
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
        if quad is not None:
            res["quad"] = quad[idx].reshape(4, 2).tolist()
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


@function_requires_deps("shapely")
def _quad_geometry(quads):
    """Build the polygons, areas and AABBs of quads [N, 8] once.

    The pairwise helpers below are O(N^2), so the shapely polygons are built a
    single time per box rather than per pair.
    """
    points = np.asarray(quads, dtype=np.float64).reshape(-1, 4, 2)
    polygons = [make_valid(Polygon(p)) for p in points]
    areas = np.array([p.area for p in polygons], dtype=np.float64)
    extents = np.concatenate([points.min(axis=1), points.max(axis=1)], axis=1)
    return polygons, areas, extents


def quad_area(quad):
    """Shoelace area of a quad [8] (or [4, 2])."""
    pts = quad.reshape(4, 2)
    area = 0.0
    for i in range(4):
        j = (i + 1) % 4
        area += pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1]
    return abs(area) / 2.0


def _take_quad_geometry(geometry, indices):
    """Select the entries ``indices`` of ``_quad_geometry`` output."""
    polygons, areas, extents = geometry
    return [polygons[i] for i in indices], areas[indices], extents[indices]


def _quad_intersection(geometry, i, j):
    """Intersection area of quads ``i`` and ``j`` of ``_quad_geometry`` output."""
    polygons, _, extents = geometry
    a, b = extents[i], extents[j]
    # Disjoint (or edge-touching) AABBs cannot share any area.
    if a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]:
        return 0.0
    return polygons[i].intersection(polygons[j]).area


def nms_quad(boxes, geometry, iou_same=0.6, iou_diff=0.95):
    """``nms`` on the IoU of quads; ``boxes`` supplies classes and scores.

    ``geometry`` is the ``_quad_geometry`` output for the quads of ``boxes``.
    """
    areas = geometry[1]
    indices = np.argsort(boxes[:, 1])[::-1].tolist()
    selected_boxes = []

    while len(indices) > 0:
        current = indices.pop(0)
        selected_boxes.append(current)
        filtered_indices = []
        for i in indices:
            inter_area = _quad_intersection(geometry, current, i)
            union_area = areas[current] + areas[i] - inter_area
            iou_value = inter_area / union_area if union_area > 0 else 0.0
            threshold = iou_same if boxes[current, 0] == boxes[i, 0] else iou_diff
            # If the IoU is below the threshold, keep the box
            if iou_value < threshold:
                filtered_indices.append(i)
        indices = filtered_indices
    return selected_boxes


def check_containment_quad(
    geometry, class_ids, formula_index=None, category_index=None, mode=None
):
    """``check_containment`` on quads, with the classes in ``class_ids``.

    ``geometry`` is the ``_quad_geometry`` output for the quads.
    """
    areas = geometry[1]
    n = len(areas)
    contains_other = np.zeros(n, dtype=int)
    contained_by_other = np.zeros(n, dtype=int)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if formula_index is not None:
                if class_ids[i] == formula_index and class_ids[j] != formula_index:
                    continue
            if category_index is not None and mode is not None:
                if not (
                    (mode == "large" and class_ids[j] == category_index)
                    or (mode == "small" and class_ids[i] == category_index)
                ):
                    continue
            # quads[i] is contained in quads[j] when it covers >= 90% of its area
            if areas[i] > 0 and _quad_intersection(geometry, i, j) / areas[i] >= 0.9:
                contained_by_other[i] = 1
                contains_other[j] = 1
    return contains_other, contained_by_other


def _pairwise_overlap_small(coords):
    """Pairwise ``calculate_overlap_ratio(..., "small")`` and areas of AABBs [N, 4]."""
    x1, y1, x2, y2 = coords.T
    areas = np.abs((x2 - x1) * (y2 - y1))
    inter_w = np.maximum(
        0, np.minimum(x2[:, None], x2[None, :]) - np.maximum(x1[:, None], x1[None, :])
    )
    inter_h = np.maximum(
        0, np.minimum(y2[:, None], y2[None, :]) - np.maximum(y1[:, None], y1[None, :])
    )
    small_area = np.minimum(areas[:, None], areas[None, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        overlap = np.where(small_area > 0, inter_w * inter_h / small_area, 0.0)
    return overlap, areas


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
    # Compare the quads of 4-point models rather than their AABBs, which
    # overlap heavily on rotated or skewed pages.
    if boxes and all("quad" in box for box in boxes):
        quad_geometry = _quad_geometry([box["quad"] for box in boxes])
        # The overlap is relative to the repaired polygons, while the larger
        # box is picked by the shoelace area of the predicted corners.
        polygon_areas = quad_geometry[1]
        areas = [quad_area(np.array(box["quad"])) for box in boxes]

        def get_overlap_ratio(i, j):
            small_area = min(polygon_areas[i], polygon_areas[j])
            if small_area <= 0:
                return 0.0
            return _quad_intersection(quad_geometry, i, j) / small_area

    else:
        coords = np.array([box["coordinate"] for box in boxes], dtype=np.float64)
        overlap_matrix, areas = _pairwise_overlap_small(coords.reshape(-1, 4))

        def get_overlap_ratio(i, j):
            return overlap_matrix[i, j]

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]["coordinate"]
        w, h = x2 - x1, y2 - y1
        if w < 6 or h < 6:
            dropped_indexes.add(i)
        for j in range(i + 1, len(boxes)):
            if i in dropped_indexes or j in dropped_indexes:
                continue
            overlap_ratio = get_overlap_ratio(i, j)
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
                labels = {boxes[i]["label"], boxes[j]["label"]}
                if labels & {"image", "table", "seal", "chart"} and len(labels) > 1:
                    if "table" not in labels or labels <= {
                        "table",
                        "image",
                        "seal",
                        "chart",
                    }:
                        continue
                if areas[i] >= areas[j]:
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
        polygon_points: Optional[List[ndarray]] = None,
        quad: Optional[ndarray] = None,
        qi: Optional[ndarray] = None,
        rel_logits: Optional[ndarray] = None,
        roor_logits: Optional[ndarray] = None,
    ) -> Boxes:
        """Apply post-processing to the detection boxes.

        Args:
            boxes (ndarray): The input detection boxes with scores.
            img_size (tuple): The original image size.
            quad (ndarray, optional): Corners [N, 8] predicted by 4-point models
                (PP-DocLayoutV4), in original-image coordinates.
            qi (ndarray, optional): The query index [N] of each box, for models
                whose reading order is decoded here from the pairwise
                ``rel_logits`` and ``roor_logits`` [Q, Q].

        Returns:
            Boxes: The post-processed detection boxes.
        """
        if layout_shape_mode == "rect":
            masks = None
            polygon_points = None
            quad = None
        for name, values in (
            ("masks", masks),
            ("polygon_points", polygon_points),
            ("quad", quad),
            ("qi", qi),
        ):
            if values is not None and len(values) != len(boxes):
                raise ValueError(
                    f"Expected one entry of `{name}` per box ({len(boxes)}), "
                    f"got {len(values)}."
                )
        boxes[:, 2:6] = np.round(boxes[:, 2:6]).astype(int)
        # Indices of the remaining boxes into the model output. Masks, polygons
        # and quads are gathered with it once the boxes are final.
        keep = np.arange(len(boxes))
        if isinstance(threshold, float):
            keep = np.flatnonzero((boxes[:, 1] > threshold) & (boxes[:, 0] > -1))
        elif isinstance(threshold, dict):
            keep = [
                np.flatnonzero(
                    (boxes[:, 0] == cat_id)
                    & (boxes[:, 1] > threshold.get(int(cat_id), 0.5))
                )
                for cat_id in np.unique(boxes[:, 0])
                if cat_id > -1
            ]
            keep = np.concatenate(keep) if keep else np.array([], dtype=np.int64)
        boxes = boxes[keep]

        quad_geometry = geometry_rows = None

        def kept_quad_geometry(indices):
            # Built once, on first use, for the boxes kept so far; the later
            # stages only narrow them down. ``geometry_rows`` maps a model
            # output index to its entry.
            nonlocal quad_geometry, geometry_rows
            if quad_geometry is None:
                quad_geometry = _quad_geometry(quad[keep])
                geometry_rows = np.zeros(len(quad), dtype=np.int64)
                geometry_rows[keep] = np.arange(len(keep))
            return _take_quad_geometry(quad_geometry, geometry_rows[indices])

        if qi is not None:
            order = compute_reading_order(qi[keep], rel_logits, roor_logits)
            boxes = np.column_stack([boxes, order.astype(boxes.dtype)])

        if layout_nms:
            if quad is not None:
                selected = nms_quad(
                    boxes, kept_quad_geometry(keep), iou_same=0.6, iou_diff=0.98
                )
            else:
                selected = nms(boxes[:, :6], iou_same=0.6, iou_diff=0.98)
            boxes, keep = boxes[selected], keep[selected]

        # Drop "image" boxes that cover nearly the whole page, unless that would
        # drop every box.
        if len(boxes) > 1 and "image" in self.labels:
            img_w, img_h = img_size
            area_thres = 0.82 if img_w > img_h else 0.93
            is_image = boxes[:, 0] == self.labels.index("image")
            if quad is not None:
                areas = np.array([quad_area(quad[i]) for i in keep[is_image]])
            else:
                coords = np.clip(boxes[is_image, 2:6], 0, [img_w, img_h] * 2)
                areas = (coords[:, 2] - coords[:, 0]) * (coords[:, 3] - coords[:, 1])
            is_large = np.zeros(len(boxes), dtype=bool)
            is_large[is_image] = areas > area_thres * img_w * img_h
            if not is_large.all():
                boxes, keep = boxes[~is_large], keep[~is_large]

        if layout_merge_bboxes_mode:
            formula_index = (
                self.labels.index("formula") if "formula" in self.labels else None
            )
            # A single mode applies to every category.
            if isinstance(layout_merge_bboxes_mode, str):
                merge_modes = {None: layout_merge_bboxes_mode}
            elif isinstance(layout_merge_bboxes_mode, dict):
                merge_modes = layout_merge_bboxes_mode
            else:
                merge_modes = {}
            keep_mask = np.ones(len(boxes), dtype=bool)
            if quad is not None:
                merge_geometry = kept_quad_geometry(keep)
            for category_index, layout_mode in merge_modes.items():
                assert layout_mode in [
                    "union",
                    "large",
                    "small",
                ], f"The value of `layout_merge_bboxes_mode` must be one of ['union', 'large', 'small'], but got {layout_mode}"
                if layout_mode == "union":
                    continue
                if quad is not None:
                    contains_other, contained_by_other = check_containment_quad(
                        merge_geometry,
                        boxes[:, 0],
                        formula_index,
                        category_index,
                        mode=layout_mode,
                    )
                else:
                    contains_other, contained_by_other = check_containment(
                        boxes[:, :6], formula_index, category_index, mode=layout_mode
                    )
                if layout_mode == "large":
                    # Remove boxes that are contained by other boxes
                    keep_mask &= contained_by_other == 0
                else:
                    # Keep boxes that do not contain others or are contained by others
                    keep_mask &= (contains_other == 0) | (contained_by_other == 1)
            boxes, keep = boxes[keep_mask], keep[keep_mask]

        if boxes.size == 0:
            return np.array([])

        if boxes.shape[1] in (7, 8):
            # Sort boxes by their order
            if boxes.shape[1] == 8:
                sorted_idx = np.lexsort((-boxes[:, 7], boxes[:, 6]))
            else:
                sorted_idx = np.argsort(boxes[:, 6])
            boxes, keep = boxes[sorted_idx, :6], keep[sorted_idx]

        if quad is not None:
            quad = quad[keep]
        if polygon_points is None and masks is not None:
            scale_ratio = [h / s for h, s in zip(self.scale_size, img_size)]
            polygon_points = extract_polygon_points_by_masks(
                boxes, np.asarray(masks)[keep], scale_ratio, layout_shape_mode, quad
            )
        elif polygon_points is not None:
            polygon_points = normalize_polygon_points_by_boxes(
                boxes, [polygon_points[i] for i in keep], layout_shape_mode
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
                    f"The type of `layout_unclip_ratio` must be float, Tuple[float, float] or Dict[int, Tuple[float, float]], but got {type(layout_unclip_ratio)}."
                )
            boxes = unclip_boxes(boxes, layout_unclip_ratio)

        if boxes.shape[1] == 6:
            """For Normal Object Detection"""
            boxes = restructured_boxes(
                boxes, self.labels, img_size, polygon_points, quad=quad
            )
        else:
            """Unexpected Input Box Shape"""
            raise ValueError(
                f"The shape of boxes should be 6, 7 or 8, instead of {boxes.shape[1]}"
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
            current_layout_shape_mode = layout_shape_mode
            if "masks" in output:
                masks = output["masks"]
                polygon_points = None
            elif "polygon_points" in output:
                masks = None
                polygon_points = output["polygon_points"]
            elif "quad" in output:
                # 4-point models (PP-DocLayoutV4) predict the quads directly.
                masks = None
                polygon_points = None
            else:
                current_layout_shape_mode = "rect"
                if idx == 0 and layout_shape_mode not in ["rect", "auto"]:
                    logging.warning(
                        f"The model you are using does not support polygon output, but the layout_shape_mode is specified as {layout_shape_mode}, which will be set to 'rect'"
                    )
                masks = None
                polygon_points = None
            boxes = self.apply(
                output["boxes"],
                data["ori_img_size"],
                threshold,
                layout_nms,
                layout_unclip_ratio,
                layout_merge_bboxes_mode,
                masks,
                current_layout_shape_mode,
                polygon_points=polygon_points,
                quad=output.get("quad"),
                qi=output.get("qi"),
                rel_logits=output.get("rel_logits"),
                roor_logits=output.get("roor_logits"),
            )
            if filter_overlap_boxes:
                boxes = filter_boxes(boxes, current_layout_shape_mode)
            skip_order_labels = (
                skip_order_labels
                if skip_order_labels is not None
                else SKIP_ORDER_LABELS
            )
            boxes = update_order_index(boxes, skip_order_labels)
            outputs.append(boxes)
        return outputs
