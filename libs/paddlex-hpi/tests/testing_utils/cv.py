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

from shapely.geometry import Polygon


def compute_iou(box_or_poly1, box_or_poly2):
    if isinstance(box_or_poly1[0], list):
        poly1 = box_or_poly1
        poly2 = box_or_poly2

        poly1 = Polygon(poly1)
        poly2 = Polygon(poly2)

        inter_area = poly1.intersection(poly2).area
        union_area = poly1.union(poly2).area

        iou = inter_area / (union_area + 1e-9)

        return iou
    else:
        box1 = box_or_poly1
        box2 = box_or_poly2

        x11, y11, x12, y12 = box1
        x21, y21, x22, y22 = box2

        xi1 = max(x11, x21)
        yi1 = max(y11, y21)
        xi2 = min(x12, x22)
        yi2 = min(y12, y22)

        inter_area = max(0, xi2 - xi1 + 1) * max(0, yi2 - yi1 + 1)
        box1_area = (x12 - x11 + 1) * (y12 - y11 + 1)
        box2_area = (x22 - x21 + 1) * (y22 - y21 + 1)
        union_area = box1_area + box2_area - inter_area

        iou = inter_area / (union_area + 1e-9)

        return iou


def compare_det_results(
    boxes_or_polys1,
    boxes_or_polys2,
    *,
    labels1=None,
    labels2=None,
    scores1=None,
    scores2=None,
    iou_tol=0.1,
    score_tol=1e-3,
):
    compare_labels = labels1 is not None
    compare_scores = scores1 is not None

    assert len(boxes_or_polys1) == len(boxes_or_polys2)
    if compare_labels:
        assert len(labels1) == len(labels2)
    if compare_scores:
        assert len(scores1) == len(scores2)

    boxes_or_polys2 = boxes_or_polys2.copy()
    if labels2 is not None:
        labels2 = labels2.copy()
    if scores2 is not None:
        scores2 = scores2.copy()
    for i, box_or_poly1 in enumerate(boxes_or_polys1):
        j = 0
        max_iou = 0
        for k, box_or_poly2 in enumerate(boxes_or_polys2):
            iou = compute_iou(box_or_poly1, box_or_poly2)
            if iou > max_iou:
                max_iou = iou
                j = k
        assert max_iou > 1 - iou_tol
        if compare_labels:
            assert labels1[i] == labels2[j]
        if compare_scores:
            assert abs(scores1[i] - scores2[j]) < score_tol
        del boxes_or_polys2[j]
        if compare_labels:
            del labels2[j]
        if compare_scores:
            del scores2[j]
