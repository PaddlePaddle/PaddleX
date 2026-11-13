# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""Layout geometry must survive Transformers adaptation and box processing."""

import numpy as np
import pytest

from paddlex.inference.models.layout_analysis.predictor import (
    LayoutAnalysisTransformersPredictor,
)
from paddlex.inference.models.layout_analysis.processors import LayoutAnalysisProcess
from paddlex.inference.pipelines.components.common.crop_image_regions import CropByBoxes


class _Tensor:
    """The CPU tensor interface used by the adapter, without loading model weights."""

    def __init__(self, data):
        self.data = np.asarray(data)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.data


def _adapt(model_name, boxes, polygons, order):
    predictor = object.__new__(LayoutAnalysisTransformersPredictor)
    predictor.model_name = model_name
    prediction = {
        "boxes": _Tensor(boxes),
        "scores": _Tensor(np.full(len(boxes), 0.9)),
        "labels": _Tensor(np.zeros(len(boxes), dtype=np.int64)),
        "order_seq": _Tensor(order),
        "polygon_points": (
            _Tensor(polygons) if model_name == "PP-DocLayoutV4" else polygons
        ),
    }
    return predictor._format_layout_transformers_output(prediction)


def _postprocess(output, **kwargs):
    settings = dict(
        threshold=0.5,
        layout_nms=False,
        layout_unclip_ratio=None,
        layout_merge_bboxes_mode=None,
        layout_shape_mode="poly",
        filter_overlap_boxes=False,
        skip_order_labels=[],
    )
    settings.update(kwargs)
    return LayoutAnalysisProcess(
        labels=["text", "table", "image"], scale_size=[100, 100]
    )([output], [{"ori_img_size": (100, 100)}], **settings)[0]


@pytest.mark.parametrize("model_name", ["PP-DocLayoutV3", "PP-DocLayoutV4"])
@pytest.mark.parametrize("shape_mode", ["poly", "auto", "quad", "rect"])
def test_transformers_geometry_reaches_crop(model_name, shape_mode):
    diamond = np.array([[40, 10], [70, 40], [40, 70], [10, 40]], dtype=np.float32)
    output = _adapt(model_name, [[10, 10, 70, 70]], [diamond], [0])
    boxes = _postprocess(output, layout_shape_mode=shape_mode)

    assert len(boxes) == 1
    assert boxes[0]["coordinate"] == [10, 10, 70, 70]
    if shape_mode == "rect":
        assert "polygon_points" not in boxes[0]
        assert "quad" not in boxes[0]
    else:
        key = "quad" if model_name == "PP-DocLayoutV4" else "polygon_points"
        assert key in boxes[0]
        points = np.asarray(boxes[0][key])
        # Quad normalization may pick a different starting corner.
        np.testing.assert_allclose(
            points[np.lexsort(points.T)], diamond[np.lexsort(diamond.T)], atol=1e-5
        )
        other_key = "polygon_points" if key == "quad" else "quad"
        assert other_key not in boxes[0]

    crop = CropByBoxes()(np.zeros((100, 100, 3), dtype=np.uint8), boxes, shape_mode)[0]
    assert crop["img"].shape == (60, 60, 3)
    np.testing.assert_array_equal(crop["img"][30, 30], [0, 0, 0])
    expected_corner = [0, 0, 0] if shape_mode == "rect" else [255, 255, 255]
    np.testing.assert_array_equal(crop["img"][0, 0], expected_corner)


def test_v3_ragged_polygons_follow_reading_order():
    concave = np.array([[10, 10], [30, 10], [30, 20], [20, 20], [20, 30], [10, 30]])
    rectangle = np.array([[50, 50], [70, 50], [70, 70], [50, 70]])
    output = _adapt(
        "PP-DocLayoutV3",
        [[10, 10, 30, 30], [50, 50, 70, 70]],
        [concave, rectangle],
        [1, 0],
    )
    boxes = _postprocess(output)
    np.testing.assert_array_equal(boxes[0]["polygon_points"], rectangle)
    np.testing.assert_array_equal(boxes[1]["polygon_points"], concave)
    crop = CropByBoxes()(np.zeros((100, 100, 3), dtype=np.uint8), boxes, "poly")[1][
        "img"
    ]
    np.testing.assert_array_equal(crop[15, 15], [255, 255, 255])
    np.testing.assert_array_equal(crop[5, 5], [0, 0, 0])


@pytest.mark.parametrize("geometry_key", ["polygon_points", "quad"])
@pytest.mark.parametrize("threshold", [0.5, {0: 0.75, 1: 0.9, 2: 0.9}])
@pytest.mark.parametrize("order_columns", [1, 2])
@pytest.mark.parametrize(
    "merge_mode, expected_indices",
    [
        (None, [1, 0, 5]),
        ("large", [1, 0]),
        ("small", [1, 5]),
        ({0: "large"}, [1, 0]),
        ({0: "small"}, [1, 5]),
    ],
)
def test_geometry_tracks_threshold_nms_image_filter_containment_and_sort(
    geometry_key, threshold, order_columns, merge_mode, expected_indices
):
    # Reorder by class, suppress a duplicate, drop a low score and a page-sized
    # image, then filter nested text and restore reading order.
    boxes = np.array(
        [
            [0, 0.8, 50, 10, 70, 30, 2],
            [1, 0.95, 10, 10, 30, 30, 0],
            [0, 0.7, 50, 10, 70, 30, 9],
            [0, 0.2, 10, 60, 30, 80, 1],
            [2, 0.99, 0, 0, 100, 100, 8],
            [0, 0.85, 55, 15, 60, 20, 3],
        ],
        dtype=np.float32,
    )
    if order_columns == 2:
        boxes = np.column_stack([boxes, np.ones(len(boxes), dtype=np.float32)])
    polygons = [
        np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        for x1, y1, x2, y2 in boxes[:, 2:6]
    ]
    geometry = (
        np.asarray(polygons).reshape(-1, 8) if geometry_key == "quad" else polygons
    )
    result = _postprocess(
        {"boxes": boxes, geometry_key: geometry},
        threshold=threshold,
        layout_nms=True,
        layout_merge_bboxes_mode=merge_mode,
    )
    assert [box["coordinate"] for box in result] == [
        boxes[i, 2:6].tolist() for i in expected_indices
    ]
    for box, index in zip(result, expected_indices):
        np.testing.assert_array_equal(box[geometry_key], polygons[index])
        other_key = "polygon_points" if geometry_key == "quad" else "quad"
        assert other_key not in box


@pytest.mark.parametrize("geometry_key", ["polygon_points", "quad"])
def test_geometry_tracks_reading_order_tiebreaker(geometry_key):
    boxes = np.array([[0, 0.8, 10, 10, 30, 30, 1, 2], [0, 0.9, 50, 50, 70, 70, 1, 5]])
    polygons = [
        np.array([[10, 10], [30, 10], [30, 30], [10, 30]]),
        np.array([[50, 50], [70, 50], [70, 70], [50, 70]]),
    ]
    geometry = (
        np.asarray(polygons).reshape(-1, 8) if geometry_key == "quad" else polygons
    )
    result = _postprocess({"boxes": boxes, geometry_key: geometry})
    np.testing.assert_array_equal(result[0][geometry_key], polygons[1])
    np.testing.assert_array_equal(result[1][geometry_key], polygons[0])


def test_v4_quad_tracks_query_reading_order():
    polygons = np.array(
        [
            [[10, 10], [30, 10], [30, 30], [10, 30]],
            [[50, 50], [70, 50], [70, 70], [50, 70]],
        ],
        dtype=np.float32,
    )
    output = {
        "boxes": np.array([[0, 0.8, 10, 10, 30, 30], [0, 0.9, 50, 50, 70, 70]]),
        "quad": polygons.reshape(-1, 8),
        "qi": np.array([1, 0]),
        "rel_logits": np.array([[0, 10], [-10, 0]]),
        "roor_logits": np.array([[-10, 10], [-10, -10]]),
    }
    result = _postprocess(output, layout_nms=True)
    np.testing.assert_array_equal(result[0]["quad"], polygons[1])
    np.testing.assert_array_equal(result[1]["quad"], polygons[0])
    assert all("polygon_points" not in box for box in result)


def test_v3_overlap_filter_uses_polygons_instead_of_enclosing_boxes():
    concave = np.array([[10, 10], [70, 10], [70, 25], [25, 25], [25, 70], [10, 70]])
    rectangle = np.array([[40, 40], [70, 40], [70, 70], [40, 70]])
    output = _adapt(
        "PP-DocLayoutV3",
        [[10, 10, 70, 70], [40, 40, 70, 70]],
        [concave, rectangle],
        [0, 1],
    )
    result = _postprocess(output, filter_overlap_boxes=True)
    # The second AABB is inside the first, but their polygons are disjoint.
    assert len(result) == 2
    np.testing.assert_array_equal(result[0]["polygon_points"], concave)
    np.testing.assert_array_equal(result[1]["polygon_points"], rectangle)


def test_v4_nms_and_overlap_filter_use_quad_geometry():
    polygons = np.array(
        [
            [[10, 10], [14, 10], [70, 66], [66, 66]],
            [[10, 18], [14, 18], [70, 74], [66, 74]],
        ],
        dtype=np.float32,
    )
    output = _adapt(
        "PP-DocLayoutV4", [[10, 10, 70, 66], [10, 18, 70, 74]], polygons, [0, 1]
    )
    result = _postprocess(output, layout_nms=True, filter_overlap_boxes=True)
    # AABB IoU is 0.75, but the two thin quads do not overlap.
    assert len(result) == 2
    for box, polygon in zip(result, polygons):
        np.testing.assert_array_equal(box["quad"], polygon)
        assert "polygon_points" not in box


@pytest.mark.parametrize("geometry_key", [None, "polygon_points", "quad"])
@pytest.mark.parametrize(
    "second_label, second_box",
    [("text", [41, 15, 61, 25]), ("inline_formula", [46, 15, 66, 25])],
)
def test_overlap_filter_compares_output_geometry(
    geometry_key, second_label, second_box
):
    # The second box overlaps the first by 0.7 (0.45 for inline formulas), which
    # the filter keeps; expanded by 1.1 their boxes overlap by more. Quads are
    # not expanded, so they still overlap by 0.7.
    labels = ["text", "inline_formula"]
    boxes = np.array(
        [[0, 0.9, 5, 5, 55, 55], [labels.index(second_label), 0.8, *second_box]],
        dtype=np.float32,
    )
    output = {"boxes": boxes}
    if geometry_key is not None:
        polygons = [
            np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
            for x1, y1, x2, y2 in boxes[:, 2:6]
        ]
        output[geometry_key] = (
            np.asarray(polygons).reshape(-1, 8) if geometry_key == "quad" else polygons
        )
    result = LayoutAnalysisProcess(labels=labels, scale_size=[100, 100])(
        [output],
        [{"ori_img_size": (100, 100)}],
        threshold=0.5,
        layout_nms=False,
        layout_unclip_ratio=1.1,
        layout_merge_bboxes_mode=None,
        layout_shape_mode="poly" if geometry_key else "rect",
        filter_overlap_boxes=True,
        skip_order_labels=[],
    )[0]
    assert len(result) == (2 if geometry_key == "quad" else 1)
    assert result[0]["coordinate"] == [2, 2, 57, 57]


@pytest.mark.parametrize("source", ["masks", "polygon_points", "quad"])
def test_unclip_expands_boxes_only(source):
    diamond = np.array([[50, 30], [70, 50], [50, 70], [30, 50]], dtype=np.float32)
    output = {"boxes": np.array([[0, 0.9, 30, 30, 70, 70]], dtype=np.float32)}
    if source == "masks":
        output["masks"] = np.ones((1, 25, 25), dtype=np.uint8)
    elif source == "polygon_points":
        output["polygon_points"] = [diamond]
    else:
        output["quad"] = diamond.reshape(1, 8)
    plain = _postprocess(dict(output, boxes=output["boxes"].copy()))[0]
    result = _postprocess(output, layout_unclip_ratio=1.5)[0]
    assert plain["coordinate"] == [30, 30, 70, 70]
    assert result["coordinate"] == [20, 20, 80, 80]
    key = "quad" if source == "quad" else "polygon_points"
    np.testing.assert_array_equal(result[key], plain[key])
    if source != "masks":
        np.testing.assert_array_equal(np.reshape(result[key], (4, 2)), diamond)


def test_paddle_masks_keep_precedence_over_decoded_polygons():
    output = {
        "boxes": np.array([[0, 0.9, 10, 10, 70, 70]]),
        "masks": np.ones((1, 25, 25), dtype=np.uint8),
        "polygon_points": [np.array([[40, 10], [70, 40], [40, 70], [10, 40]])],
    }
    result = _postprocess(output)
    assert len(result[0]["polygon_points"]) == 4
    crop = CropByBoxes()(np.zeros((100, 100, 3), dtype=np.uint8), result, "poly")[0][
        "img"
    ]
    # The rectangular mask keeps this pixel; the supplied diamond would hide it.
    np.testing.assert_array_equal(crop[5, 5], [0, 0, 0])


@pytest.mark.parametrize("model_name", ["PP-DocLayoutV3", "PP-DocLayoutV4"])
@pytest.mark.parametrize("threshold", [0.5, {0: 0.5}])
def test_empty_transformers_geometry(model_name, threshold):
    output = _adapt(model_name, np.empty((0, 4)), np.empty((0, 4, 2)), [])
    result = _postprocess(output, threshold=threshold, layout_nms=True)
    assert len(result) == 0
