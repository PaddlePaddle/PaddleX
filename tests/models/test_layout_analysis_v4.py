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

"""PP-DocLayoutV4 Paddle outputs and query-level reading order."""

import numpy as np
import pytest

from paddlex.inference.models.layout_analysis.predictor import (
    LayoutAnalysisRunnerPredictor,
)
from paddlex.inference.models.layout_analysis.processors import LayoutAnalysisProcess
from paddlex.inference.models.layout_analysis.utils import (
    _get_order_single_image,
    compute_reading_order,
)

# Two pages of three emitted rows each. On page 0, query 2 is emitted under two
# labels; its first row is a rotated quad.
_QUADS = np.array(
    [
        [20, 10, 40, 20, 30, 40, 10, 30],
        [50, 50, 70, 50, 70, 70, 50, 70],
        [10, 60, 30, 60, 30, 80, 10, 80],
        [10, 10, 30, 10, 30, 30, 10, 30],
        [40, 10, 60, 10, 60, 30, 40, 30],
        [70, 10, 90, 10, 90, 30, 70, 30],
    ],
    dtype=np.float32,
)
_LABELS = np.array([[0, 0.9], [1, 0.8], [2, 0.7], [0, 0.9], [0, 0.8], [0, 0.1]])


def _paddle_outputs(with_warp):
    # Query order 2 -> 0 -> 1 on page 0 and 0 -> 1 -> 2 on page 1.
    roor = np.full((2, 3, 3), -10.0, dtype=np.float32)
    roor[0, 2, 0] = roor[0, 0, 1] = roor[0, 2, 1] = 10.0
    roor[1, 0, 1] = roor[1, 1, 2] = 10.0
    rel = np.zeros((2, 3, 3), dtype=np.float32)
    pred = [
        np.column_stack([_LABELS, _QUADS]).astype(np.float32),
        np.array([3, 3], dtype=np.int32),
        np.array([[2, 1, 2], [0, 1, 2]], dtype=np.int64),
        rel,
        roor,
    ]
    if with_warp:
        pred.append(
            np.array([[1, 0.9, 2, 0.1, 0.2], [3, 0.8, 4, 0.3, 0.4]], dtype=np.float32)
        )
    return pred


def _runner_predictor():
    return object.__new__(LayoutAnalysisRunnerPredictor)


@pytest.mark.parametrize("with_warp", [False, True])
def test_paddle_outputs_split_per_image(with_warp):
    results = _runner_predictor()._format_output(_paddle_outputs(with_warp))
    assert len(results) == 2
    first = results[0]
    # The detection box is the axis-aligned box of the quad.
    np.testing.assert_allclose(first["boxes"][0], [0, 0.9, 10, 10, 40, 40])
    np.testing.assert_array_equal(first["quad"], _QUADS[:3])
    np.testing.assert_array_equal(results[1]["qi"], [0, 1, 2])
    assert first["roor_logits"].shape == (3, 3)
    if with_warp:
        assert results[1]["warp_params"] == pytest.approx([3, 0.8, 4, 0.3, 0.4])
    else:
        assert all("warp_params" not in result for result in results)


@pytest.mark.parametrize(
    "index, value, message",
    [
        (0, np.zeros((6, 8), dtype=np.float32), "bbox_pred"),
        (5, np.zeros((2, 4), dtype=np.float32), "warp_params"),
    ],
)
def test_paddle_outputs_reject_unexpected_shapes(index, value, message):
    pred = _paddle_outputs(with_warp=True)
    pred[index] = value
    with pytest.raises(ValueError, match=message):
        _runner_predictor()._format_output(pred)


@pytest.mark.parametrize("with_warp", [False, True])
def test_paddle_outputs_without_quads_split_per_image(with_warp):
    pred = _paddle_outputs(with_warp)
    pred[0] = np.column_stack([_LABELS, _QUADS[:, [0, 1, 4, 5]]])
    results = _runner_predictor()._format_output(pred)
    assert len(results) == 2
    np.testing.assert_array_equal(results[1]["boxes"], pred[0][3:])
    np.testing.assert_array_equal(results[1]["qi"], [0, 1, 2])
    assert all("quad" not in result for result in results)
    assert all(("warp_params" in result) == with_warp for result in results)


def test_other_paddle_outputs_use_detection_format():
    boxes = np.array([[0, 0.9, 10, 10, 30, 30], [1, 0.8, 40, 40, 60, 60]])
    results = _runner_predictor()._format_output([boxes, np.array([1, 1])])
    assert [set(result) for result in results] == [{"boxes"}, {"boxes"}]
    np.testing.assert_array_equal(results[1]["boxes"], boxes[1:])


@pytest.mark.parametrize("with_order", [False, True])
def test_plain_detection_quads_split_per_image(with_order):
    bbox_pred = np.column_stack([_LABELS, _QUADS]).astype(np.float32)
    if with_order:
        # A pre-decoded reading order follows the quad.
        bbox_pred = np.column_stack([bbox_pred, [1, 0, 2, 2, 1, 0]])
    results = _runner_predictor()._format_output([bbox_pred, np.array([3, 3])])
    assert [set(result) for result in results] == [{"boxes", "quad"}] * 2
    np.testing.assert_array_equal(results[1]["quad"], _QUADS[3:])
    expected = [0, 0.9, 10, 10, 40, 40] + ([1] if with_order else [])
    np.testing.assert_allclose(results[0]["boxes"][0], expected)

    page = LayoutAnalysisProcess(
        labels=["text", "table", "image"], scale_size=[100, 100]
    )(
        results,
        [{"ori_img_size": (100, 100)}] * 2,
        threshold=0.5,
        layout_nms=False,
        layout_unclip_ratio=None,
        layout_merge_bboxes_mode=None,
        layout_shape_mode="auto",
        filter_overlap_boxes=True,
        skip_order_labels=[],
    )
    coordinates = [box["coordinate"] for box in page[1]]
    if with_order:
        assert coordinates == [[40, 10, 60, 30], [10, 10, 30, 30]]
    else:
        assert coordinates == [[10, 10, 30, 30], [40, 10, 60, 30]]


@pytest.mark.parametrize("with_quad", [False, True])
def test_paddle_outputs_with_masks_split_per_image(with_quad):
    pred = _paddle_outputs(with_warp=False)
    if not with_quad:
        pred[0] = np.array(
            [[0, 0.9, 10, 10, 40, 40]] * 3 + [[1, 0.8, 50, 50, 70, 70]] * 3,
            dtype=np.float32,
        )
    masks = np.arange(6 * 4 * 4, dtype=np.uint8).reshape(6, 4, 4)
    pred.insert(2, masks)
    results = _runner_predictor()._format_output(pred)
    assert len(results) == 2
    second = results[1]
    np.testing.assert_array_equal(second["masks"], masks[3:])
    np.testing.assert_array_equal(second["qi"], [0, 1, 2])
    assert second["rel_logits"].shape == (3, 3)
    assert all("warp_params" not in result for result in results)
    if with_quad:
        np.testing.assert_allclose(results[0]["boxes"][0], [0, 0.9, 10, 10, 40, 40])
        np.testing.assert_array_equal(second["quad"], _QUADS[3:])
    else:
        np.testing.assert_array_equal(second["boxes"], pred[0][3:])
        assert all("quad" not in result for result in results)


def test_paddle_outputs_reach_boxes_in_query_order():
    results = _runner_predictor()._format_output(_paddle_outputs(with_warp=True))
    page = LayoutAnalysisProcess(
        labels=["text", "table", "image"], scale_size=[100, 100]
    )(
        results,
        [{"ori_img_size": (100, 100)}] * 2,
        threshold=0.5,
        layout_nms=False,
        layout_unclip_ratio=None,
        layout_merge_bboxes_mode=None,
        layout_shape_mode="auto",
        filter_overlap_boxes=False,
        skip_order_labels=[],
    )
    # Page 0: both rows of query 2 are read before query 1. Page 1: the
    # low-score row is dropped.
    assert [box["cls_id"] for box in page[0]] == [0, 2, 1]
    np.testing.assert_array_equal(page[0][0]["quad"], _QUADS[0].reshape(4, 2))
    assert [box["coordinate"] for box in page[1]] == [
        [10, 10, 30, 30],
        [40, 10, 60, 30],
    ]


def test_duplicate_queries_share_a_rank():
    rng = np.random.default_rng(0)
    rel, roor = rng.normal(size=(2, 6, 6)) * 5
    qi = np.array([4, 1, 4, 3, 1])
    order = compute_reading_order(qi, rel, roor)
    unique = np.array([1, 3, 4])
    expected = _get_order_single_image(
        roor[np.ix_(unique, unique)], rel[np.ix_(unique, unique)]
    )
    np.testing.assert_array_equal(order, expected[[2, 0, 2, 1, 0]])


@pytest.mark.parametrize("count", [0, 1])
def test_reading_order_of_at_most_one_box(count):
    order = compute_reading_order(np.arange(count), np.zeros((3, 3)), np.zeros((3, 3)))
    np.testing.assert_array_equal(order, np.zeros(count))


@pytest.mark.parametrize("key", ["quad", "qi"])
def test_geometry_count_must_match_boxes(key):
    output = {
        "boxes": np.array([[0, 0.9, 10, 10, 30, 30], [0, 0.8, 40, 40, 60, 60]]),
        "quad": _QUADS[:2],
        "qi": np.array([0, 1]),
        "rel_logits": np.zeros((2, 2)),
        "roor_logits": np.zeros((2, 2)),
    }
    output[key] = output[key][:1]
    with pytest.raises(ValueError, match=key):
        LayoutAnalysisProcess(labels=["text"], scale_size=[100, 100])(
            [output],
            [{"ori_img_size": (100, 100)}],
            threshold=0.5,
            layout_nms=False,
            layout_unclip_ratio=None,
            layout_merge_bboxes_mode=None,
            layout_shape_mode="auto",
            filter_overlap_boxes=False,
            skip_order_labels=[],
        )
