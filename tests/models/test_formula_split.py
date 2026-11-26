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

import copy
from pathlib import Path

import cv2
import numpy as np
import pytest

from paddlex.inference.models.layout_analysis.formula_split import split_formula_boxes
from paddlex.inference.models.layout_analysis.processors import LayoutAnalysisProcess


def _page():
    image = np.full((220, 360, 3), 255, np.uint8)
    for baseline in [45, 105, 165]:
        cv2.putText(
            image,
            "x + y = 123",
            (35, baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
        )
    box = dict(
        label="display_formula",
        cls_id=0,
        score=0.9,
        coordinate=[20, 15, 300, 185],
        quad=[[20, 15], [300, 15], [300, 185], [20, 185]],
        order=1,
    )
    return image, box


def test_separated_rows_preserve_pixels_metadata_and_input():
    image, box = _page()
    original = copy.deepcopy(box)
    rows = split_formula_boxes(image, [box])
    assert len(rows) == 3
    assert box == original
    covered = np.zeros(image.shape[:2], np.uint8)
    for child in rows:
        assert child["score"] == box["score"]
        assert child["cls_id"] == box["cls_id"]
        x1, y1, x2, y2 = child["coordinate"]
        covered[y1 : y2 + 1, x1 : x2 + 1] = 1
    assert np.all(covered[image[:, :, 0] == 0])
    assert all(a["coordinate"][3] < b["coordinate"][1] for a, b in zip(rows, rows[1:]))


@pytest.mark.parametrize("label", ["text", "inline_formula", "table", "formula_number"])
def test_non_display_labels_are_untouched(label):
    image, box = _page()
    box["label"] = label
    assert split_formula_boxes(image, [box]) == [box]


def test_shared_matrix_bracket_prevents_splitting():
    image, box = _page()
    cv2.line(image, (25, 20), (25, 180), (0, 0, 0), 2)
    cv2.line(image, (25, 20), (33, 20), (0, 0, 0), 2)
    cv2.line(image, (25, 180), (33, 180), (0, 0, 0), 2)
    assert split_formula_boxes(image, [box]) == [box]


def test_fraction_and_detached_limit_are_not_split():
    image = np.full((150, 320, 3), 255, np.uint8)
    cv2.putText(image, "x + 123", (60, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.line(image, (55, 78), (200, 78), (0, 0, 0), 2)
    cv2.putText(
        image, "y + 456", (60, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2
    )
    box = dict(
        label="display_formula", coordinate=[20, 10, 280, 130], score=0.9, cls_id=0
    )
    assert split_formula_boxes(image, [box]) == [box]
    # A detached small upper limit must not become a separate formula.
    cv2.putText(image, "n", (110, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    assert split_formula_boxes(image, [box]) == [box]


def test_skewed_quads_are_split_and_mapped_back():
    image, box = _page()
    matrix = np.float32([[1, 0.18, 20], [0.08, 1, 15], [0, 0, 1]])
    image = cv2.warpPerspective(image, matrix, (450, 300), borderValue=(255, 255, 255))
    quad = cv2.perspectiveTransform(np.float32([box["quad"]]), matrix)[0]
    box["quad"] = quad.tolist()
    box["polygon_points"] = quad.tolist()
    box["coordinate"] = [*quad.min(0).tolist(), *quad.max(0).tolist()]
    rows = split_formula_boxes(image, [box])
    assert len(rows) == 3
    for row in rows:
        assert row["quad"] == row["polygon_points"]
        local = cv2.perspectiveTransform(
            np.float32([row["quad"]]), np.linalg.inv(matrix)
        )[0]
        assert np.ptp(local[:2, 1]) < 1
        assert np.ptp(local[2:, 1]) < 1


def test_empty_and_invalid_regions_fall_back():
    image, box = _page()
    assert split_formula_boxes(np.full_like(image, 255), [box]) == [box]
    box["quad"] = [[0, 0]] * 4
    assert split_formula_boxes(image, [box]) == [box]


def test_perspective_keeps_model_vertex_order():
    # For a long, thin formula, bottom-left can have smaller x+y than top-left.
    # Reordering that quad would rotate the crop and split the formula by column.
    image = np.full((70, 520, 3), 255, np.uint8)
    cv2.putText(
        image,
        "x + y = 123 + 456 + 789",
        (10, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
    )
    src = np.float32([[0, 0], [519, 0], [519, 69], [0, 69]])
    dst = np.float32([[150, 20], [550, 36], [500, 89], [30, 73]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, matrix, (600, 120), borderValue=(255, 255, 255))
    box = dict(
        label="display_formula",
        cls_id=0,
        score=0.9,
        coordinate=[30, 20, 550, 89],
        quad=dst.tolist(),
    )
    assert np.argmin(dst.sum(axis=1)) == 3
    assert split_formula_boxes(warped, [box]) == [box]


def test_postprocessor_switch_and_reading_order():
    image, box = _page()
    raw = dict(
        boxes=np.array([[0, 0.9, *box["coordinate"]]], np.float32),
        quad=np.float32([box["quad"]]).reshape(1, 8),
    )
    post = LayoutAnalysisProcess(labels=["display_formula"], scale_size=[800, 800])
    settings = dict(
        threshold=0.3,
        layout_nms=True,
        layout_shape_mode="auto",
        filter_overlap_boxes=True,
    )
    # Default mode does not need access to pixels.
    off = post([copy.deepcopy(raw)], [dict(ori_img_size=(360, 220))], **settings)[0]
    on = post(
        [copy.deepcopy(raw)],
        [dict(ori_img_size=(360, 220), ori_img=image)],
        layout_split_formula=True,
        **settings,
    )[0]
    assert len(off) == 1
    assert len(on) == 3
    assert [b["order"] for b in on] == [1, 2, 3]


@pytest.mark.parametrize("shadow_axis", [0, 1])
@pytest.mark.parametrize("noise", [False, True])
def test_shaded_noisy_rows_keep_their_ink(shadow_axis, noise):
    image, box = _page()
    original_ink = image[:, :, 0] == 0
    shade = np.linspace(0.25, 1.0, image.shape[shadow_axis])
    shade = shade[:, None, None] if shadow_axis == 0 else shade[None, :, None]
    image = (image.astype(float) * shade).astype(np.uint8)
    if noise:
        speckles = np.random.default_rng(42).random(image.shape[:2]) < 0.001
        image[speckles] = 0
    rows = split_formula_boxes(image, [box])
    assert len(rows) == 3
    covered = np.zeros(image.shape[:2], np.uint8)
    for child in rows:
        cv2.fillPoly(covered, [np.round(child["quad"]).astype(np.int32)], 1)
    assert np.all(covered[original_ink])


@pytest.mark.parametrize("gap", [6, 8, 10])
def test_small_fraction_denominators_are_not_separate_rows(gap):
    # Short bars and a large gap beneath them resemble inter-line whitespace.
    # This models the m/q = p/n regression found in a real historical scan.
    image = np.full((90, 120, 3), 255, np.uint8)
    for x, numerator, denominator in [(15, "m", "q"), (75, "p", "n")]:
        cv2.putText(
            image, numerator, (x, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
        )
        cv2.line(image, (x, 34), (x + 15, 34), (0, 0, 0), 1)
        cv2.putText(
            image,
            denominator,
            (x, 34 + gap + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )
    cv2.putText(image, "=", (43, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    box = dict(label="display_formula", coordinate=[5, 5, 110, 80])
    assert split_formula_boxes(image, [box]) == [box]


def test_detached_small_number_does_not_cancel_other_splits():
    image, box = _page()
    # A small equation-number-like component near the first line belongs to
    # that line, and must neither become a row nor veto the later line gaps.
    cv2.putText(image, "1", (270, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 0), 1)
    rows = split_formula_boxes(image, [box])
    assert len(rows) == 3
    x1, y1, x2, y2 = rows[0]["coordinate"]
    assert x1 <= 272 <= x2 and y1 <= 60 <= y2


@pytest.mark.parametrize("angle", [0, 15, -15, 30])
def test_resampled_small_scan_fraction_stays_whole(angle):
    # Crop of old_scans_math/4_pg186.pdf in allenai/olmOCR-bench, revision
    # 54a96a6fb6a2bd3b297e59869491db4d3625b711. Rotation/resampling can make a
    # two-pixel fraction bar three pixels thick, or join it to the numerator.
    image = cv2.imread(str(Path(__file__).parent / "assets/formula_fraction_scan.png"))
    h, w = image.shape[:2]
    quad = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    matrix = np.eye(3, dtype=np.float32)
    matrix[:2] = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    matrix[:2, 2] += 40
    warped = cv2.warpPerspective(
        image, matrix, (w + 80, h + 80), borderValue=(255, 255, 255)
    )
    quad = cv2.perspectiveTransform(quad[None], matrix)[0]
    box = dict(
        label="display_formula",
        coordinate=[*quad.min(0).tolist(), *quad.max(0).tolist()],
        quad=quad.tolist(),
    )
    assert split_formula_boxes(warped, [box]) == [box]


def _detected_rows():
    import json

    assets = Path(__file__).parent / "assets"
    image = cv2.imread(str(assets / "formula_detected_rows.png"))
    # Crop/detections from old_scans_math/3_pg204 at the same pinned benchmark
    # revision as formula_fraction_scan.png. The model predicts a parent and
    # four stronger rows, with gaps below the image-only splitting threshold.
    boxes = json.loads((assets / "formula_detected_rows.json").read_text())
    parent = max(boxes, key=lambda b: b["coordinate"][3] - b["coordinate"][1])
    rows = sorted(
        [b for b in boxes if b is not parent], key=lambda b: b["coordinate"][1]
    )
    return image, parent, rows


@pytest.mark.parametrize("warp", ["clean", "rotate", "perspective"])
def test_detector_rows_survive_large_filter_with_switch(warp):
    image, parent, rows = _detected_rows()
    if warp != "clean":
        h, w = image.shape[:2]
        if warp == "rotate":
            matrix = np.eye(3, dtype=np.float32)
            matrix[:2] = cv2.getRotationMatrix2D((w / 2, h / 2), 15, 1)
            matrix[:2, 2] += 40
        else:
            matrix = cv2.getPerspectiveTransform(
                np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]),
                np.float32([[50, 20], [w + 30, 35], [w + 10, h + 30], [20, h + 10]]),
            )
        image = cv2.warpPerspective(
            image, matrix, (w + 100, h + 100), borderValue=(255, 255, 255)
        )
        for box in [parent, *rows]:
            quad = cv2.perspectiveTransform(np.float32([box["quad"]]), matrix)[0]
            box["quad"] = quad.tolist()
            box["coordinate"] = [*quad.min(0).tolist(), *quad.max(0).tolist()]
    # Shuffle rows to make sure output follows the parent and visual row order.
    detections = [rows[2], parent, rows[0], rows[3], rows[1]]
    raw = dict(
        boxes=np.array(
            [[0, b["score"], *b["coordinate"]] for b in detections], np.float32
        ),
        quad=np.float32([b["quad"] for b in detections]).reshape(-1, 8),
    )
    post = LayoutAnalysisProcess(labels=["display_formula"], scale_size=[800, 800])
    settings = dict(
        threshold=0.3,
        layout_nms=True,
        layout_merge_bboxes_mode={0: "large"},
        layout_shape_mode="auto",
        filter_overlap_boxes=True,
    )
    h, w = image.shape[:2]
    off = post([copy.deepcopy(raw)], [dict(ori_img_size=(w, h))], **settings)[0]
    on = post(
        [copy.deepcopy(raw)],
        [dict(ori_img_size=(w, h), ori_img=image)],
        layout_split_formula=True,
        **settings,
    )[0]
    assert len(off) == 1
    assert len(on) == 4
    assert [b["order"] for b in on] == [1, 2, 3, 4]
    assert [b["score"] for b in on] == pytest.approx([b["score"] for b in rows])
    assert np.allclose([b["quad"] for b in on], [b["quad"] for b in rows])


@pytest.mark.parametrize(
    "invalid",
    ["missing_row", "weak_scores", "overlapping", "cropped_ink", "shared_bracket"],
)
def test_incomplete_or_unsafe_detector_rows_fall_back(invalid):
    image, parent, rows = _detected_rows()
    if invalid == "missing_row":
        rows.pop(1)
    elif invalid == "weak_scores":
        for row in rows:
            row["score"] = parent["score"]
    elif invalid == "overlapping":
        rows.append(copy.deepcopy(rows[0]))
    elif invalid == "cropped_ink":
        for point in rows[1]["quad"][1:3]:
            point[0] -= 30
    else:
        x, y, xx, yy = parent["coordinate"]
        cv2.line(image, (x + 1, y + 2), (x + 1, yy - 2), (0, 0, 0), 2)
    original = copy.deepcopy((parent, rows))
    assert split_formula_boxes(image, [parent], rows) == [parent]
    assert (parent, rows) == original


def test_detector_fraction_candidates_do_not_override_structure_guard():
    image = cv2.imread(str(Path(__file__).parent / "assets/formula_fraction_scan.png"))
    h, w = image.shape[:2]
    parent = dict(
        label="display_formula", cls_id=0, score=0.4, coordinate=[0, 0, w - 1, h - 1]
    )
    rows = [
        dict(label="display_formula", cls_id=0, score=0.9, coordinate=[0, a, w - 1, b])
        for a, b in [(0, 22), (24, h - 1)]
    ]
    assert split_formula_boxes(image, [parent], rows) == [parent]


def test_detector_candidates_already_present_are_not_restored_twice():
    image, parent, rows = _detected_rows()
    # With filtering disabled the caller may intentionally keep both levels.
    boxes = [parent, *rows]
    assert split_formula_boxes(image, boxes, boxes) == boxes


# ------------------------------------------- row height & fraction guards ----


def _ink_coverage(image, boxes, shape_key="quad"):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    original_ink = gray < 128
    covered = np.zeros(image.shape[:2], np.uint8)
    for box in boxes:
        if shape_key:
            assert shape_key in box
            cv2.fillPoly(covered, [np.round(box[shape_key]).astype(np.int32)], 1)
        else:
            assert len(box["coordinate"]) == 4
            x1, y1, x2, y2 = box["coordinate"]
            covered[y1 : y2 + 1, x1 : x2 + 1] = 1
    total = int(np.count_nonzero(original_ink))
    return float(np.count_nonzero(covered[original_ink])) / max(1, total)


@pytest.mark.parametrize("text", ["A + B = C + D", "1 + 2 = 3 + 4", "x + z = w + v"])
def test_uniform_height_rows_split_cleanly(text):
    image = np.full((220, 360, 3), 255, np.uint8)
    for baseline in [45, 105, 165]:
        cv2.putText(
            image, text, (35, baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2
        )
    box = dict(
        label="display_formula",
        cls_id=0,
        score=0.9,
        coordinate=[20, 15, 310, 185],
        quad=[[20, 15], [310, 15], [310, 185], [20, 185]],
    )
    rows = split_formula_boxes(image, [box])
    assert len(rows) == 3
    assert _ink_coverage(image, rows, "quad") == 1.0


def test_mixed_row_heights_do_not_merge_plain_middle_row():
    image = np.full((220, 360, 3), 255, np.uint8)
    cv2.putText(
        image, "x + y = 123", (35, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2
    )
    cv2.putText(
        image, "A + B = C + D", (35, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2
    )
    cv2.putText(
        image, "x + y = 456", (35, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2
    )
    box = dict(
        label="display_formula",
        cls_id=0,
        score=0.9,
        coordinate=[20, 15, 310, 185],
        quad=[[20, 15], [310, 15], [310, 185], [20, 185]],
    )
    rows = split_formula_boxes(image, [box])
    assert len(rows) == 3
    assert _ink_coverage(image, rows, "quad") == 1.0


def test_touching_fraction_bar_stays_whole():
    image = np.full((150, 320, 3), 255, np.uint8)
    # Descender of 'y' touches the fraction bar at y=78.
    cv2.putText(
        image, "x + y + 123", (60, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2
    )
    cv2.line(image, (55, 78), (225, 78), (0, 0, 0), 2)
    cv2.putText(
        image, "a + b = 456", (60, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2
    )
    box = dict(
        label="display_formula", coordinate=[20, 10, 280, 135], score=0.9, cls_id=0
    )
    assert split_formula_boxes(image, [box]) == [box]


def test_single_char_numerator_fraction_stays_whole_and_splits_from_other_line():
    image = np.full((220, 360, 3), 255, np.uint8)
    cv2.putText(
        image, "x + y = 123", (35, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2
    )
    cv2.putText(image, "1", (115, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.line(image, (35, 118), (225, 118), (0, 0, 0), 2)
    cv2.putText(
        image, "a + b = 456", (35, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2
    )
    # Standalone fraction region stays intact.
    frac_box = dict(
        label="display_formula", coordinate=[20, 75, 260, 180], score=0.9, cls_id=0
    )
    assert split_formula_boxes(image, [frac_box]) == [frac_box]
    # Two-line region splits into line 1 and the intact fraction (2 rows total).
    full_box = dict(
        label="display_formula", coordinate=[20, 15, 280, 185], score=0.9, cls_id=0
    )
    rows = split_formula_boxes(image, [full_box])
    assert len(rows) == 2
    assert rows[0]["coordinate"][3] < 80
    assert rows[1]["coordinate"][1] <= 86 and rows[1]["coordinate"][3] >= 155


# ----------------------------------------- tilted & curved formula cases ----


def _warp_columns(image, delta):
    h, w = image.shape[:2]
    xs = np.arange(w, dtype=np.float32)
    map_x = np.broadcast_to(xs[None, :], (h, w)).astype(np.float32)
    map_y = (np.arange(h, dtype=np.float32)[:, None] - delta[None, :]).astype(
        np.float32
    )
    return cv2.remap(
        image,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


@pytest.mark.parametrize("angle", [-18, -12, -6, 6, 12, 18])
@pytest.mark.parametrize("with_quad", [False, True])
def test_tilted_rows_inside_axis_aligned_box(angle, with_quad):
    image = np.full((300, 440, 3), 255, np.uint8)
    for baseline in [85, 145, 205]:
        cv2.putText(
            image,
            "x + y = 123 + a",
            (75, baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 0),
            2,
        )
    matrix = cv2.getRotationMatrix2D((220, 150), angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (440, 300), borderValue=(255, 255, 255))
    ys, xs = np.nonzero(cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY) < 200)
    x1, y1 = int(xs.min()) - 6, int(ys.min()) - 6
    x2, y2 = int(xs.max()) + 6, int(ys.max()) + 6
    box = dict(
        label="display_formula",
        cls_id=0,
        score=0.9,
        coordinate=[x1, y1, x2, y2],
    )
    if with_quad:
        box["quad"] = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    rows = split_formula_boxes(rotated, [box])
    assert len(rows) == 3
    assert _ink_coverage(rotated, rows, "quad" if with_quad else None) == 1.0
    # Rows must be distinct and in reading order, not copies of the parent.
    centers_y = [(r["coordinate"][1] + r["coordinate"][3]) / 2 for r in rows]
    assert centers_y[0] < centers_y[1] < centers_y[2]
    assert len({tuple(r["coordinate"]) for r in rows}) == 3
    if not with_quad:
        from paddlex.inference.models.layout_analysis.processors import filter_boxes

        # Tilted rows have overlapping AABBs; all must survive filter_boxes.
        assert len(filter_boxes(rows, "rect")) == 3
    if with_quad:
        # Child quads should follow the row tilt and not overlap each other.
        masks = []
        for row in rows:
            m = np.zeros(rotated.shape[:2], np.uint8)
            cv2.fillPoly(m, [np.round(row["quad"]).astype(np.int32)], 1)
            masks.append(m > 0)
        assert not np.any(masks[0] & masks[1])
        assert not np.any(masks[1] & masks[2])


@pytest.mark.parametrize(
    "curve_type",
    [
        "arch",
        "bowl",
        "s_wave",
        "book_spine_curl",
        "tilt_plus_arch_shaded",
        "tilt_plus_wave",
    ],
)
def test_curved_multiline_formulas(curve_type):
    h, w = 280, 420
    base = np.full((h, w, 3), 255, np.uint8)
    row_layers = []
    for baseline in [75, 135, 195]:
        layer = np.full((h, w, 3), 255, np.uint8)
        cv2.putText(
            layer,
            "x + y = 123 + a",
            (45, baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 0),
            2,
        )
        base = np.minimum(base, layer)
        row_layers.append(layer)
    xs = np.arange(w, dtype=np.float32)
    u = (2.0 * xs - w) / w
    if curve_type == "arch":
        delta = 18.0 * (u**2 - 1.0 / 3.0)
    elif curve_type == "bowl":
        delta = -18.0 * (u**2 - 1.0 / 3.0)
    elif curve_type == "s_wave":
        delta = 24.0 * (u**3 - 0.6 * u)
    elif curve_type == "book_spine_curl":
        # Strong curl near the left book spine, flattening toward the right.
        t = np.clip(1.0 - xs / 210.0, 0.0, 1.0)
        delta = 24.0 * (t**2)
        delta -= delta.mean()
    elif curve_type == "tilt_plus_arch_shaded":
        delta = 14.0 * u - 16.0 * (u**2 - 1.0 / 3.0)
    else:
        delta = -12.0 * u + 8.0 * (u**2 - 1.0 / 3.0) + 18.0 * (u**3 - 0.6 * u)

    warped = _warp_columns(base, delta)
    orig_ink = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) < 128
    row_inks = [
        cv2.cvtColor(_warp_columns(layer, delta), cv2.COLOR_BGR2GRAY) < 128
        for layer in row_layers
    ]
    if curve_type == "tilt_plus_arch_shaded":
        shade = np.linspace(0.35, 1.0, w)[None, :, None]
        warped = (warped.astype(float) * shade).astype(np.uint8)
    ys_i, xs_i = np.nonzero(orig_ink)
    x1, y1 = int(xs_i.min()) - 6, int(ys_i.min()) - 6
    x2, y2 = int(xs_i.max()) + 6, int(ys_i.max()) + 6
    box = dict(
        label="display_formula",
        cls_id=0,
        score=0.9,
        coordinate=[x1, y1, x2, y2],
        quad=[[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
    )
    rows = split_formula_boxes(warped, [box])
    assert len(rows) == 3
    covered = np.zeros(warped.shape[:2], np.uint8)
    for row in rows:
        cv2.fillPoly(covered, [np.round(row["quad"]).astype(np.int32)], 1)
    assert np.all(covered[orig_ink])
    centers_y = [np.mean([p[1] for p in r["quad"]]) for r in rows]
    assert centers_y[0] < centers_y[1] < centers_y[2]
    for i, row in enumerate(rows):
        mask = np.zeros(warped.shape[:2], np.uint8)
        cv2.fillPoly(mask, [np.round(row["quad"]).astype(np.int32)], 1)
        # Covers 100% of its own row's ink and 0% of neighboring rows' ink.
        assert np.all(mask[row_inks[i]] > 0)
        for j, other_ink in enumerate(row_inks):
            if j != i:
                assert not np.any(mask[other_ink] > 0)


@pytest.mark.parametrize(
    "polygon",
    [
        # Tilted 4-point segmentation polygon.
        [[30, 30], [300, 10], [310, 170], [40, 190]],
        # Concave L-shape.
        [[20, 15], [300, 15], [300, 70], [110, 70], [110, 185], [20, 185]],
    ],
)
def test_segmentation_polygon_without_quad_stays_intact(polygon):
    # Only PP-DocLayoutV4 quads are split; other polygons keep the box whole.
    image, box = _page()
    box.pop("quad", None)
    box["polygon_points"] = polygon
    assert split_formula_boxes(image, [box]) == [box]


@pytest.mark.parametrize(
    "case",
    [
        "tilted_single_line",
        "curved_single_line",
        "tilted_fraction",
        "curved_fraction",
        "curved_matrix_bracket",
    ],
)
def test_tilted_and_curved_unsplittable_formulas_stay_intact(case):
    h, w = 240, 400
    base = np.full((h, w, 3), 255, np.uint8)
    xs = np.arange(w, dtype=np.float32)
    u = (2.0 * xs - w) / w
    if case == "tilted_single_line":
        cv2.putText(
            base,
            "x + y = 123 + 456",
            (40, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 0),
            2,
        )
        delta = 18.0 * u
    elif case == "curved_single_line":
        cv2.putText(
            base,
            "x + y = 123 + 456",
            (40, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 0),
            2,
        )
        delta = 18.0 * (u**2 - 1.0 / 3.0)
    elif case in ("tilted_fraction", "curved_fraction"):
        cv2.putText(
            base, "x + 123", (80, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2
        )
        cv2.line(base, (70, 106), (250, 106), (0, 0, 0), 2)
        cv2.putText(
            base, "y + 456", (80, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2
        )
        delta = 15.0 * u if case == "tilted_fraction" else 16.0 * (u**2 - 1.0 / 3.0)
    else:
        for baseline in [65, 125, 185]:
            cv2.putText(
                base,
                "x + y = 123",
                (70, baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 0),
                2,
            )
        cv2.line(base, (55, 40), (55, 195), (0, 0, 0), 2)
        cv2.line(base, (55, 40), (65, 40), (0, 0, 0), 2)
        cv2.line(base, (55, 195), (65, 195), (0, 0, 0), 2)
        delta = 18.0 * (u**2 - 1.0 / 3.0)

    warped = _warp_columns(base, delta)
    box = dict(
        label="display_formula",
        cls_id=0,
        score=0.9,
        coordinate=[20, 10, 380, 230],
    )
    assert split_formula_boxes(warped, [box]) == [box]


@pytest.mark.parametrize("arch_px", [10.0, -10.0])
def test_curved_real_scan_fraction_stays_whole(arch_px):
    image = cv2.imread(str(Path(__file__).parent / "assets/formula_fraction_scan.png"))
    assert image is not None, "missing assets/formula_fraction_scan.png"
    h, w = image.shape[:2]
    padded = np.full((h + 40, w + 40, 3), 255, np.uint8)
    padded[20 : 20 + h, 20 : 20 + w] = image
    pw = padded.shape[1]
    u = (2.0 * np.arange(pw, dtype=np.float32) - pw) / pw
    warped = _warp_columns(padded, arch_px * (u**2 - 1.0 / 3.0))
    box = dict(
        label="display_formula",
        cls_id=0,
        score=0.9,
        coordinate=[0, 0, pw - 1, padded.shape[0] - 1],
    )
    assert split_formula_boxes(warped, [box]) == [box]


def test_tilted_detector_candidates_restore_rows_inside_axis_aligned_parent():
    image, parent, rows = _detected_rows()
    h, w = image.shape[:2]
    matrix = np.eye(3, dtype=np.float32)
    matrix[:2] = cv2.getRotationMatrix2D((w / 2, h / 2), 10, 1.0)
    matrix[:2, 2] += 40
    warped = cv2.warpPerspective(
        image, matrix, (w + 100, h + 100), borderValue=(255, 255, 255)
    )
    # Parent is an axis-aligned box (no quad, e.g. rect mode or loose box),
    # while detector row candidates carry tilted quads matching the scan angle.
    warped_parent_quad = cv2.perspectiveTransform(np.float32([parent["quad"]]), matrix)[
        0
    ]
    p_x1, p_y1 = np.floor(warped_parent_quad.min(0)).astype(int).tolist()
    p_x2, p_y2 = np.ceil(warped_parent_quad.max(0)).astype(int).tolist()
    aa_parent = dict(
        cls_id=parent["cls_id"],
        label=parent["label"],
        score=parent["score"],
        coordinate=[p_x1, p_y1, p_x2, p_y2],
    )
    tilted_rows = []
    for row in rows:
        tr = copy.deepcopy(row)
        q = cv2.perspectiveTransform(np.float32([row["quad"]]), matrix)[0]
        tr["quad"] = q.tolist()
        tr["coordinate"] = [*q.min(0).tolist(), *q.max(0).tolist()]
        tilted_rows.append(tr)
    out = split_formula_boxes(warped, [aa_parent], tilted_rows)
    assert len(out) == 4
    assert [b["score"] for b in out] == pytest.approx([b["score"] for b in rows])


# ----------------------- formula_number, rect mode, overlap, serving/CLI ----


@pytest.mark.parametrize("skewed", [False, True])
def test_formula_numbers_interleaved_after_split_and_markdown_pairing(skewed):
    from paddlex.inference.common.result.converter import MarkdownConverter
    from paddlex.inference.pipelines.paddleocr_vl.result import PaddleOCRVLBlock

    image, parent = _page()
    fn_boxes = [
        dict(
            label="formula_number",
            cls_id=1,
            score=0.95,
            coordinate=[315, y1, 345, y2],
            quad=[[315, y1], [345, y1], [345, y2], [315, y2]],
        )
        for y1, y2 in [(25, 50), (85, 110), (145, 170)]
    ]
    if skewed:
        matrix = np.float32([[1, 0.12, 15], [0.06, 1, 10], [0, 0, 1]])
        image = cv2.warpPerspective(
            image, matrix, (460, 280), borderValue=(255, 255, 255)
        )
        for b in [parent, *fn_boxes]:
            q = cv2.perspectiveTransform(np.float32([b["quad"]]), matrix)[0]
            b["quad"] = q.tolist()
            b["coordinate"] = [*q.min(0).tolist(), *q.max(0).tolist()]

    # Original reading order has parent followed by all three formula numbers.
    detections = [parent, *fn_boxes]
    raw = dict(
        boxes=np.array(
            [[b["cls_id"], b["score"], *b["coordinate"]] for b in detections],
            np.float32,
        ),
        quad=np.float32([b["quad"] for b in detections]).reshape(-1, 8),
    )
    post = LayoutAnalysisProcess(
        labels=["display_formula", "formula_number"], scale_size=[800, 800]
    )
    h, w = image.shape[:2]
    out = post(
        [raw],
        [dict(ori_img_size=(w, h), ori_img=image)],
        threshold=0.3,
        layout_nms=True,
        layout_shape_mode="auto",
        filter_overlap_boxes=True,
        layout_split_formula=True,
    )[0]
    assert [b["label"] for b in out] == [
        "display_formula",
        "formula_number",
        "display_formula",
        "formula_number",
        "display_formula",
        "formula_number",
    ]
    assert [b["order"] for b in out] == [1, 2, 3, 4, 5, 6]
    # Each formula number must follow the split row it is aligned with.
    for i, (row, number) in enumerate(zip(out[0::2], out[1::2])):
        num_cy = 0.5 * (number["coordinate"][1] + number["coordinate"][3])
        assert row["coordinate"][1] <= num_cy <= row["coordinate"][3]
        assert number["coordinate"] == pytest.approx(fn_boxes[i]["coordinate"], abs=1)
    # Verify MarkdownConverter pairs each split row with its own formula number.
    blocks = []
    num_idx = 1
    for b in out:
        if b["label"] == "display_formula":
            content = f"$$ x_{num_idx} = {num_idx} $$"
        else:
            content = f"({num_idx})"
            num_idx += 1
        blocks.append(PaddleOCRVLBlock(b["label"], b["coordinate"], content))
    md = MarkdownConverter.convert(
        blocks,
        handle_funcs_dict={"display_formula": lambda b: b.content},
        show_formula_number=True,
    )["markdown_texts"]
    assert "\\tag*{(1)}" in md and "\\tag*{(2)}" in md and "\\tag*{(3)}" in md


def test_two_column_formula_number_not_stolen():
    image, left_parent = _page()
    # Expand canvas to a two-column page; place a text column and a right-column
    # formula_number that must NOT be pulled into the left-column formula.
    wide = np.full((220, 760, 3), 255, np.uint8)
    wide[:, :360] = image
    mid_text = dict(
        label="text",
        cls_id=2,
        score=0.95,
        coordinate=[380, 10, 620, 200],
    )
    right_fn = dict(
        label="formula_number",
        cls_id=1,
        score=0.95,
        coordinate=[660, 85, 700, 110],
    )
    out = split_formula_boxes(wide, [left_parent, mid_text, right_fn])
    assert [b["label"] for b in out] == [
        "display_formula",
        "display_formula",
        "display_formula",
        "text",
        "formula_number",
    ]


def test_detector_rows_survive_large_filter_in_rect_mode():
    image, parent, rows = _detected_rows()
    detections = [rows[2], parent, rows[0], rows[3], rows[1]]
    raw = dict(
        boxes=np.array(
            [[0, b["score"], *b["coordinate"]] for b in detections], np.float32
        ),
        quad=np.float32([b["quad"] for b in detections]).reshape(-1, 8),
    )
    post = LayoutAnalysisProcess(labels=["display_formula"], scale_size=[800, 800])
    h, w = image.shape[:2]
    out = post(
        [raw],
        [dict(ori_img_size=(w, h), ori_img=image)],
        threshold=0.3,
        layout_nms=True,
        layout_merge_bboxes_mode={0: "large"},
        layout_shape_mode="rect",
        filter_overlap_boxes=True,
        layout_split_formula=True,
    )[0]
    assert len(out) == 4
    assert [b["score"] for b in out] == pytest.approx([b["score"] for b in rows])


def test_deferred_overlap_filter_keeps_dropped_parent_whole():
    from paddlex.inference.models.layout_analysis.processors import filter_boxes

    image, parent = _page()
    # A larger text box overlaps the parent formula by > 0.7 of the formula
    # area, so filter_boxes drops the formula, while each split row overlaps it
    # by < 0.7. Deferred filtering must not resurrect the rows.
    text = dict(
        label="text",
        cls_id=1,
        score=0.8,
        coordinate=[100, 0, 360, 220],
        quad=[[100, 0], [360, 0], [360, 220], [100, 220]],
    )
    raw = dict(
        boxes=np.array(
            [
                [0, parent["score"], *parent["coordinate"]],
                [1, text["score"], *text["coordinate"]],
            ],
            np.float32,
        ),
        quad=np.float32([parent["quad"], text["quad"]]).reshape(-1, 8),
    )
    post = LayoutAnalysisProcess(
        labels=["display_formula", "text"], scale_size=[800, 800]
    )
    settings = dict(threshold=0.3, layout_shape_mode="auto", layout_split_formula=True)
    data = [dict(ori_img_size=(360, 220), ori_img=image)]
    filtered_first = post(
        [copy.deepcopy(raw)], data, filter_overlap_boxes=True, **settings
    )[0]
    deferred = post([copy.deepcopy(raw)], data, filter_overlap_boxes=False, **settings)[
        0
    ]
    after_pipeline_filter = filter_boxes(deferred, "auto")
    assert [b["label"] for b in filtered_first] == ["text"]
    assert [b["label"] for b in after_pipeline_filter] == ["text"]


def test_deferred_overlap_filter_does_not_drop_split_formula_rows():
    from paddlex.inference.models.layout_analysis.processors import filter_boxes

    image, parent = _page()
    # A smaller overlapping text box covering lines 1-2 of the parent formula.
    # Without overlap guarding, splitting the parent shrinks each child below
    # the text box's area and downstream PaddleOCR-VL filter_boxes deletes them.
    smaller_text = dict(
        label="text",
        cls_id=1,
        score=0.8,
        coordinate=[25, 20, 295, 130],
        quad=[[25, 20], [295, 20], [295, 130], [25, 130]],
    )
    raw = dict(
        boxes=np.array(
            [
                [0, parent["score"], *parent["coordinate"]],
                [1, smaller_text["score"], *smaller_text["coordinate"]],
            ],
            np.float32,
        ),
        quad=np.float32([parent["quad"], smaller_text["quad"]]).reshape(-1, 8),
    )
    post = LayoutAnalysisProcess(
        labels=["display_formula", "text"], scale_size=[800, 800]
    )
    h, w = image.shape[:2]
    # Simulate PaddleOCR-VL: filter_overlap_boxes=False in layout model,
    # followed by filter_boxes in pipeline page preparation.
    det_out = post(
        [raw],
        [dict(ori_img_size=(w, h), ori_img=image)],
        threshold=0.3,
        layout_nms=True,
        layout_shape_mode="auto",
        filter_overlap_boxes=False,
        layout_split_formula=True,
    )[0]
    after_pipeline_filter = filter_boxes(det_out, "auto")
    assert len(after_pipeline_filter) == 3
    assert all(b["label"] == "display_formula" for b in after_pipeline_filter)


def test_layout_split_formula_only_enabled_for_pp_doclayout_v4():
    from types import SimpleNamespace

    from paddlex.inference.models.layout_analysis.predictor import (
        _resolve_layout_split_formula,
    )

    v4 = SimpleNamespace(model_name="PP-DocLayoutV4")
    v3 = SimpleNamespace(model_name="PP-DocLayoutV3")
    assert _resolve_layout_split_formula(v4, True) is True
    assert _resolve_layout_split_formula(v4, None) is False
    assert _resolve_layout_split_formula(v3, None) is False
    assert _resolve_layout_split_formula(v3, True) is False
    assert v3._split_formula_warned is True


def test_runner_predictor_layout_split_formula_falls_back_to_config():
    from paddlex.inference.models.layout_analysis.predictor import (
        LayoutAnalysisRunnerPredictor,
    )

    def resolve(init_value, config_value):
        predictor = LayoutAnalysisRunnerPredictor.__new__(LayoutAnalysisRunnerPredictor)
        predictor.threshold = 0.5
        predictor.layout_nms = None
        predictor.layout_unclip_ratio = None
        predictor.layout_merge_bboxes_mode = None
        predictor.layout_split_formula = init_value
        # ``config`` is a read-only alias of ``_model_config``.
        predictor._model_config = {"label_list": ["display_formula"]}
        if config_value is not None:
            predictor._model_config["layout_split_formula"] = config_value
        predictor.build_postprocess()
        return predictor.layout_split_formula

    assert resolve(None, None) is False
    assert resolve(None, True) is True
    assert resolve(False, True) is False
    assert resolve(True, False) is True


def test_serving_schema_cli_and_predictor_expose_layout_split_formula():
    import ast
    import inspect

    from paddlex.inference.models.layout_analysis.predictor import (
        LayoutAnalysisRunnerPredictor,
        LayoutAnalysisTransformersPredictor,
    )
    from paddlex.utils.deps import is_serving_plugin_available
    from paddlex.utils.pipeline_arguments import PIPELINE_ARGUMENTS

    if is_serving_plugin_available():
        from paddlex.inference.serving.schemas.paddleocr_vl import InferRequest

        req = InferRequest(file="dummy.png", layoutSplitFormula=True)
        assert req.layoutSplitFormula is True
    else:
        schema_src = (
            Path(__file__).parents[2]
            / "paddlex/inference/serving/schemas/paddleocr_vl.py"
        ).read_text()
        tree = ast.parse(schema_src)
        infer_cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "InferRequest"
        )
        field_names = {
            stmt.target.id
            for stmt in infer_cls.body
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
        }
        assert "layoutSplitFormula" in field_names

    # The serving app must forward the request field to pipeline.infer().
    app_src = (
        Path(__file__).parents[2]
        / "paddlex/inference/serving/basic_serving/_pipeline_apps/paddleocr_vl.py"
    ).read_text()
    forwarded = [
        kw.value
        for node in ast.walk(ast.parse(app_src))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "infer"
        for kw in node.keywords
        if kw.arg == "layout_split_formula"
    ]
    assert len(forwarded) == 1
    assert isinstance(forwarded[0], ast.Attribute)
    assert isinstance(forwarded[0].value, ast.Name)
    assert (forwarded[0].value.id, forwarded[0].attr) == (
        "request",
        "layoutSplitFormula",
    )

    vl_args = {arg["name"] for arg in PIPELINE_ARGUMENTS["PaddleOCR-VL"]}
    assert "--layout_split_formula" in vl_args
    assert (
        "layout_split_formula"
        in inspect.signature(LayoutAnalysisRunnerPredictor.__init__).parameters
    )
    assert (
        "layout_split_formula"
        in inspect.signature(LayoutAnalysisTransformersPredictor.__init__).parameters
    )
