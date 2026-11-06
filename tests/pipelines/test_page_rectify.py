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

"""page_rectify must stay identical to LayoutModel's tools/rectify_page.py.

Run: python -m pytest tests/pipelines/test_page_rectify.py -q
The reference comparison runs only when LAYOUTMODEL_RECTIFY_REFERENCE points
to that rectify_page.py.
"""

import importlib.util
import os

import cv2
import numpy as np
import pytest

from paddlex.inference.pipelines.components.common.crop_image_regions import CropByBoxes
from paddlex.inference.pipelines.components.common.page_rectify import (
    build_page_homography,
    keystone_deg,
    plan_page_correction,
    split_roll,
)
from paddlex.inference.pipelines.paddleocr_vl.uilts import tokenize_figure_of_table

REF = os.environ.get("LAYOUTMODEL_RECTIFY_REFERENCE")


@pytest.fixture(scope="module")
def ref():
    if not REF or not os.path.exists(REF):
        pytest.skip("reference rectify_page.py not available")
    spec = importlib.util.spec_from_file_location("rectify_page", REF)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _random_params(rng):
    return (
        rng.uniform(-180, 180),
        1.0,
        rng.uniform(-25, 25),
        rng.uniform(-1.2, 1.2),
        rng.uniform(-1.2, 1.2),
    )


def test_matches_reference_up_to_framing(ref):
    """Same map as build_rectify modulo the translation/uniform scale of _fit."""
    rng = np.random.default_rng(0)
    for _ in range(300):
        h, w = int(rng.integers(300, 5000)), int(rng.integers(300, 5000))
        wp = _random_params(rng)
        h_ref, _ = ref.build_rectify((h, w), wp, max_scale=1e9)
        h_ours = build_page_homography((h, w), wp)
        pts = rng.uniform([0, 0], [w, h], size=(64, 1, 2))
        a = cv2.perspectiveTransform(pts, h_ref).reshape(-1, 2)
        b = cv2.perspectiveTransform(pts, h_ours).reshape(-1, 2)
        assert np.abs((a - a.mean(0)) - (b - b.mean(0))).max() < 1e-6


def test_identity_below_thresholds():
    assert build_page_homography((800, 600), (0.1, 1.0, 0.1, 0.001, -0.001)) is None


def _linear_part(H, w, h):
    """Jacobian of H at the image centre."""
    p = H @ [w / 2.0, h / 2.0, 1.0]
    z = p[2]
    return (H[:2, :2] * z - np.outer(p[:2], H[2, :2])) / z**2


@pytest.mark.parametrize("roll", [90.0, -90.0, 180.0, 270.0, -270.0])
@pytest.mark.parametrize("hw", [(959, 1600), (1600, 959), (4096, 3072)])
def test_quarter_turn_is_rigid(roll, hw):
    """A pure quarter turn of a non-square page must be a rotation, not a squash.

    Independent of the reference (which shares the code): unsquashing a rotated
    page with the input's own W/H flattens it by (W/H)^2 across its lines.
    """
    h, w = hw
    H = build_page_homography((h, w), (roll, 1.0, 0.0, 0.0, 0.0))
    J = _linear_part(H, w, h)
    a = np.radians(roll)
    np.testing.assert_allclose(
        J, [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], atol=1e-9
    )
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    ow, oh = out.max(0) - out.min(0)
    expect = (h, w) if int(round(roll / 90)) % 2 else (w, h)
    np.testing.assert_allclose((ow, oh), expect, atol=1e-6)


def test_quarter_turn_thin_block_not_downscaled():
    """A thin vertical line on a page rotated 90 deg keeps its pixel size."""
    h, w = 959, 1600
    img = np.full((h, w, 3), 255, np.uint8)
    quad = [
        [700, 300],
        [726, 300],
        [726, 507],
        [700, 507],
    ]  # 26 x 207, like a rotated title
    box = {
        "cls_id": 0,
        "label": "text",
        "coordinate": [700, 300, 726, 507],
        "quad": quad,
    }
    H = build_page_homography((h, w), (89.84, 1.0, 0.0, 0.0, 0.0))
    crop = CropByBoxes()(img, [box], "auto", page_homography=H)[0]["img"]
    ch, cw = crop.shape[:2]
    assert abs(cw - 207) <= 3 and abs(ch - 26) <= 3, crop.shape


def test_near_quarter_turn_keeps_aspect():
    """roll ~ 90 plus small shear/keystone (the wild_omnidocbench failure case)."""
    h, w = 959, 1600
    H = build_page_homography((h, w), (-89.36, 1.0, -0.31, -0.111, 0.009))
    s = np.linalg.svd(_linear_part(H, w, h), compute_uv=False)
    assert s[0] / s[1] < 1.1


def test_wrong_arity_rejected():
    with pytest.raises(ValueError):
        build_page_homography((800, 600), (5.0,))
    with pytest.raises(ValueError):
        plan_page_correction((800, 600), (5.0,))


def test_two_value_warp_params_is_pure_rotation():
    """Older (roll, conf) models: missing terms are 0, as in the reference script."""
    two = build_page_homography((800, 600), (5.0, 1.0))
    five = build_page_homography((800, 600), (5.0, 1.0, 0.0, 0.0, 0.0))
    np.testing.assert_allclose(two, five)
    assert plan_page_correction((800, 600), (90.4, 1.0)).rot90_k == 3  # bare turn
    assert plan_page_correction((800, 600), (0.8, 1.0)) is None  # below gate
    assert plan_page_correction((800, 600), (3.0, 1.0)).homography is not None
    assert plan_page_correction((800, 600), (90.0, 0.5)) is None  # low conf


def test_non_finite_warp_params_keep_plain_crop():
    nan, inf = float("nan"), float("inf")
    for wp in [(nan, 1.0, 0, 0, 0), (90.0, 1.0, nan, 0, 0), (3.0, 1.0, 0, inf, 0)]:
        assert plan_page_correction((800, 600), wp) is None


def _block_scale(quad, H):
    """The equal-area factor rectify_shape_crop applies to this block."""
    q = cv2.perspectiveTransform(quad.reshape(-1, 1, 2), H).reshape(-1, 2)
    return np.sqrt(
        cv2.contourArea(quad.astype(np.float32)) / cv2.contourArea(q.astype(np.float32))
    )


def test_block_crop_equals_whole_page_crop(ref):
    """A block rectified on its own == the same block cut from rectify(page),
    up to the block's uniform equal-area scale."""
    rng = np.random.default_rng(1)
    h, w = 1500, 1100
    img = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
    img = cv2.GaussianBlur(img, (0, 0), 3)  # smooth, so interpolation error is tiny
    wp = (12.0, 1.0, -8.0, 0.35, -0.25)
    quad = np.array([[420, 510], [760, 540], [740, 830], [400, 800]], np.float64)
    box = {
        "cls_id": 0,
        "label": "text",
        "coordinate": [400, 510, 760, 830],
        "quad": quad.tolist(),
    }

    H = build_page_homography(img.shape, wp)
    crop = CropByBoxes()(img, [box], "auto", page_homography=H)[0]["img"]

    page, h_page = ref.rectify(img, wp, max_scale=1e9)
    # Same geometry as the reference; only the block's scale differs.  Bring the
    # whole-page result to the block's scale and sub-pixel offset, then compare.
    s = _block_scale(quad, h_page)
    q = cv2.perspectiveTransform(quad.reshape(-1, 1, 2), h_page).reshape(-1, 2) * s
    x0, y0 = q.min(0)
    ch, cw = crop.shape[:2]
    shift = np.array([[s, 0, -x0], [0, s, -y0]], np.float64)
    ref_crop = cv2.warpAffine(page, shift, (cw, ch), flags=cv2.INTER_LINEAR)
    poly = np.round(q - [x0, y0]).astype(np.int32)
    inside = np.zeros((ch, cw), np.uint8)
    cv2.fillPoly(inside, [poly], 1)
    inside = cv2.erode(inside, np.ones((7, 7), np.uint8)).astype(bool)
    diff = np.abs(crop.astype(int) - ref_crop.astype(int))[inside]
    assert inside.sum() > 0.6 * ch * cw * 0.5
    assert np.median(diff) <= 2 and np.percentile(diff, 99) <= 10


def test_equal_area():
    """Rectified block covers the same pixel area as the original quad."""
    wp = (7.8, 1.0, -16.5, -0.23, 0.78)  # strong keystone (a real 91B7 prediction)
    img = np.full((2048, 1536, 3), 128, np.uint8)
    H = build_page_homography(img.shape, wp)
    for quad in (
        [[100, 100], [700, 120], [690, 300], [110, 280]],  # far side
        [[300, 1700], [1300, 1720], [1290, 1950], [310, 1930]],
    ):  # near side
        quad = np.asarray(quad, np.float64)
        box = {
            "cls_id": 0,
            "label": "text",
            "coordinate": [0, 0, 1, 1],
            "quad": quad.tolist(),
        }
        crop = CropByBoxes()(img, [box], "auto", page_homography=H)[0]["img"]
        covered = int((crop != 255).any(-1).sum())
        assert abs(covered / cv2.contourArea(quad.astype(np.float32)) - 1) < 0.03


def test_off_switch_is_plain_crop():
    rng = np.random.default_rng(2)
    img = rng.integers(0, 255, size=(400, 300, 3), dtype=np.uint8)
    box = {
        "cls_id": 0,
        "label": "text",
        "coordinate": [20, 30, 200, 150],
        "quad": [[20, 30], [200, 35], [198, 150], [22, 148]],
    }
    a = CropByBoxes()(img, [box], "auto")[0]["img"]
    b = CropByBoxes()(img, [box], "auto", page_homography=None)[0]["img"]
    assert np.array_equal(a, b)


# ---------------------------------------------------------------- gating ----


def test_split_roll():
    assert split_roll(0.4) == (0, 0.4)
    q, r = split_roll(89.6)
    assert q == 1 and abs(r + 0.4) < 1e-9
    q, r = split_roll(-90.7)
    assert q == 3 and abs(r + 0.7) < 1e-9
    q, r = split_roll(179.5)
    assert q == 2 and abs(r + 0.5) < 1e-9
    q, r = split_roll(-178.0)
    assert q == 2 and abs(r - 2.0) < 1e-9


def test_keystone_deg_matches_border_tilt():
    """The keystone angle is the tilt the correction removes from the page border."""
    for h, w in [(1600, 1200), (959, 1600)]:
        for kx, ky in [(0.05, 0.0), (0.0, 0.05), (0.15, 0.0), (0.0, -0.2)]:
            H = build_page_homography(
                (h, w), (0, 1, 0, kx, ky), min_roll=99, min_shear=99, min_tilt=0
            )
            c = cv2.perspectiveTransform(
                np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64).reshape(
                    -1, 1, 2
                ),
                H,
            ).reshape(-1, 2)
            top = np.degrees(np.arctan2(*(c[1] - c[0])[::-1]))
            left = np.degrees(np.arctan2(*(c[3] - c[0])))
            measured = max(abs(top) % 180, abs(left) % 180)
            measured = min(measured, 180 - measured)
            assert abs(measured - keystone_deg((h, w), kx, ky)) < 0.05


def test_plan_low_confidence_is_none():
    assert plan_page_correction((800, 600), (90.0, 0.5, 5.0, 0.2, 0.2)) is None


def test_plan_small_upright_is_none():
    assert plan_page_correction((800, 600), (1.5, 1.0, -1.2, 0.01, 0.02)) is None


def test_plan_small_quarter_turn_is_bare_rot90():
    plan = plan_page_correction((800, 600), (90.6, 1.0, 0.8, 0.005, -0.01))
    assert plan.homography is None and plan.rot90_k == 3
    plan = plan_page_correction((800, 600), (-179.4, 1.0, 0.1, 0.0, 0.0))
    assert plan.homography is None and plan.rot90_k == 2


@pytest.mark.parametrize(
    "wp",
    [
        (2.5, 1.0, 0.0, 0.0, 0.0),  # residual roll
        (0.1, 1.0, -2.5, 0.0, 0.0),  # shear
        (0.1, 1.0, 0.0, 0.0, 0.12),  # keystone (~2.6 deg on 800x600)
        (91.0, 1.0, 3.0, 0.0, 0.0),  # quarter turn + significant fine part
    ],
)
def test_plan_significant_is_full_homography_with_all_four(wp):
    """Above threshold, all four values are applied -- not only the one that tripped it."""
    plan = plan_page_correction((800, 600), wp)
    assert plan.rot90_k == 0 and plan.homography is not None
    full = build_page_homography((800, 600), wp, min_roll=0, min_shear=0, min_tilt=0)
    np.testing.assert_allclose(plan.homography, full)


def test_plan_inf_thresholds_leave_quarter_turns_only():
    inf = float("inf")
    kw = dict(min_roll=inf, min_shear=inf, min_keystone=inf)
    assert plan_page_correction((800, 600), (8.0, 1.0, 9.0, 0.4, 0.4), **kw) is None
    assert (
        plan_page_correction((800, 600), (-84.0, 1.0, 9.0, 0.4, 0.4), **kw).rot90_k == 1
    )


@pytest.mark.parametrize("roll", [90.0, 180.0, 270.0, -90.0])
def test_bare_rot90_equals_homography_quarter_turn(roll):
    """The interpolation-free turn is the same picture as the homography's pure turn."""
    rng = np.random.default_rng(3)
    h, w = 300, 500
    img = cv2.GaussianBlur(
        rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8), (0, 0), 2
    )
    box = {
        "cls_id": 0,
        "label": "text",
        "coordinate": [0, 0, w, h],
        "quad": [[0, 0], [w, 0], [w, h], [0, h]],
    }
    H = build_page_homography(
        (h, w), (roll, 1.0, 0, 0, 0), min_roll=0, min_shear=99, min_tilt=99
    )
    via_h = CropByBoxes()(img, [box], "auto", page_homography=H)[0]["img"]
    plan = plan_page_correction((h, w), (roll, 1.0, 0, 0, 0))
    via_rot = CropByBoxes()(img, [box], "auto", rot90_k=plan.rot90_k)[0]["img"]
    assert via_rot.shape == via_h.shape
    # Same orientation; the homography crop sits up to one pixel off because it
    # frames the quad's pixel-extent corners (w, h) rather than pixel centres.
    best = min(
        np.abs(
            np.roll(via_rot, (dy, dx), (0, 1))[3:-3, 3:-3].astype(int)
            - via_h[3:-3, 3:-3].astype(int)
        ).mean()
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
    )
    assert best < 0.5
    # ...and every other orientation is far off, so the direction is pinned.
    for other in {0, 1, 2, 3} - {plan.rot90_k}:
        alt = np.rot90(CropByBoxes()(img, [box], "auto")[0]["img"], other)
        if alt.shape == via_h.shape:
            assert (
                np.abs(
                    alt[3:-3, 3:-3].astype(int) - via_h[3:-3, 3:-3].astype(int)
                ).mean()
                > 5
            )


def test_bare_rot90_is_lossless():
    rng = np.random.default_rng(4)
    img = rng.integers(0, 255, size=(120, 200, 3), dtype=np.uint8)
    box = {"cls_id": 0, "label": "text", "coordinate": [10, 20, 90, 60]}
    plain = CropByBoxes()(img, [box], "rect")[0]["img"]
    turned = CropByBoxes()(img, [box], "rect", rot90_k=1)[0]["img"]
    assert np.array_equal(np.rot90(turned, -1), plain)


# ------------------------------------------------- figures inside a table ----


def _figure_page(h=600, w=800):
    """White page, a table box, and one dark figure square inside it."""
    img = np.full((h, w, 3), 255, np.uint8)
    table = [100, 120, 700, 520]
    fig = [380, 260, 480, 360]
    img[fig[1] : fig[3], fig[0] : fig[2]] = 30
    return img, table, fig


def _token_where_figure_was(crop_before, crop_after):
    """The token is a white box with dark text painted over the figure, so where the
    (dark) figure was, the crop turns mostly white: mean brightness there, 0-255."""
    dark_before = crop_before.min(axis=2) < 80
    return (
        float(crop_after.min(axis=2)[dark_before].mean()) if dark_before.any() else 0.0
    )


@pytest.mark.parametrize(
    "wp",
    [
        (90.3, 1.0, 0.2, 0.0, 0.0),  # bare quarter turn
        (-89.0, 1.0, 3.0, 0.05, 0.0),  # full, with a quarter turn
        (4.0, 1.0, -3.0, 0.1, 0.05),
    ],
)  # full, upright page
def test_table_figure_token_lands_on_figure_after_correction(wp):
    img, table, fig = _figure_page()
    box = {
        "cls_id": 0,
        "label": "table",
        "coordinate": table,
        "quad": [
            [table[0], table[1]],
            [table[2], table[1]],
            [table[2], table[3]],
            [table[0], table[3]],
        ],
    }
    plan = plan_page_correction(img.shape, wp)
    blk = CropByBoxes()(
        img, [box], "auto", page_homography=plan.homography, rot90_k=plan.rot90_k
    )[0]
    assert "crop_matrix" in blk
    figures = [{"coordinate": fig, "path": "imgs/f.jpg"}]
    out, token_map, dropped = tokenize_figure_of_table(
        blk["img"], blk["box"], figures, crop_matrix=blk["crop_matrix"]
    )
    assert token_map and dropped == ["imgs/f.jpg"]
    assert _token_where_figure_was(blk["img"], out) > 150
    # Without the matrix the token is painted at the pre-correction offset and the
    # figure stays (mostly) visible.
    wrong, _, _ = tokenize_figure_of_table(blk["img"], blk["box"], figures)
    assert (
        _token_where_figure_was(blk["img"], wrong)
        < _token_where_figure_was(blk["img"], out) - 50
    )


def test_table_figure_token_plain_crop_unchanged():
    """No correction: no crop_matrix key, and tokenisation is byte-identical to before."""
    img, table, fig = _figure_page()
    box = {"cls_id": 0, "label": "table", "coordinate": table}
    blk = CropByBoxes()(img, [box], "rect")[0]
    assert "crop_matrix" not in blk
    figures = [{"coordinate": fig, "path": "imgs/f.jpg"}]
    a = tokenize_figure_of_table(blk["img"], blk["box"], figures)
    b = tokenize_figure_of_table(blk["img"], blk["box"], figures, crop_matrix=None)
    assert np.array_equal(a[0], b[0]) and a[1] == b[1] and a[2] == b[2]
    assert _token_where_figure_was(blk["img"], a[0]) > 150
