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

"""Conservative, image-only splitting of displayed formula regions.

This finds separated ink rows, not semantic equation boundaries. Shared tall
delimiters block cuts; ambiguous/dense regions are deliberately left intact.
Residual row tilt and mild book-page curvature inside a crop are compensated
before horizontal gap detection when doing so genuinely opens inter-row gaps.
"""

import numpy as np

from ....utils.deps import function_requires_deps, is_dep_available

if is_dep_available("opencv-contrib-python"):
    import cv2


def _formula_ink(gray):
    # Estimate the paper illumination before Otsu: a global threshold alone
    # mistakes the darker side of a shaded scan for foreground. Closing fills
    # thin dark strokes without averaging them into the local background.
    k = max(15, (min(gray.shape) // 8) | 1)
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    norm = cv2.divide(gray, np.maximum(background, 1), scale=255)
    _, ink = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return ink


def _estimate_char_height(stats):
    widths, heights, areas = stats[:, 2], stats[:, 3], stats[:, 4]
    chars = (heights >= 3) & (widths <= 3 * heights) & (areas >= 4)
    if np.count_nonzero(chars) < 4:
        return None
    ch_h = heights[chars]
    ch_w = widths[chars]
    # Exclude thin horizontal strokes (e.g. '=' bars, '-') and tiny punctuation
    # dots when enough body glyphs exist so they do not pull char_height down.
    main = (ch_h >= 5) & ~((ch_w >= 2 * ch_h) & (ch_h <= 6))
    if np.count_nonzero(main) >= 4:
        main_h = ch_h[main]
        p75 = float(np.percentile(main_h, 75))
        body = main_h[main_h >= 0.4 * p75]
        if len(body) >= 4:
            return float(np.median(body))
        return float(np.median(main_h))
    return float(np.median(ch_h))


def _analyze_ink(gray):
    """Compute binarized ink, connected components, and character height once."""
    if min(gray.shape) < 8 or np.ptp(gray) < 20:
        return None
    ink = _formula_ink(gray)
    # Dark backgrounds and densely filled regions are outside this heuristic.
    if np.count_nonzero(ink) > 0.45 * ink.size:
        return None
    _, components, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    stats = stats[1:]
    if len(stats) < 4:
        return None
    char_height = _estimate_char_height(stats)
    if char_height is None:
        return None
    return ink, components, stats, char_height


def _precompute_row_stats(ink):
    height, width = ink.shape
    ink_bool = ink > 0
    row_counts = np.count_nonzero(ink_bool, axis=1)
    has_row = row_counts > 0
    row_xmin = np.full(height, width, dtype=np.int32)
    row_xmax = np.full(height, -1, dtype=np.int32)
    if np.any(has_row):
        sub = ink_bool[has_row]
        row_xmin[has_row] = np.argmax(sub, axis=1)
        row_xmax[has_row] = width - 1 - np.argmax(sub[:, ::-1], axis=1)
    return row_counts, row_xmin, row_xmax


def _slice_ink_summary(row_counts, row_xmin, row_xmax, a, b):
    sub_counts = row_counts[a:b]
    nz = np.flatnonzero(sub_counts)
    if not len(nz):
        return None
    ymin = a + int(nz[0])
    ymax = a + int(nz[-1]) + 1
    xmin = int(row_xmin[a:b][nz].min())
    xmax = int(row_xmax[a:b][nz].max()) + 1
    count = int(sub_counts[nz].sum())
    return ymin, ymax, xmin, xmax, count


def _is_satellite(ymin, ymax, xmin, xmax, count, char_height):
    row_h = ymax - ymin
    row_w = xmax - xmin
    if row_h < 0.72 * char_height or count < 0.85 * char_height**2:
        return True
    # Detached limits, accents, or isolated equation numbers are both short and
    # narrow; full single-height rows like "A + B = C + D" have wider span.
    if row_h < 1.15 * char_height and row_w < 1.8 * char_height:
        return True
    return False


def _protected_fraction_intervals(ink, components, stats, char_height):
    widths, heights = stats[:, 2], stats[:, 3]
    max_bar_h = max(3, 0.35 * char_height)
    min_bar_w = max(4, int(round(0.8 * char_height)))
    bars = stats[
        (widths >= 0.8 * char_height) & (heights <= max_bar_h) & (widths >= 2 * heights)
    ]
    # Horizontal morphological opening also recovers fraction bars that touch a
    # numerator descender or denominator ascender after scanning/resampling.
    opened = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, min_bar_w), np.uint8))
    _, _, opened_stats, _ = cv2.connectedComponentsWithStats(opened, 8)
    opened_stats = opened_stats[1:]
    if len(opened_stats):
        ow, oh = opened_stats[:, 2], opened_stats[:, 3]
        opened_bars = opened_stats[
            (ow >= 0.8 * char_height) & (oh <= max_bar_h) & (ow >= 2 * oh)
        ]
        if len(opened_bars):
            bars = np.vstack([bars, opened_bars]) if len(bars) else opened_bars
    if not len(bars):
        return []
    # Numerators/denominators need character-height components. Other short
    # horizontal strokes (e.g. equals signs on adjacent dense rows) must not
    # make an equals sign look like a fraction bar. Exclude the opened bar
    # pixels themselves so a connected bar cannot validate itself.
    fraction_ink = np.r_[False, heights >= max(4, 0.5 * char_height)][components]
    if np.any(opened):
        fraction_ink = fraction_ink & (opened == 0)
    protected = []
    reach = max(2, int(round(1.5 * char_height)))
    for x, y, w, h, _ in bars:
        above_start = max(0, int(y) - reach)
        above = fraction_ink[above_start : int(y), int(x) : int(x + w)]
        below = fraction_ink[int(y + h) : int(y + h) + reach, int(x) : int(x + w)]
        # Wide fraction bars (>= 2 * char_height) cannot be '-' or '=' signs,
        # so allow thin single-character numerators/denominators like "1".
        min_ink = (0.07 if w >= 2.0 * char_height else 0.15) * char_height**2
        if min(np.count_nonzero(above), np.count_nonzero(below)) >= min_ink:
            ay = np.flatnonzero(above.any(axis=1))
            by = np.flatnonzero(below.any(axis=1))
            if not len(ay) or not len(by):
                continue
            dist_above = float(y - (above_start + int(ay[-1]) + 1))
            dist_below = float(int(by[0]))
            # A bar touching ink on both top and bottom (e.g. the horizontal
            # crossbar of a '+' sign) has no internal gap to protect and must
            # not reach outward to block an adjacent inter-row cut.
            if dist_above == 0 and dist_below == 0:
                continue
            # A real fraction bar sits close to at least one of its numerator or
            # denominator; an operator ('-' or '=') in a dense multi-row formula
            # can otherwise reach across inter-row gaps into neighboring rows.
            if min(dist_above, dist_below) > 0.75 * char_height:
                continue
            # Protect only the side(s) that have a gap between the bar and its
            # numerator/denominator; avoid reaching outward into neighboring rows.
            top_p = (
                min(y - 0.75 * char_height, above_start + float(ay[-1]) - 1.0)
                if dist_above > 0
                else float(y)
            )
            bot_p = (
                max(y + h + 0.75 * char_height, y + h + float(by[0]) + 1.0)
                if dist_below > 0
                else float(y + h)
            )
            protected.append((top_p, bot_p))
    return protected


def _find_row_boundaries(ink, components, stats, char_height, min_gap_ratio=0.6):
    """Find validated row boundaries and rectangles in a single-orientation ink map."""
    height, width = ink.shape
    areas = stats[:, 4]
    # Only discard isolated speckles when finding gaps. All binarized ink is
    # used below to bound the children, preserving punctuation and small scripts.
    noise_area = max(1, int(char_height * char_height * 0.04))
    occupied = np.zeros(height, dtype=bool)
    for _, y, _, h, _ in stats[areas > noise_area]:
        occupied[int(y) : int(y + h)] = True
    transitions = np.diff(np.r_[False, ~occupied, False].astype(np.int8))
    starts, ends = np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)
    min_gap = max(3.0, min_gap_ratio * char_height)
    raw_cuts = [
        ((int(a) + int(b)) // 2, int(b - a))
        for a, b in zip(starts, ends)
        if a > 0 and b < height and (b - a) >= min_gap
    ]
    if not raw_cuts:
        return None
    protected = _protected_fraction_intervals(ink, components, stats, char_height)
    kept_cuts = [
        (c, gap) for c, gap in raw_cuts if not any(a < c < b for a, b in protected)
    ]
    if not kept_cuts:
        return None
    boundaries = [0, *[c for c, _ in kept_cuts], height]
    cut_gaps = [gap for _, gap in kept_cuts]
    row_counts, row_xmin, row_xmax = _precompute_row_stats(ink)
    # Attach small detached limits/numbers to their nearest body. A satellite
    # should not veto valid splits elsewhere in a long derivation.
    while len(boundaries) > 2:
        regions = []
        for a, b in zip(boundaries, boundaries[1:]):
            summary = _slice_ink_summary(row_counts, row_xmin, row_xmax, a, b)
            if summary is None:
                return None
            regions.append(summary)
        invalid = [
            i
            for i, (ymin, ymax, xmin, xmax, count) in enumerate(regions)
            if _is_satellite(ymin, ymax, xmin, xmax, count, char_height)
        ]
        if not invalid:
            break
        i = invalid[0]
        if i == 0:
            remove = 1
        elif i == len(regions) - 1:
            remove = i
        else:
            ymin_i, ymax_i, xmin_i, xmax_i, _ = regions[i]
            gap_up = ymin_i - regions[i - 1][1]
            gap_down = regions[i + 1][0] - ymax_i
            ov_up = max(
                0, min(xmax_i, regions[i - 1][3]) - max(xmin_i, regions[i - 1][2])
            )
            ov_down = max(
                0, min(xmax_i, regions[i + 1][3]) - max(xmin_i, regions[i + 1][2])
            )
            if (
                ov_up > 0
                and ov_down == 0
                and gap_up <= 2.5 * max(gap_down, 1) + char_height
            ):
                remove = i
            elif (
                ov_down > 0
                and ov_up == 0
                and gap_down <= 2.5 * max(gap_up, 1) + char_height
            ):
                remove = i + 1
            else:
                remove = i if gap_up <= gap_down else i + 1
        boundaries.pop(remove)
        cut_gaps.pop(remove - 1)
    if len(boundaries) < 3:
        return None
    rects = []
    pad = max(1, int(round(char_height * 0.12)))
    for a, b in zip(boundaries, boundaries[1:]):
        summary = _slice_ink_summary(row_counts, row_xmin, row_xmax, a, b)
        if summary is None:
            return None
        ymin, ymax, xmin, xmax, count = summary
        if _is_satellite(ymin, ymax, xmin, xmax, count, char_height):
            return None
        rects.append(
            [
                max(0, xmin - pad),
                max(a, ymin - pad),
                min(width, xmax + pad),
                min(b, ymax + pad),
            ]
        )
    return boundaries, rects, min(cut_gaps)


def _estimate_baseline_delta(ink, char_height):
    """Estimate smooth column-wise vertical baseline drift delta(x) in pixels.

    Tracks horizontally closed single-row ribbons across vertical column slices
    and fits a regularized cubic polynomial over the ink horizontal span.
    Returns ``None`` when the region is too small, lacks reliable ribbons, the
    fitted slope exceeds ~37 degrees, or the drift is too small to need
    unwarping (peak-to-peak below ``max(4, 0.35 * char_height)``).
    """
    height, width = ink.shape
    if width < max(24, int(3 * char_height)) or height < max(
        16, int(1.5 * char_height)
    ):
        return None
    col_any = np.any(ink > 0, axis=0)
    nz_cols = np.flatnonzero(col_any)
    if len(nz_cols) < max(20, int(2.5 * char_height)):
        return None
    ink_x0, ink_x1 = int(nz_cols[0]), int(nz_cols[-1])
    ink_span = float(max(ink_x1 - ink_x0, 1))
    kw = max(5, int(round(1.6 * char_height)))
    kh = max(3, int(round(0.28 * char_height)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kw, kh))
    ribbon = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, kernel) > 0
    step = max(1, int(ink_span) // 64)
    xs_sample = np.arange(ink_x0 + step // 2, ink_x1 + 1, step)
    if len(xs_sample) < 6:
        return None
    ux_sample = (2.0 * (xs_sample - ink_x0) - ink_span) / ink_span
    max_gap_bins = max(2, int(round(2.2 * char_height / step)))
    min_run = max(3.0, 0.35 * char_height)
    max_run = 1.85 * char_height
    chains = []
    active = []
    for b_idx, x in enumerate(xs_sample):
        col = ribbon[:, int(x)]
        diffs = np.diff(np.r_[False, col, False].astype(np.int8))
        starts, ends = np.flatnonzero(diffs == 1), np.flatnonzero(diffs == -1)
        lengths = ends - starts
        keep = (lengths >= min_run) & (lengths <= max_run)
        ycs = 0.5 * (starts[keep] + ends[keep] - 1.0)
        new_active = []
        used_chains = set()
        for yc in ycs:
            best_cid = None
            best_score = float("inf")
            for cid, last_b, last_yc in active:
                db = b_idx - last_b
                if cid in used_chains or db > max_gap_bins:
                    continue
                dx = db * step
                dy = abs(float(yc) - last_yc)
                if dy <= 0.35 * char_height + 0.45 * dx:
                    score = dy + 0.2 * dx
                    if score < best_score:
                        best_score = score
                        best_cid = cid
            if best_cid is not None:
                chains[best_cid].append((b_idx, float(yc)))
                used_chains.add(best_cid)
                new_active.append((best_cid, b_idx, float(yc)))
            else:
                cid = len(chains)
                chains.append([(b_idx, float(yc))])
                new_active.append((cid, b_idx, float(yc)))
        for cid, last_b, last_yc in active:
            if cid not in used_chains and b_idx - last_b < max_gap_bins:
                new_active.append((cid, last_b, last_yc))
        active = new_active

    X_rows, y_rows = [], []
    for ch_pts in chains:
        if len(ch_pts) < 5:
            continue
        b_inds = np.array([p[0] for p in ch_pts], dtype=np.int32)
        if xs_sample[b_inds[-1]] - xs_sample[b_inds[0]] < 2.0 * char_height:
            continue
        yc = np.array([p[1] for p in ch_pts], dtype=np.float64)
        u_c = ux_sample[b_inds]
        basis = np.column_stack([u_c, u_c**2, u_c**3])
        basis -= basis.mean(axis=0, keepdims=True)
        X_rows.append(basis)
        y_rows.append(yc - yc.mean())
    if not X_rows:
        return None
    X = np.vstack(X_rows)
    y = np.concatenate(y_rows)
    reg = np.diag([0.2, 2.0, 6.0])
    try:
        coef = np.linalg.solve(X.T @ X + reg, X.T @ y)
        resid = np.abs(y - X @ coef)
        scale = max(1.5, 1.4826 * float(np.median(resid)))
        w = np.where(
            resid <= 2.0 * scale,
            1.0,
            (2.0 * scale / np.maximum(resid, 1e-6)) ** 2,
        )
        xw = X * w[:, None]
        coef = np.linalg.solve(xw.T @ X + reg, xw.T @ y)
    except np.linalg.LinAlgError:
        return None

    # Clamp normalized x to [-1, 1] outside the ink span to prevent cubic
    # extrapolation into empty margins.
    ux_all = np.clip(
        (2.0 * (np.arange(width, dtype=np.float64) - ink_x0) - ink_span) / ink_span,
        -1.0,
        1.0,
    )
    delta = (coef[0] * ux_all + coef[1] * ux_all**2 + coef[2] * ux_all**3).astype(
        np.float32
    )
    delta -= float(np.mean(delta[ink_x0 : ink_x1 + 1]))
    # Reject implausibly steep slopes (> ~37 deg) that could shear single lines.
    if width > 1 and float(np.max(np.abs(np.diff(delta[ink_x0 : ink_x1 + 1])))) > 0.75:
        return None
    if float(np.ptp(delta[ink_x0 : ink_x1 + 1])) < max(4.0, 0.35 * char_height):
        return None
    return delta


def _unwarp_ink_by_delta(ink, delta):
    """Shift each column x vertically by -round(delta[x]) into a padded binary canvas."""
    height, width = ink.shape
    shifts = np.rint(delta).astype(np.int32)
    pad_top = max(0, int(shifts.max()))
    pad_bot = max(0, int(-shifts.min()))
    new_h = height + pad_top + pad_bot
    ys, xs = np.nonzero(ink)
    new_ys = ys + pad_top - shifts[xs]
    valid = (new_ys >= 0) & (new_ys < new_h)
    unwarped = np.zeros((new_h, width), dtype=np.uint8)
    unwarped[new_ys[valid], xs[valid]] = 255
    return unwarped, shifts, pad_top


def _rows_have_horizontal_overlap(rects):
    for a, b in zip(rects, rects[1:]):
        if min(a[2], b[2]) <= max(a[0], b[0]):
            return False
    return True


def _build_unwarped_row_geometries(
    ink, char_height, boundaries, delta, shifts, pad_top
):
    """Map unwarped row boundaries back to oriented quads and curved polygons."""
    height, width = ink.shape
    pad = max(1, int(round(char_height * 0.12)))
    ys_all, xs_all = np.nonzero(ink)
    if not len(xs_all):
        return None
    y_flat_all = ys_all + pad_top - shifts[xs_all]
    geometries = []
    for a, b in zip(boundaries, boundaries[1:]):
        in_row = (y_flat_all >= a) & (y_flat_all < b)
        if not np.any(in_row):
            return None
        ys = ys_all[in_row]
        xs = xs_all[in_row]
        left = max(0, int(xs.min()) - pad)
        right = min(width, int(xs.max()) + 1 + pad)
        if right <= left:
            return None
        # Fit the linear baseline trend along this row so the 4-point quad
        # tilts with the row while still containing 100% of its curved ink.
        span_x = np.arange(left, right, dtype=np.float64)
        span_d = delta[left:right].astype(np.float64)
        if len(span_x) >= 2:
            slope, intercept = np.polyfit(span_x, span_d, 1)
        else:
            slope, intercept = 0.0, float(span_d[0])
        resid_lin = ys.astype(np.float64) - (slope * xs.astype(np.float64) + intercept)
        r_min = float(resid_lin.min()) - pad
        r_max = float(resid_lin.max()) + 1.0 + pad
        x_l = float(left)
        x_r = float(max(left, right - 1))
        local_quad = np.float32(
            [
                [x_l, slope * x_l + intercept + r_min],
                [x_r, slope * x_r + intercept + r_min],
                [x_r, slope * x_r + intercept + r_max],
                [x_l, slope * x_l + intercept + r_max],
            ]
        )
        # Also compute a curved polygon following delta(x) when curvature
        # deviates noticeably from the linear trend.
        sagitta = float(np.max(np.abs(span_d - (slope * span_x + intercept))))
        local_poly = None
        if sagitta >= 2.5 and right - left >= 12:
            resid_curve = ys.astype(np.float64) - delta[xs].astype(np.float64)
            c_min = float(resid_curve.min()) - pad
            c_max = float(resid_curve.max()) + 1.0 + pad
            num_pts = min(8, max(4, (right - left) // max(12, int(char_height)) + 1))
            sample_x = np.linspace(x_l, x_r, num_pts)
            sample_d = np.interp(sample_x, np.arange(width, dtype=np.float64), delta)
            top_pts = np.column_stack([sample_x, sample_d + c_min])
            bot_pts = np.column_stack([sample_x[::-1], sample_d[::-1] + c_max])
            local_poly = np.vstack([top_pts, bot_pts]).astype(np.float32)
        geometries.append(
            (local_quad, local_poly, float(ys.min()), float(ys.max()) + 1.0)
        )
    return geometries


def _line_rectangles(
    gray, min_gap_ratio=0.6, analysis=None, allow_unwarp=False, unwarp_cache=None
):
    """Return crop-local line rectangles, or None when no safe split is found.

    With ``allow_unwarp=False`` the result is a list of ``(left, top, right,
    bottom)`` rectangles. With ``allow_unwarp=True`` it is a 2-tuple: either
    ``(rects, None)`` for a flat split, or ``(None, geoms)`` for a curved split
    where each geom is ``(local_quad, local_poly, top, bottom)`` as built by
    ``_build_unwarped_row_geometries`` (``local_poly`` may be None). Passing
    the same ``unwarp_cache`` dict for one ``analysis`` reuses the baseline fit
    and unwarped ink across calls.
    """
    if analysis is None:
        analysis = _analyze_ink(gray)
    if analysis is None:
        return None
    ink, components, stats, char_height = analysis
    flat = _find_row_boundaries(ink, components, stats, char_height, min_gap_ratio)
    if not allow_unwarp:
        return flat[1] if flat is not None else None
    if unwarp_cache is not None and "delta" in unwarp_cache:
        delta, unwarped_parts = unwarp_cache["delta"], unwarp_cache["unwarped"]
    else:
        delta = _estimate_baseline_delta(ink, char_height)
        unwarped_parts = None
        if delta is not None:
            unwarped, shifts, pad_top = _unwarp_ink_by_delta(ink, delta)
            _, u_comp, u_stats, _ = cv2.connectedComponentsWithStats(unwarped, 8)
            unwarped_parts = (unwarped, shifts, pad_top, u_comp, u_stats[1:])
        if unwarp_cache is not None:
            unwarp_cache["delta"] = delta
            unwarp_cache["unwarped"] = unwarped_parts
    if delta is None:
        return (flat[1], None) if flat is not None else None
    unwarped, shifts, pad_top, u_comp, u_stats = unwarped_parts
    curved = (
        _find_row_boundaries(unwarped, u_comp, u_stats, char_height, min_gap_ratio)
        if len(u_stats) >= 4
        else None
    )
    use_curved = False
    if curved is not None and _rows_have_horizontal_overlap(curved[1]):
        if flat is None or len(curved[0]) > len(flat[0]):
            use_curved = True
        elif len(curved[0]) == len(flat[0]) and curved[2] >= flat[2] + max(
            3.0, 0.25 * char_height
        ):
            use_curved = True
    if use_curved:
        geoms = _build_unwarped_row_geometries(
            ink, char_height, curved[0], delta, shifts, pad_top
        )
        if geoms is not None:
            return None, geoms
    return (flat[1], None) if flat is not None else None


def _candidate_lines(
    gray, transform, parent, candidates, analysis=None, unwarp_cache=None
):
    """Use a complete, stronger detector row set, with image validation."""
    if not candidates:
        return None
    height, width = gray.shape
    selected = []
    for candidate in candidates:
        if candidate["label"] != "display_formula":
            continue
        if candidate["score"] < max(0.5, parent["score"] + 0.05):
            continue
        x1, y1, x2, y2 = candidate["coordinate"]
        shape = candidate.get("quad", candidate.get("polygon_points"))
        quad = np.asarray(
            shape if shape is not None else [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            dtype=np.float32,
        )
        if quad.shape != (4, 2) or not np.isfinite(quad).all():
            continue
        if not cv2.isContourConvex(quad):
            continue
        local = cv2.perspectiveTransform(quad[None], transform)[0]
        lo, hi = local.min(axis=0), local.max(axis=0)
        # Allow rounding at the parent edges, but no neighboring-column boxes,
        # same-size duplicates, or nested symbols mistaken for entire rows.
        if np.any(lo < -3) or np.any(hi > [width + 3, height + 3]):
            continue
        if hi[1] - lo[1] > 0.65 * height or hi[0] - lo[0] < 0.25 * width:
            continue
        selected.append((float(local[:, 1].mean()), candidate, local))
    if not 2 <= len(selected) <= 16:
        return None
    selected.sort(key=lambda item: item[0])
    if analysis is None:
        analysis = _analyze_ink(gray)
    if analysis is None:
        return None
    ink = analysis[0] > 0
    # Detector agreement permits narrower gaps, while the same fraction-bar,
    # shared-delimiter and satellite checks still protect mathematical structure.
    rows = _line_rectangles(gray, min_gap_ratio=0.25, analysis=analysis)
    row_masks = None
    if rows is None or len(rows) != len(selected):
        split_out = _line_rectangles(
            gray,
            min_gap_ratio=0.25,
            analysis=analysis,
            allow_unwarp=True,
            unwarp_cache=unwarp_cache,
        )
        if (
            split_out is None
            or split_out[1] is None
            or len(split_out[1]) != len(selected)
        ):
            return None
        curved_geoms = split_out[1]
        rows = []
        row_masks = []
        for local_quad, local_poly, r_top, r_bot in curved_geoms:
            poly = local_poly if local_poly is not None else local_quad
            mask = np.zeros(gray.shape, np.uint8)
            cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 1)
            row_masks.append(ink & (mask > 0))
            rows.append(
                [
                    max(0, int(np.floor(poly[:, 0].min()))),
                    max(0, int(np.floor(r_top))),
                    min(width, int(np.ceil(poly[:, 0].max())) + 1),
                    min(height, int(np.ceil(r_bot))),
                ]
            )
    covered = np.zeros(gray.shape, np.uint8)
    children = []
    row_spans = []
    for idx, ((left, top, right, bottom), (_, candidate, local)) in enumerate(
        zip(rows, selected)
    ):
        lo, hi = local.min(axis=0), local.max(axis=0)
        if not top <= local[:, 1].mean() < bottom:
            return None
        region = np.zeros(gray.shape, np.uint8)
        cv2.fillPoly(region, [np.round(local).astype(np.int32)], 1)
        region_area = int(np.count_nonzero(region))
        # A slight boundary overlap is normal for detected line boxes; a
        # substantial one indicates competing/nested candidates, not separate rows.
        if region_area == 0 or np.count_nonzero(covered & region) > 0.15 * region_area:
            return None
        if row_masks is not None:
            row_ink = row_masks[idx]
            covered_ink = row_ink & (region > 0)
        else:
            row_ink = ink[top:bottom, left:right]
            covered_ink = row_ink & (region[top:bottom, left:right] > 0)
        if np.count_nonzero(covered_ink) < 0.98 * np.count_nonzero(row_ink):
            return None
        covered |= region
        child = parent.copy()
        child.update(candidate)
        child["order"] = parent.get("order")
        if "polygon_points" in parent and "quad" in child:
            child["polygon_points"] = child["quad"]
        children.append(child)
        row_spans.append((float(lo[1]), float(hi[1])))
    if np.count_nonzero(ink & (covered > 0)) < 0.98 * np.count_nonzero(ink):
        return None
    return children, row_spans


def _coord4(box):
    """Return ``box["coordinate"]`` as a 4-tuple, or None if missing or malformed."""
    coord = box.get("coordinate")
    if coord is None or len(coord) != 4:
        return None
    return tuple(coord)


def _local_to_page(local_pts, inverse, page_w, page_h):
    """Map crop-local points onto the page, clipped to its bounds.

    Returns ``(points, coord)``; ``coord`` is the integer AABB, or None when it
    is degenerate.
    """
    points = cv2.perspectiveTransform(np.float32(local_pts)[None], inverse)[0]
    points[:, 0] = np.clip(points[:, 0], 0, page_w)
    points[:, 1] = np.clip(points[:, 1], 0, page_h)
    coord = [
        *np.floor(points.min(axis=0)).astype(int).tolist(),
        *np.ceil(points.max(axis=0)).astype(int).tolist(),
    ]
    if coord[2] <= coord[0] or coord[3] <= coord[1]:
        return points, None
    return points, coord


def _extract_region_quad(box):
    """Resolve the crop quad of a display_formula box, or None to keep it whole.

    Splitting targets PP-DocLayoutV4 quads and axis-aligned boxes. Segmentation
    polygons do not encode text orientation, so any polygon other than the box
    rectangle itself leaves the box intact.
    """
    coord = _coord4(box)
    if coord is None:
        return None
    x1, y1, x2, y2 = coord
    rect = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    if "quad" in box and box["quad"] is not None:
        quad = np.asarray(box["quad"], dtype=np.float32)
        if quad.shape != (4, 2) or not np.isfinite(quad).all():
            return None
        # V4 quads already carry ordered TL, TR, BR, BL vertices. Reordering by
        # min(x+y) can rotate a wide, perspective-skewed formula by 90 degrees
        # and cause cuts between symbols instead of between formula rows.
        if not cv2.isContourConvex(quad) or cv2.contourArea(quad) < 64:
            return None
        return quad
    if not np.isfinite(rect).all() or cv2.contourArea(rect) < 64:
        return None
    shape = box.get("polygon_points")
    if shape is not None:
        poly = np.asarray(shape, dtype=np.float32)
        if poly.shape != (4, 2) or not np.isfinite(poly).all():
            return None
        distances = np.linalg.norm(poly[:, None] - rect[None], axis=2)
        if not np.all(distances.min(axis=0) <= 2):
            return None
    return rect


def _box_center(box):
    shape = box.get("quad", box.get("polygon_points"))
    if shape is not None:
        pts = np.asarray(shape, dtype=np.float32)
        if pts.ndim == 2 and pts.shape[1] == 2 and len(pts) >= 3:
            return pts.mean(axis=0)
    coord = _coord4(box)
    if coord is None:
        return None
    x1, y1, x2, y2 = coord
    return np.float32([(x1 + x2) * 0.5, (y1 + y2) * 0.5])


def _interleave_formula_numbers(entries, split_records, all_boxes):
    """Pair formula_number boxes right after their aligned split formula rows.

    When a multi-line display_formula is split into rows, keeping all
    formula_number boxes grouped after the last row breaks row-by-row reading
    order and Markdown ``show_formula_number`` pairing.
    """
    fn_entries = [
        (orig_idx, box) for orig_idx, box in entries if box["label"] == "formula_number"
    ]
    assignments = {}
    if split_records and fn_entries:
        for fn_idx, fn_box in fn_entries:
            fn_center = _box_center(fn_box)
            fn_coord = _coord4(fn_box)
            if fn_center is None or fn_coord is None:
                continue
            fn_x1, fn_y1, fn_x2, fn_y2 = fn_coord
            best_match = None
            best_key = (float("inf"), float("inf"))
            for parent_idx, rec in split_records.items():
                local_pt = cv2.perspectiveTransform(
                    fn_center.reshape(1, 1, 2), rec["transform"]
                )[0, 0]
                lx, ly = float(local_pt[0]), float(local_pt[1])
                width, height = rec["width"], rec["height"]
                row_spans = rec["row_spans"]
                avg_h = max(1.0, float(np.mean([b - a for a, b in row_spans])))
                # Must fall vertically within the parent formula band.
                if ly < -0.45 * avg_h or ly > height + 0.45 * avg_h:
                    continue
                # Must be horizontally close to this formula (not across a
                # multi-column gutter with another block in between).
                horiz_dist = 0.0 if 0 <= lx <= width else min(abs(lx), abs(lx - width))
                if horiz_dist > max(width * 1.5, 400.0):
                    continue
                # Reject if another non-formula-number block sits strictly
                # between this formula parent and the formula_number on the
                # same band.
                p_coord = _coord4(rec["parent_box"])
                if p_coord is None:
                    continue
                p_x1, p_y1, p_x2, p_y2 = p_coord
                if p_x2 <= fn_x1:
                    gap_x1, gap_x2 = p_x2, fn_x1
                elif fn_x2 <= p_x1:
                    gap_x1, gap_x2 = fn_x2, p_x1
                else:
                    # Overlapping boxes leave no room for a block in between.
                    gap_x1 = gap_x2 = 0
                blocked = False
                if gap_x2 > gap_x1 + 2:
                    for other_idx, other in enumerate(all_boxes):
                        if (
                            other_idx in (parent_idx, fn_idx)
                            or other["label"] == "formula_number"
                        ):
                            continue
                        o_coord = _coord4(other)
                        if o_coord is None:
                            continue
                        ox1, oy1, ox2, oy2 = o_coord
                        if (
                            ox1 < gap_x2
                            and ox2 > gap_x1
                            and ox1 >= min(p_x1, fn_x1)
                            and ox2 <= max(p_x2, fn_x2)
                            and min(oy2, fn_y2) > max(oy1, fn_y1)
                        ):
                            blocked = True
                            break
                if blocked:
                    continue
                # A nearer unsplit display_formula on the same band owns this
                # number (e.g. its own formula in the other column).
                own_gap = max(0, p_x1 - fn_x2, fn_x1 - p_x2)
                rival = False
                for other_idx, other in enumerate(all_boxes):
                    if (
                        other_idx in split_records
                        or other_idx == fn_idx
                        or other["label"] != "display_formula"
                    ):
                        continue
                    o_coord = _coord4(other)
                    if o_coord is None:
                        continue
                    ox1, oy1, ox2, oy2 = o_coord
                    if (
                        min(oy2, fn_y2) > max(oy1, fn_y1)
                        and max(0, ox1 - fn_x2, fn_x1 - ox2) < own_gap
                    ):
                        rival = True
                        break
                if rival:
                    continue
                # Match to the vertically closest split child row.
                best_row = None
                best_row_dy = float("inf")
                for row_i, (r_top, r_bot) in enumerate(row_spans):
                    r_mid = 0.5 * (r_top + r_bot)
                    r_h = max(1.0, r_bot - r_top)
                    if r_top - 0.45 * r_h <= ly <= r_bot + 0.45 * r_h:
                        dy = abs(ly - r_mid)
                        if dy < best_row_dy:
                            best_row_dy = dy
                            best_row = row_i
                # Prefer the horizontally closest formula; on a tie (e.g.
                # stacked formulas sharing a right edge) the vertically closer row.
                if best_row is not None:
                    key = (horiz_dist, best_row_dy / avg_h)
                    if key < best_key:
                        best_key = key
                        best_match = (parent_idx, best_row)
            if best_match is not None:
                assignments[fn_idx] = best_match

    attached = {}
    for fn_idx, fn_box in fn_entries:
        if fn_idx in assignments:
            attached.setdefault(assignments[fn_idx], []).append(fn_box)

    final_boxes = []
    for orig_idx, box in entries:
        if orig_idx in assignments:
            continue
        if orig_idx in split_records:
            rec = split_records[orig_idx]
            for row_i, child in enumerate(rec["children"]):
                final_boxes.append(child)
                if (orig_idx, row_i) in attached:
                    final_boxes.extend(attached[(orig_idx, row_i)])
        else:
            final_boxes.append(box)
    return final_boxes


@function_requires_deps("opencv-contrib-python")
def split_formula_boxes(image, boxes, candidates=None, keep_whole=None):
    """Replace separable display formulas with line boxes in reading order.

    ``image`` is the original uint8 page (BGR or grayscale). The original list
    and dictionaries are not mutated. Call before assigning final order indices,
    normally after overlap/containment filtering; callers that defer filtering
    (e.g. PaddleOCR-VL) go through ``_split_formula_boxes_with_overlap_guard``
    in processors.py. Image-derived child scores retain the parent detection
    scores; they are not confidence estimates for the split.
    Optional ``candidates`` are display-formula detections after threshold/NMS
    but before containment filtering; accepted detector children keep their own
    scores. When a formula is split and the page has ``formula_number`` boxes,
    each number aligned with a split row is moved right after that row. Neither
    input list is mutated.

    Args:
        image (np.ndarray): Original uint8 page, BGR or grayscale.
        boxes (List[dict]): Layout boxes in reading order.
        candidates (Optional[List[dict]]): Display-formula detections suppressed
            by containment filtering, used as row hints. Defaults to None.
        keep_whole (Optional[Set[int]]): Indices into ``boxes`` of display
            formulas to leave intact. Defaults to None.

    Returns:
        List[dict]: Boxes in reading order, with each separable display_formula
            replaced by its row boxes.
    """
    page_h, page_w = image.shape[:2]
    if candidates:
        # Restore only suppressed detections. A caller can disable containment
        # or overlap filtering, in which case row boxes may already be present.
        def geometry_key(box):
            shape = box.get("quad", box.get("polygon_points", box["coordinate"]))
            return box["label"], tuple(np.asarray(shape).ravel())

        present = {geometry_key(box) for box in boxes}
        candidates = [box for box in candidates if geometry_key(box) not in present]

    entries = []
    split_records = {}
    for orig_idx, box in enumerate(boxes):
        if box["label"] != "display_formula" or orig_idx in (keep_whole or ()):
            entries.append((orig_idx, box))
            continue
        quad = _extract_region_quad(box)
        if quad is None:
            entries.append((orig_idx, box))
            continue
        width = int(
            round(
                max(
                    np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[3])
                )
            )
        )
        height = int(
            round(
                max(
                    np.linalg.norm(quad[3] - quad[0]), np.linalg.norm(quad[2] - quad[1])
                )
            )
        )
        if min(width, height) < 8 or width * height > 4 * page_h * page_w:
            entries.append((orig_idx, box))
            continue
        dst = np.float32(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
        )
        transform = cv2.getPerspectiveTransform(quad, dst)
        crop = cv2.warpPerspective(
            image, transform, (width, height), borderValue=(255, 255, 255)
        )
        gray = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        analysis = _analyze_ink(gray)
        if analysis is None:
            entries.append((orig_idx, box))
            continue

        unwarp_cache = {}
        detected = _candidate_lines(
            gray,
            transform,
            box,
            candidates or [],
            analysis=analysis,
            unwarp_cache=unwarp_cache,
        )
        if detected is not None:
            children, row_spans = detected
            split_records[orig_idx] = dict(
                parent_box=box,
                children=children,
                transform=transform,
                width=width,
                height=height,
                row_spans=row_spans,
            )
            entries.append((orig_idx, box))
            continue

        split_out = _line_rectangles(
            gray, analysis=analysis, allow_unwarp=True, unwarp_cache=unwarp_cache
        )
        if split_out is None:
            entries.append((orig_idx, box))
            continue
        rects, curved_geoms = split_out
        inverse = cv2.getPerspectiveTransform(dst, quad)
        children = []
        row_spans = []
        if rects is not None:
            for left, top, right, bottom in rects:
                points, coord = _local_to_page(
                    [
                        [left, top],
                        [right - 1, top],
                        [right - 1, bottom - 1],
                        [left, bottom - 1],
                    ],
                    inverse,
                    page_w,
                    page_h,
                )
                if coord is None:
                    children = []
                    break
                child = box.copy()
                child["coordinate"] = coord
                if "quad" in child:
                    child["quad"] = points.tolist()
                if "polygon_points" in child:
                    child["polygon_points"] = points.tolist()
                children.append(child)
                row_spans.append((float(top), float(bottom)))
        else:
            for local_quad, _, r_top, r_bot in curved_geoms:
                points, coord = _local_to_page(local_quad, inverse, page_w, page_h)
                if coord is None:
                    children = []
                    break
                child = box.copy()
                child["coordinate"] = coord
                if "quad" in child:
                    child["quad"] = points.tolist()
                if "polygon_points" in child:
                    child["polygon_points"] = points.tolist()
                children.append(child)
                row_spans.append((r_top, r_bot))

        if len(children) < 2:
            entries.append((orig_idx, box))
            continue
        split_records[orig_idx] = dict(
            parent_box=box,
            children=children,
            transform=transform,
            width=width,
            height=height,
            row_spans=row_spans,
        )
        entries.append((orig_idx, box))

    if not split_records:
        return [box for _, box in entries]
    if any(box["label"] == "formula_number" for _, box in entries):
        return _interleave_formula_numbers(entries, split_records, boxes)

    result = []
    for orig_idx, box in entries:
        if orig_idx in split_records:
            result.extend(split_records[orig_idx]["children"])
        else:
            result.append(box)
    return result
