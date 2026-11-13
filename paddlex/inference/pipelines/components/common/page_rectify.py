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

"""Page rectification from the layout model's ``warp_params``.

``warp_params`` = ``(roll_deg, quadrant_conf, shear_deg, kx, ky)``.  The four
geometric numbers are *rectifying* values defined on the network input, i.e. the
page squashed to a square (``Resize(keep_ratio=False)``), so the correction is::

    H = S_out^-1 @ T(+c) @ U(shear) @ R(roll) @ P(kx, ky) @ T(-c) @ S_in

with ``S_in`` the squash of the input to a ``side x side`` square, ``c`` its
centre, and ``S_out`` the same squash for the *rectified* page -- which has the
input's W and H swapped when roll is an odd number of quarter turns.  This is a
port of ``build_rectify`` in LayoutModel's ``tools/rectify_page.py`` -- same
order (RQ), same thresholds, same degeneracy floor -- minus the whole-page
re-framing (``_fit`` / ``max_scale``), because the cropper applies this one page
matrix to every block and frames each block itself.  Ported rather than imported:
that script lives in the training repo and sets ``cv2.setNumThreads(0)`` at import.

``plan_page_correction`` decides *whether* to apply it.  The predicted fine
deformation (residual roll, shear, keystone) is only worth a resampling when it
clearly exceeds the model's own error; below that the matrix only perturbs the
pixels handed to the recogniser.  A quarter turn is different: it is never noise,
and it can be undone exactly without interpolation.
"""

import math
from typing import NamedTuple, Optional, Sequence

import numpy as np

# Gate defaults, in degrees.  On wild_omnidocbench the measured tilt of blocks on
# pages that are not a quarter turn has median 0.3-0.5 deg; correcting them made
# 47.7% of blocks straighter, i.e. no better than chance.  From 2 deg of measured
# tilt on, 87% or more of pages come out straighter.
DEFAULT_CONF_THRESH = 0.9
DEFAULT_MIN_ROLL = 2.0
DEFAULT_MIN_SHEAR = 2.0
DEFAULT_MIN_KEYSTONE = 2.0

# Smallest projective denominator allowed anywhere on the canvas.  The denominator
# is 1 + kx*u/W + ky*v/H with (u, v) measured from the canvas centre, so it bottoms
# out at 1 - (|kx| + |ky|)/2; where it reaches zero is the vanishing line.  The
# floor caps the local magnification near 6.7x: hitting it under-corrects, which
# beats folding the page over itself.
_MIN_DENOM = 0.15


def _unpack(warp_params: Sequence[float]):
    """``(roll, conf, shear, kx, ky)`` as floats, missing trailing terms as 0.

    Older layout models emit only ``(roll, conf)``; padding the rest with zeros --
    the same convention as the reference ``rectify_page.py`` -- turns them into a
    pure rotation.  Fewer than two values cannot carry a confidence and are
    rejected.
    """
    vals = [float(v) for v in list(warp_params)]
    if len(vals) < 2:
        raise ValueError(
            f"warp_params needs at least (roll, conf), got {len(vals)} value(s)."
        )
    return tuple((vals + [0.0] * 5)[:5])


def _projective(kx: float, ky: float, w: float, h: float) -> np.ndarray:
    """``[[1,0,0],[0,1,0],[kx/W, ky/H, 1]]``, scaled back if it would degenerate."""
    half = 0.5 * (abs(kx) + abs(ky))
    if 1.0 - half < _MIN_DENOM and half > 1e-12:
        # Same factor on both terms: keeps the direction, only weakens it.
        s = (1.0 - _MIN_DENOM) / half
        kx, ky = kx * s, ky * s
    m = np.eye(3, dtype=np.float64)
    m[2, 0], m[2, 1] = kx / w, ky / h
    return m


def build_page_homography(
    shape: Sequence[int],
    warp_params: Sequence[float],
    min_roll: float = 0.3,
    min_shear: float = 0.3,
    min_tilt: float = 0.005,
    max_tilt: float = 1.5,
) -> Optional[np.ndarray]:
    """Rectifying homography in the pixels of an image of ``shape``, or None.

    Args:
        shape: ``(H, W, ...)`` of the image the layout model was run on.
        warp_params: ``(roll_deg, quadrant_conf, shear_deg, kx, ky)``; trailing
            terms may be missing (older ``(roll, conf)`` models) and count as 0.
        min_roll, min_shear, min_tilt: each stage is skipped below its threshold.
        max_tilt: refuses the projective stage for out-of-range keystone.

    Returns:
        A 3x3 float64 matrix mapping original pixels to rectified pixels (up to a
        translation and uniform scale, which the cropper chooses per block), or
        None when every stage is below threshold, i.e. nothing to correct.
    """
    roll, _conf, shear, kx, ky = _unpack(warp_params)
    h, w = shape[:2]
    side = float(max(w, h))

    # Right to left: projectivity, then roll, then shear -- the RQ order.
    m = np.eye(3, dtype=np.float64)
    if min_tilt <= max(abs(kx), abs(ky)) <= max_tilt:
        m = _projective(kx, ky, side, side)
    if abs(roll) >= min_roll:
        a = math.radians(roll)
        c, s = math.cos(a), math.sin(a)
        m = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ m
    if abs(shear) >= min_shear:
        t = math.tan(math.radians(shear))
        m = np.array([[1.0, t, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]) @ m

    if np.allclose(m, np.eye(3)):
        return None

    # Defined about the centre of the square network input: squash the input
    # (S_in), conjugate by the centre, then unsquash to the rectified page's
    # aspect (S_out).  After a quarter turn the upright page's width is the
    # input's height, so an odd quadrant swaps the two; unsquashing with the
    # input's own W/H would flatten every block by (W/H)^2 across its lines.
    odd_quadrant = int(round(roll / 90.0)) % 2 == 1
    out_w, out_h = (h, w) if odd_quadrant else (w, h)
    sq_in = np.diag([side / w, side / h, 1.0])
    sq_out = np.diag([side / out_w, side / out_h, 1.0])
    ctr = np.array([[1.0, 0.0, -side / 2.0], [0.0, 1.0, -side / 2.0], [0.0, 0.0, 1.0]])
    return np.linalg.inv(sq_out) @ np.linalg.inv(ctr) @ m @ ctr @ sq_in


class PageCorrection(NamedTuple):
    """What the cropper should do to every block of one page.

    Exactly one of the two is set: ``homography`` is the full rectification (the
    quarter turn included); ``rot90_k`` in {1, 2, 3} is a bare quarter turn, undone
    per block with ``np.rot90(crop, rot90_k)`` -- an exact pixel permutation, no
    interpolation.  ``rot90_k = -q mod 4`` for a roll of ``q`` quarter turns, which
    reproduces exactly what ``build_page_homography`` does for a pure quarter turn
    (checked pixel for pixel in the tests).
    """

    homography: Optional[np.ndarray] = None
    rot90_k: int = 0


def split_roll(roll: float):
    """``roll = 90 * q + residual`` with ``q`` in {0..3} and residual in [-45, 45]."""
    n = int(round(float(roll) / 90.0))
    return n % 4, float(roll) - 90.0 * n


def keystone_deg(shape: Sequence[int], kx: float, ky: float) -> float:
    """The keystone as an angle: how far correcting it turns the page's border.

    ``kx`` tilts the top and bottom edges by about ``atan(kx * H / 2W)``, ``ky``
    the left and right edges by ``atan(ky * W / 2H)``; the larger is returned.
    The border is the most affected place on the page, so this is an upper bound
    on what any block experiences.
    """
    h, w = shape[:2]
    return math.degrees(
        max(math.atan(abs(kx) * h / (2.0 * w)), math.atan(abs(ky) * w / (2.0 * h)))
    )


def plan_page_correction(
    shape: Sequence[int],
    warp_params: Sequence[float],
    conf_thresh: float = DEFAULT_CONF_THRESH,
    min_roll: float = DEFAULT_MIN_ROLL,
    min_shear: float = DEFAULT_MIN_SHEAR,
    min_keystone: float = DEFAULT_MIN_KEYSTONE,
) -> Optional[PageCorrection]:
    """Decide how to correct one page, or None to keep the plain crops.

    1. ``conf < conf_thresh``: None -- a wrong quarter turn hurts recognition far
       more than no correction.
    2. Split roll into a quarter turn ``q`` and a residual.  The fine deformation
       is significant when ``|residual| >= min_roll``, ``|shear| >= min_shear`` or
       ``keystone_deg >= min_keystone``.
    3. Significant: the full homography with all four values as predicted -- they
       are one joint estimate, so a subset would be a correction the model never
       predicted.  Otherwise, with ``q != 0``: the bare quarter turn.  Otherwise
       None.

    Setting all three minima to ``inf`` leaves quarter turns only.  Non-finite
    values give None.  A two-value ``(roll, conf)`` prediction from an older model
    is a pure rotation (shear and keystone 0), so it is gated on the residual roll
    alone.
    """
    roll, conf, shear, kx, ky = _unpack(warp_params)
    if not all(math.isfinite(v) for v in (roll, conf, shear, kx, ky)):
        return None  # a broken prediction: keep the plain crops
    if conf < conf_thresh:
        return None
    q, residual = split_roll(roll)
    significant = (
        abs(residual) >= min_roll
        or abs(shear) >= min_shear
        or keystone_deg(shape, kx, ky) >= min_keystone
    )
    if significant:
        homography = build_page_homography(
            shape, warp_params, min_roll=0.0, min_shear=0.0, min_tilt=0.0
        )
        if homography is not None:
            return PageCorrection(homography=homography)
    if q:
        return PageCorrection(rot90_k=(-q) % 4)
    return None
