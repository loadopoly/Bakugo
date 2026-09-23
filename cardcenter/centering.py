"""The measurement pipeline.

    image
      -> outer card quad (subpixel, via line fits)
      -> camera pose (needs focal length)
      -> iterative refraction solve
      -> rectify
      -> inner frame detection per side
      -> map back through refraction to TRUE card coordinates
      -> border widths, ratios, error budget

The iteration in step 3 exists because of a circularity: the card's own corners
are displaced by refraction, so the homography we fit from them is itself
contaminated, so the pose we derive is slightly wrong, so the displacement we
compute is slightly wrong. Three passes converge to well under a micron of
change for any realistic slab, which is far below anything else in the budget.

Scale invariance note: the centering *ratio* is L/(L+R), which is invariant to
the overall scale factor. So an error in the assumed physical card size cancels
out entirely and does not appear in the error budget. Errors in locating either
individual edge do not cancel, and those dominate.
"""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

from .detect import SIDES, SideProfile, detect_all_borders
from .illumination import correct_quad_for_shadow, detect_edge_shadow
from .geometry import (
    apply_h,
    card_plane_corners_mm,
    enforce_portrait,
    find_card_quad,
    order_quad,
    rectify,
)
from .optics import (
    CameraPose,
    inplane_shift_measured,
    inplane_shift_mm,
    pose_from_homography,
)
from .information import measure_channel
from .types import (
    STANDARD_CARD_H_MM,
    STANDARD_CARD_W_MM,
    BorderPair,
    CaptureSpec,
    CenteringResult,
    DetectionError,
    DetectionQuality,
    Measured,
    SlabSpec,
    SLAB_PRESETS,
)

# Used only when the camera's focal length is unknown and the card is slabbed.
# Deliberately pessimistic: we would rather report a wide band than a confident
# wrong one.
ASSUMED_TILT_DEG = 20.0
ASSUMED_TILT_SIGMA_DEG = 15.0


def _sleeve_margin(P, prof, g, depth, bg, j, k, s) -> Optional[float]:
    """Where the card starts inside a sleeve's margin, if the outline is on
    the sleeve (mm from the outline), else None.

    A penny sleeve runs past the card: ~2 mm below it on the Meganium field
    frame, ~1.2 mm above at the open end. The detectors outline the sleeve
    (it is the outermost clean edge), and the border widths measured from it
    are the sleeve's margin plus the card's border: 4.6 mm at the bottom
    where the card's is ~2.3, and the photo measured 77/23. Through the
    plastic the margin is the background, lightened: the background's hue,
    a lightness less than 0.6 of the way to the card's, then a step onto the
    card that is stronger than the first. The colour alone is not enough: a
    silver border in shade beside a green frame passed it on this frame; the
    step strength is what keeps this off the printed border when the outline
    is already on the card. Asked whatever holder is selected (a sleeve is
    easy to forget in the settings): the 144 synthetic raw captures measure
    as before, and the Meganium frame measures the same with the holder set
    to raw."""
    first_end = float(depth[j])
    w2 = (depth >= first_end + 0.3) & (depth <= 3.5)
    if not w2.any():
        return None
    k2 = int(np.argmax(np.where(w2, g, -1e9)))
    # The card's own edge is the strongest step in the profile. Outlined on
    # the card, the step after it (border to printed frame) is weaker; on a
    # sleeve, the sleeve's thin edge is the weaker one and the card's comes
    # second. (Meganium: sleeve 38 and 48 then card 42 and 81 at the top and
    # bottom; on the card already, 67 and 145 then 54 and 77 at the sides.)
    if g[k2] < max(g[k], 25.0):
        return None
    d2 = float(depth[k2])
    col = np.median(P, axis=1)                                    # depth x 3
    band_m = (depth >= first_end + 0.05) & (depth <= d2 - 0.25)
    card_m = (depth >= d2 + 0.25) & (depth <= d2 + 0.55)
    if band_m.sum() < 3 or card_m.sum() < 2:
        return None
    band = np.median(col[band_m], axis=0)
    card = np.median(col[card_m], axis=0)
    bg = np.median(np.asarray(bg).reshape(-1, 3), axis=0)         # one colour
    # the band keeps the background's colour and sits nearer its lightness
    # than the card's; a card border beside coloured artwork (silver next to
    # a green frame) is as light as the card and fails this
    dl_card = float(card[0] - bg[0])
    dl_band = float(band[0] - bg[0])
    if not (abs(dl_card) >= 15.0 and dl_band * dl_card > 0
            and abs(dl_band) <= 0.6 * abs(dl_card)
            and float(np.linalg.norm(band[1:] - bg[1:])) <= 12.0
            and float(np.linalg.norm(card - band)) >= 12.0):
        return None
    half = 0.5 * float(np.linalg.norm(card - band))
    near = (depth >= d2 - 0.5) & (depth <= d2 + 0.5)

    def cross(p):
        idx = np.nonzero((p[:-1] < half) & (p[1:] >= half) & near[:-1])[0]
        if len(idx) == 0:
            return float("nan")
        i = int(idx[0])
        return float(depth[i] + (half - p[i]) / max(float(p[i + 1] - p[i]), 1e-6) / s)

    db = np.linalg.norm(P - band[None, None, :], axis=2)
    edge = cross(np.median(db, axis=1))
    if not np.isfinite(edge):
        return None
    per = np.array([cross(db[:, c]) for c in range(db.shape[1])])
    if float(np.mean(np.abs(per - edge) <= 0.35)) < 0.6:
        return None
    return edge


def seat_outer_edges(image: np.ndarray, corners: np.ndarray, px_per_mm: float,
                     pad_mm: float = 1.2, sleeved: bool = True) -> tuple[np.ndarray, dict]:
    """Put each side of the outline on the card's own edge.

    Every border width is measured FROM the outline, so an outline sitting
    half a millimetre off the card on one side moves that border by half a
    millimetre -- on a 2 mm border, 10 points of centering. The detectors
    place a side on the strongest gradient near it, which is the card's edge
    when that edge is sharp. When it is not -- defocus, a penny sleeve's open
    end, a soft shadow -- the edge is a ramp up to a millimetre wide and the
    side can sit anywhere on it. A live frame from a counter (Meganium in a
    penny sleeve, 2.18.0) had its top side 0.4 mm out, on the start of the
    ramp; the top border then measured 0.7 mm where it is ~2.5, and the app
    settled on 74/26 for a card that is about 51/49.

    For each side, the colour profile across the edge (median along the
    middle 70% of the side) is taken from 1.2 mm outside to 1.6 mm inside,
    and the edge is put where the colour is half way from the background just
    outside to the card just past the ramp. A side moves only when the step
    is clear (the card differs from the background by dE >= 12, the
    background band is flat) and at least 60% of positions along the side
    agree to within 0.3 mm. Returns (corners, {side: shift_mm inward}).

    Tried as well: starting the printed-border search past the end of the
    ramp instead of 0.35 mm in. It made no difference on the Meganium frame
    and made thin borders under heavy blur worse (31 against 16 of 144
    synthetic captures more than 3 points out), so it is not done."""
    from .types import STANDARD_CARD_H_MM as CH, STANDARD_CARD_W_MM as CW

    corners = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    s = max(float(px_per_mm), 8.0)
    src = np.float32([[0, 0], [CW, 0], [CW, CH], [0, CH]])
    try:
        M = cv2.getPerspectiveTransform(src, corners.astype(np.float32)).astype(np.float64)
    except cv2.error:
        return corners, {}
    T = np.array([[s, 0, pad_mm * s], [0, s, pad_mm * s], [0, 0, 1.0]])
    size = (int(round((CW + 2 * pad_mm) * s)), int(round((CH + 2 * pad_mm) * s)))
    rect = cv2.warpPerspective(image, M @ np.linalg.inv(T), size,
                               flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
    lab = cv2.cvtColor(rect, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1] -= 128.0
    lab[..., 2] -= 128.0
    # each side turned so axis 0 runs inward; depth in mm from the side, from
    # the pixel's own position (a flipped axis is not the same grid)
    rows, cols = lab.shape[:2]
    pos_y = np.arange(rows) / s - pad_mm
    pos_x = np.arange(cols) / s - pad_mm
    views = {"top": (lab, pos_y), "bottom": (lab[::-1], CH - pos_y[::-1]),
             "left": (lab.transpose(1, 0, 2), pos_x),
             "right": (lab.transpose(1, 0, 2)[::-1], CW - pos_x[::-1])}
    reach = int(round((pad_mm + 4.2) * s))
    shift: dict = {}
    for side, (v, dd) in views.items():
        n = v.shape[1]
        P = np.ascontiguousarray(v[:reach, int(0.15 * n):int(0.85 * n)])
        P = cv2.GaussianBlur(P, (0, 0), sigmaX=max(0.6, 0.3 * s), sigmaY=max(0.5, 0.06 * s))
        depth = dd[:reach]
        bgm = (depth >= -0.8) & (depth <= -0.4)
        bg = np.median(P[bgm], axis=0)
        de = np.linalg.norm(P - bg[None], axis=2)
        prof = np.median(de, axis=1)
        g = np.gradient(prof) * s
        win = (depth >= -0.4) & (depth <= 1.2)
        k = int(np.argmax(np.where(win, g, -1e9)))
        j = k
        while j < len(g) - 1 and not (g[j] < 0.3 * g[k] and depth[j] - depth[k] > 0.05):
            j += 1
        plateau = float(np.median(prof[j:j + max(2, int(0.2 * s))]))
        noise = float(np.median(prof[bgm]))
        half = 0.5 * (noise + plateau)

        def cross(p):
            idx = np.nonzero((p[:-1] < half) & (p[1:] >= half) & win[:-1])[0]
            if len(idx) == 0:
                return float("nan")
            i = int(idx[0])
            return float(depth[i] + (half - p[i]) / max(float(p[i + 1] - p[i]), 1e-6) / s)

        edge = cross(prof)
        if not np.isfinite(edge):
            shift[side] = 0.0
            continue
        per = np.array([cross(de[:, c]) for c in range(de.shape[1])])
        agree = float(np.mean(np.abs(per - edge) <= 0.3))
        ok = plateau >= 12.0 and noise <= 0.25 * plateau and agree >= 0.6
        # Inward from 0.12 mm (the outline off the card, the failure seen);
        # outward only from 0.25: on a thin border under heavy blur the
        # card's edge ramp runs into the border's own and the half-way point
        # slides out by ~0.14 mm on an outline that was right (synthetic, 1.0
        # and 1.2 mm borders at 3 px of blur).
        shift[side] = edge if ok and (edge >= 0.12 or edge <= -0.25) else 0.0
        sleeve = _sleeve_margin(P, prof, g, depth, bg, j, k, s) if ok and sleeved else None
        if sleeve is not None:
            shift[side] = sleeve
    if not any(shift.values()):
        return corners, shift
    l, t, r, b = (shift.get(k, 0.0) for k in ("left", "top", "right", "bottom"))
    mm = np.float32([[l, t], [CW - r, t], [CW - r, CH - b], [l, CH - b]]).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(mm, M.astype(np.float32)).reshape(4, 2).astype(np.float64)
    return out, shift


def _solve_apparent_corners(
    image_corners: np.ndarray,
    K: np.ndarray,
    slab: SlabSpec,
    iterations: int = 3,
) -> tuple[np.ndarray, Optional[CameraPose]]:
    """Find where the card's true corners *appear* to be, in card mm coords."""
    nominal = card_plane_corners_mm()
    apparent = nominal.copy()
    pose: Optional[CameraPose] = None

    for _ in range(iterations):
        H = cv2.getPerspectiveTransform(
            apparent.astype(np.float32), image_corners.astype(np.float32)
        ).astype(np.float64)
        p = pose_from_homography(H, K)
        if p is None:
            break
        pose = p
        if not slab.is_optically_active:
            break
        theta = pose.incidence_angles(nominal)
        mag = np.asarray(inplane_shift_mm(theta, slab), dtype=float)
        u = pose.inplane_directions(nominal)
        apparent = nominal - mag[:, None] * u

    return apparent, pose


def _pick_scale(image_corners: np.ndarray) -> float:
    """Rectification scale, chosen to neither throw away detail nor invent it."""
    w_px = 0.5 * (
        np.linalg.norm(image_corners[1] - image_corners[0])
        + np.linalg.norm(image_corners[2] - image_corners[3])
    )
    h_px = 0.5 * (
        np.linalg.norm(image_corners[3] - image_corners[0])
        + np.linalg.norm(image_corners[2] - image_corners[1])
    )
    scale = 0.5 * (w_px / STANDARD_CARD_W_MM + h_px / STANDARD_CARD_H_MM)
    return float(np.clip(scale, 6.0, 40.0))


def _side_sample_point_rect(
    side: str, depth_mm: float, px_per_mm: float
) -> np.ndarray:
    """Midpoint of a detected border edge, in rectified pixel coords."""
    w = STANDARD_CARD_W_MM * px_per_mm
    h = STANDARD_CARD_H_MM * px_per_mm
    d = depth_mm * px_per_mm
    if side == "left":
        return np.array([[d, h / 2.0]])
    if side == "right":
        return np.array([[w - 1.0 - d, h / 2.0]])
    if side == "top":
        return np.array([[w / 2.0, d]])
    if side == "bottom":
        return np.array([[w / 2.0, h - 1.0 - d]])
    raise ValueError(side)


def measure_centering(
    image: np.ndarray,
    slab: SlabSpec | str = "raw",
    capture: Optional[CaptureSpec] = None,
    keep_rectified: bool = True,
    card_quad: Optional[np.ndarray] = None,
    quad_residual_px: float = 0.0,
) -> CenteringResult:
    """Measure a card's centering from a single image.

    ``card_quad`` lets a caller supply an already-located card boundary, in
    TL/TR/BR/BL order. The multi-card scanner and the live stream both locate
    cards in the full frame, where there is far more context than in a tight
    crop; re-running detection on the crop discards that and routinely finds a
    worse boundary. Pass ``quad_residual_px`` alongside it so the edge-location
    uncertainty stays honest.

    Raises DetectionError rather than returning a low-confidence number.
    """
    if isinstance(slab, str):
        if slab not in SLAB_PRESETS:
            raise KeyError(
                f"unknown slab preset '{slab}'. Available: {', '.join(SLAB_PRESETS)}"
            )
        slab = SLAB_PRESETS[slab]
    capture = capture or CaptureSpec()

    if image is None or image.size == 0:
        raise DetectionError("empty image")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    quality = DetectionQuality()

    # --- 1. Outer boundary -------------------------------------------------
    if card_quad is not None:
        image_corners = enforce_portrait(
            order_quad(np.asarray(card_quad, dtype=np.float64).reshape(4, 2))
        )
        outer_residual_px = float(quad_residual_px)
    else:
        # A single-card measurement is a photograph OF a card, and the subject
        # of a photograph is the thing the photographer framed: the large
        # object containing the centre. Without that prior the detector treats
        # every card-shaped region equally, which is fine on a plain desk and
        # useless in the real setting -- a card held over a bin, a binder page,
        # or a shop's display, where the background is made ENTIRELY of other
        # cards and a corner artifact can outscore the subject. The multi-card
        # scanner passes card_quad explicitly and is unaffected.
        h, w = image.shape[:2]
        image_corners, _contour, outer_residual_px = find_card_quad(
            image, prefer_point=(w / 2.0, h / 2.0)
        )
    quality.outer_residual_px = outer_residual_px

    px_per_mm = _pick_scale(image_corners)

    # A card is ~0.3mm thick and obliquely lit it shadows its own edge, on one
    # side only. Measured in simulation: at 30 degrees of light elevation a
    # perfectly centred card reads 82.8/17.2, and the error bar misses truth by
    # seven sigma. It is the single largest failure mode in this package and it
    # is caused entirely by where the lamp is, so it is checked before anything
    # else is believed.
    shadow = detect_edge_shadow(image, image_corners, px_per_mm)
    shadow_sigma_mm = 0.0
    if shadow.directional and shadow.correctable:
        image_corners = correct_quad_for_shadow(
            image_corners, shadow.side_index, shadow.estimated_shadow_mm, px_per_mm
        )
        px_per_mm = _pick_scale(image_corners)
        # The correction is a measurement too, and a coarse one: the band's
        # inner boundary is soft because a penumbra is not a step. Charge a
        # third of the correction as its own uncertainty on that border.
        shadow_sigma_mm = 0.33 * shadow.estimated_shadow_mm
        quality.warnings.append(
            f"edge shadow on the {shadow.darker_side} "
            f"({shadow.estimated_shadow_mm:.2f}mm, light near "
            f"{shadow.estimated_elevation_deg:.0f} deg) was detected and "
            "subtracted from the boundary. This bias is one-sided and does not "
            "cancel in the ratio, so the correction carries its own error term."
        )
    elif shadow.directional:
        qualifier = "an at-least " if shadow.shadow_unbounded else "a "
        raise DetectionError(
            f"{qualifier}{shadow.estimated_shadow_mm:.2f}mm edge shadow on the "
            f"{shadow.darker_side} is too wide to subtract reliably -- it "
            "overlaps the printed border. Change your angle relative to the "
            "light, or shoot the card from a different side of the case."
        )
    # After the shadow check, which models a one-sided shadow band and
    # charges its correction as uncertainty; what is left is an outline on a
    # soft edge (focus, a sleeve's open end).
    image_corners, seated = seat_outer_edges(image, image_corners, px_per_mm)
    moved = {k: v for k, v in seated.items() if abs(v) >= 0.2}
    if moved:
        px_per_mm = _pick_scale(image_corners)
        quality.warnings.append(
            "outline moved onto the card's edge ("
            + ", ".join(f"{k} {v:+.2f}mm" for k, v in moved.items())
            + "): the edge there is soft (focus, or a sleeve)"
        )
    if outer_residual_px > 2.0:
        quality.warnings.append(
            f"card edges deviate {outer_residual_px:.1f}px from straight lines. "
            "The card may be bent, or the background is bleeding into the edge."
        )

    # --- 2. Pose and refraction geometry -----------------------------------
    K = capture.intrinsics(image.shape)
    pose: Optional[CameraPose] = None
    apparent_corners = card_plane_corners_mm()

    if K is not None:
        apparent_corners, pose = _solve_apparent_corners(image_corners, K, slab)
        if pose is None:
            quality.warnings.append(
                "homography decomposition failed; refraction correction skipped"
            )
    elif slab.is_optically_active:
        quality.warnings.append(
            "no focal length supplied, so camera tilt is unknown and the "
            "refraction correction through the slab cannot be computed. "
            "Uncertainty has been inflated to cover the plausible range. "
            "Pass --fov or --focal-px for a real correction."
        )

    H_app_to_image = cv2.getPerspectiveTransform(
        apparent_corners.astype(np.float32), image_corners.astype(np.float32)
    ).astype(np.float64)
    H_image_to_app = np.linalg.inv(H_app_to_image)

    # --- 3. Rectify --------------------------------------------------------
    rect, _ = rectify(image, image_corners, px_per_mm)
    M_rect = cv2.getPerspectiveTransform(
        image_corners.astype(np.float32),
        np.array(
            [
                [0, 0],
                [STANDARD_CARD_W_MM * px_per_mm - 1, 0],
                [STANDARD_CARD_W_MM * px_per_mm - 1, STANDARD_CARD_H_MM * px_per_mm - 1],
                [0, STANDARD_CARD_H_MM * px_per_mm - 1],
            ],
            dtype=np.float32,
        ),
    ).astype(np.float64)
    M_rect_inv = np.linalg.inv(M_rect)

    # --- 4. Inner frame ----------------------------------------------------
    profiles: dict[str, SideProfile] = detect_all_borders(rect, px_per_mm)
    quality.inner_confidence_per_side = {s: p.confidence for s, p in profiles.items()}
    quality.inner_confidence = float(min(p.confidence for p in profiles.values()))

    for s, p in profiles.items():
        rot_mm_per_mm = abs(p.slope_mm_per_mm)
        if rot_mm_per_mm > 0.012:  # ~0.7 degrees of print rotation
            quality.warnings.append(
                f"'{s}' border width drifts {rot_mm_per_mm * 100:.1f}% along the side; "
                "the printing appears rotated relative to the cut. Centering on a "
                "rotated print is genuinely ambiguous and graders may disagree."
            )
        if p.rejected_frac > 0.3:
            quality.warnings.append(
                f"'{s}' border: {p.rejected_frac * 100:.0f}% of scan columns rejected "
                "as outliers (text or artwork intruding into the border)."
            )

    # --- 5. Map border points to TRUE card coordinates ---------------------
    outer_sigma_mm = max(outer_residual_px, 0.3) / px_per_mm
    max_shift = 0.0
    true_edges: dict[str, Measured] = {}

    for side in SIDES:
        prof = profiles[side]
        pt_rect = _side_sample_point_rect(side, prof.depth_mm, px_per_mm)
        pt_img = apply_h(M_rect_inv, pt_rect)
        pt_app = apply_h(H_image_to_app, pt_img)

        refract_sigma_mm = 0.0
        if slab.is_optically_active:
            if pose is not None:
                theta = float(pose.incidence_angles(pt_app)[0])
                shift = inplane_shift_measured(theta, slab, math.radians(1.5))
                u = pose.inplane_directions(pt_app)[0]
                pt_true = pt_app + shift.value * u
                refract_sigma_mm = shift.sigma
                max_shift = max(max_shift, abs(shift.value))
                quality.refraction_applied = True
            else:
                # Unknown geometry: no correction, but the shift that *might*
                # be there becomes uncertainty.
                unknown = inplane_shift_measured(
                    math.radians(ASSUMED_TILT_DEG),
                    slab,
                    math.radians(ASSUMED_TILT_SIGMA_DEG),
                )
                pt_true = pt_app
                refract_sigma_mm = math.hypot(unknown.value, unknown.sigma)
                max_shift = max(max_shift, abs(unknown.value))
        else:
            pt_true = pt_app

        x, y = float(pt_true[0][0]), float(pt_true[0][1])
        if side == "left":
            width = x
        elif side == "right":
            width = STANDARD_CARD_W_MM - x
        elif side == "top":
            width = y
        else:
            width = STANDARD_CARD_H_MM - y

        sigma = math.sqrt(
            prof.sigma_mm**2
            + refract_sigma_mm**2
            + outer_sigma_mm**2
            + (shadow_sigma_mm if side == shadow.darker_side else 0.0) ** 2
        )
        if width <= 0:
            raise DetectionError(
                f"'{side}' border measured as {width:.2f}mm (non-positive). "
                "The outer boundary detection is almost certainly wrong."
            )
        true_edges[side] = Measured(width, sigma)

    quality.max_refraction_shift_mm = max_shift

    if quality.inner_confidence < 0.35:
        weakest = min(profiles.values(), key=lambda p: p.confidence)
        raise DetectionError(
            f"border detection confidence too low to report a measurement "
            f"(weakest side '{weakest.side}' at {weakest.confidence:.2f}). "
            "Re-shoot with even, diffuse lighting and no glare, or measure "
            "this card by hand."
        )

    horizontal = BorderPair(
        axis="horizontal",
        low_name="left",
        high_name="right",
        low_mm=true_edges["left"],
        high_mm=true_edges["right"],
    )
    vertical = BorderPair(
        axis="vertical",
        low_name="top",
        high_name="bottom",
        low_mm=true_edges["top"],
        high_mm=true_edges["bottom"],
    )

    # Pixel-space channel on the worst-axis low side. Failure here must
    # never sink a measurement that already cleared the confidence gate.
    channel = None
    try:
        pair = max((horizontal, vertical), key=lambda p: p.ratio_pct.value)
        channel = measure_channel(rect, pair.low_name, px_per_mm, pair.low_mm.value)
    except Exception:
        channel = None

    return CenteringResult(
        horizontal=horizontal,
        vertical=vertical,
        quality=quality,
        px_per_mm=px_per_mm,
        corners_px=image_corners,
        inner_rect_mm=(
            true_edges["left"].value,
            true_edges["top"].value,
            STANDARD_CARD_W_MM - true_edges["right"].value,
            STANDARD_CARD_H_MM - true_edges["bottom"].value,
        ),
        slab=slab,
        rectified=rect if keep_rectified else None,
        channel=channel,
    )
