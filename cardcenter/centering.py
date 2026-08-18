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
        image_corners, _contour, outer_residual_px = find_card_quad(image)
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
        raise DetectionError(
            f"a {shadow.estimated_shadow_mm:.2f}mm edge shadow on the "
            f"{shadow.darker_side} is too wide to subtract reliably -- it "
            "overlaps the printed border. Change your angle relative to the "
            "light, or shoot the card from a different side of the case."
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
    )
