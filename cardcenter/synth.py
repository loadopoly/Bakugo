"""Synthetic cards with known ground truth.

A grading tool that has never been checked against a card whose true centering
is known is just a program that returns numbers. This module renders cards with
exactly specified border widths, viewed from an exactly specified camera pose,
optionally through a simulated slab, so the pipeline's output can be compared
to truth.

SCOPE OF WHAT THIS VALIDATES -- read this before quoting any accuracy figure.

The forward refraction render uses the same displacement formula as the solver.
So these tests do NOT independently confirm the physics; they confirm the
*geometric chain* around it: homography fitting, pose decomposition, the
apparent-corner iteration, and the coordinate mapping back to true card space.
Those are where implementation bugs actually live, and a sign error or a
transposed axis in any of them would show up here immediately.

The displacement formula itself is validated separately, and genuinely
independently, by brute-force ray tracing in tests/test_optics.py.

What none of this validates: real cards. Print variation, holo foil, glare,
rounded die-cut corners, and border artwork that shades continuously into the
art are all absent here. Synthetic accuracy is an upper bound on real accuracy,
never an estimate of it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .optics import inplane_shift_mm
from .types import STANDARD_CARD_H_MM, STANDARD_CARD_W_MM, SlabSpec, SLAB_PRESETS


@dataclass(frozen=True)
class GroundTruth:
    left_mm: float
    top_mm: float
    right_mm: float
    bottom_mm: float
    tilt_deg: float
    azimuth_deg: float
    slab_name: str

    @property
    def h_ratio(self) -> float:
        tot = self.left_mm + self.right_mm
        return 100.0 * max(self.left_mm, self.right_mm) / tot

    @property
    def v_ratio(self) -> float:
        tot = self.top_mm + self.bottom_mm
        return 100.0 * max(self.top_mm, self.bottom_mm) / tot

    @property
    def worst_ratio(self) -> float:
        return max(self.h_ratio, self.v_ratio)


def make_card_texture(
    left_mm: float,
    top_mm: float,
    right_mm: float,
    bottom_mm: float,
    px_per_mm: float = 20.0,
    border_bgr: tuple[int, int, int] = (60, 200, 240),
    art_bgr: tuple[int, int, int] = (120, 70, 45),
    add_border_text: bool = False,
    seed: int = 0,
) -> np.ndarray:
    """Render a fronto-parallel card with exact border widths."""
    rng = np.random.default_rng(seed)
    w = int(round(STANDARD_CARD_W_MM * px_per_mm))
    h = int(round(STANDARD_CARD_H_MM * px_per_mm))

    card = np.zeros((h, w, 3), dtype=np.uint8)
    card[:, :] = border_bgr
    # Subtle print texture so the deltaE noise floor is not unrealistically zero.
    card = np.clip(
        card.astype(np.int16) + rng.normal(0, 2.0, card.shape).astype(np.int16), 0, 255
    ).astype(np.uint8)

    x0 = int(round(left_mm * px_per_mm))
    y0 = int(round(top_mm * px_per_mm))
    x1 = w - int(round(right_mm * px_per_mm))
    y1 = h - int(round(bottom_mm * px_per_mm))

    art = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.uint8)
    art[:, :] = art_bgr
    # Give the art internal structure so a detector cannot pass by luck.
    for i in range(0, art.shape[0], max(1, int(6 * px_per_mm))):
        cv2.line(art, (0, i), (art.shape[1], i), (200, 190, 180), 2)
    cv2.circle(
        art,
        (art.shape[1] // 2, art.shape[0] // 3),
        int(art.shape[1] * 0.28),
        (40, 40, 200),
        -1,
    )
    art = np.clip(
        art.astype(np.int16) + rng.normal(0, 3.0, art.shape).astype(np.int16), 0, 255
    ).astype(np.uint8)
    card[y0:y1, x0:x1] = art

    if add_border_text:
        # Copyright line intruding into the bottom border, and a set symbol in
        # the left border: the exact things that break a mean-based detector.
        cv2.putText(
            card,
            "(C) 2026 SYNTHETIC CARD CO.",
            (int(x0), h - int(0.35 * bottom_mm * px_per_mm)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35 * px_per_mm / 20.0 * 2.0,
            (20, 20, 20),
            max(1, int(px_per_mm / 12)),
        )
        cv2.circle(
            card,
            (int(left_mm * px_per_mm * 0.5), h // 2),
            int(min(left_mm, 2.0) * px_per_mm * 0.35),
            (30, 30, 30),
            -1,
        )
    return card


def _camera_matrices(
    tilt_deg: float,
    azimuth_deg: float,
    distance_mm: float,
    focal_px: float,
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (K, R, C) with R mapping card coords -> camera coords."""
    w_img, h_img = image_size
    K = np.array(
        [[focal_px, 0, w_img / 2.0], [0, focal_px, h_img / 2.0], [0, 0, 1]],
        dtype=np.float64,
    )

    # The card frame is x-right, y-DOWN to match image convention. That frame
    # is left-handed with respect to a viewer at +z, so a camera placed there
    # renders a mirrored card. The camera therefore sits at NEGATIVE z, which
    # makes card-x -> image-x and card-y -> image-y as a real photograph does.
    target = np.array([STANDARD_CARD_W_MM / 2.0, STANDARD_CARD_H_MM / 2.0, 0.0])
    t_rad = math.radians(tilt_deg)
    a_rad = math.radians(azimuth_deg)
    offset = distance_mm * np.array(
        [
            math.sin(t_rad) * math.cos(a_rad),
            math.sin(t_rad) * math.sin(a_rad),
            -math.cos(t_rad),
        ]
    )
    C = target + offset

    z_cam = target - C
    z_cam /= np.linalg.norm(z_cam)
    up_card = np.array([0.0, -1.0, 0.0])  # "up" on the card is -y
    if abs(float(up_card @ z_cam)) > 0.98:
        up_card = np.array([1.0, 0.0, 0.0])
    x_cam = np.cross(z_cam, up_card)
    x_cam /= np.linalg.norm(x_cam)
    y_cam = np.cross(z_cam, x_cam)
    R = np.vstack([x_cam, y_cam, z_cam])
    return K, R, C


def render_capture(
    left_mm: float = 3.0,
    top_mm: float = 3.0,
    right_mm: float = 3.0,
    bottom_mm: float = 3.0,
    tilt_deg: float = 0.0,
    azimuth_deg: float = 30.0,
    distance_mm: float = 220.0,
    focal_px: float = 2400.0,
    image_size: tuple[int, int] = (1400, 1800),
    slab: SlabSpec | str = "raw",
    background_bgr: tuple[int, int, int] = (35, 35, 38),
    noise_sigma: float = 1.5,
    vignette: float = 0.12,
    add_border_text: bool = False,
    texture_px_per_mm: float = 20.0,
    border_bgr: tuple[int, int, int] = (60, 200, 240),
    art_bgr: tuple[int, int, int] = (120, 70, 45),
    card_thickness_mm: float = 0.0,
    light_elevation_deg: float = 90.0,
    light_azimuth_deg: float = 0.0,
    shadow_darkness: float = 0.45,
    seed: int = 0,
) -> tuple[np.ndarray, GroundTruth, float]:
    """Render a photographed card. Returns (image, ground_truth, focal_px)."""
    if isinstance(slab, str):
        slab = SLAB_PRESETS[slab]
    rng = np.random.default_rng(seed)

    tex = make_card_texture(
        left_mm,
        top_mm,
        right_mm,
        bottom_mm,
        px_per_mm=texture_px_per_mm,
        border_bgr=border_bgr,
        art_bgr=art_bgr,
        add_border_text=add_border_text,
        seed=seed,
    )
    K, R, C = _camera_matrices(tilt_deg, azimuth_deg, distance_mm, focal_px, image_size)
    tvec = -R @ C
    H = K @ np.column_stack([R[:, 0], R[:, 1], tvec])  # card mm -> image px

    w_img, h_img = image_size
    out = np.empty((h_img, w_img, 3), dtype=np.uint8)
    out[:, :] = background_bgr

    # Inverse map: for each output pixel, find the card point that appears there.
    # Built by broadcasting rows against columns rather than materialising a
    # full coordinate grid, which halves peak memory on large renders.
    xs = np.arange(w_img, dtype=np.float32)
    ys = np.arange(h_img, dtype=np.float32)
    Hinv = np.linalg.inv(H).astype(np.float32)

    u = Hinv[0, 0] * xs[None, :] + Hinv[0, 1] * ys[:, None] + Hinv[0, 2]
    v = Hinv[1, 0] * xs[None, :] + Hinv[1, 1] * ys[:, None] + Hinv[1, 2]
    wgt = Hinv[2, 0] * xs[None, :] + Hinv[2, 1] * ys[:, None] + Hinv[2, 2]
    wsafe = np.where(np.abs(wgt) < 1e-12, np.float32(1e-12), wgt)
    app = np.stack([(u / wsafe).ravel(), (v / wsafe).ravel()], axis=1)
    card_h_w = wgt.ravel()

    if slab.is_optically_active:
        # Apparent -> true: move toward the camera in-plane, per equation (1).
        # The card plane is z=0, so the camera's height above it is the constant
        # C[2] and the ray geometry reduces to two in-plane differences. Doing
        # it this way avoids materialising several million-element 3-vectors.
        dx = np.float32(C[0]) - app[:, 0].astype(np.float32)
        dy = np.float32(C[1]) - app[:, 1].astype(np.float32)
        cz = np.float32(abs(C[2]))
        planar_sq = dx * dx + dy * dy
        dist3 = np.sqrt(planar_sq + cz * cz)
        theta = np.arccos(np.clip(cz / np.maximum(dist3, 1e-9), 0.0, 1.0))
        mag = np.asarray(inplane_shift_mm(theta, slab), dtype=np.float32)
        vn = np.maximum(np.sqrt(planar_sq), np.float32(1e-9))
        scale_f = mag / vn
        true_mm = np.empty_like(app, dtype=np.float32)
        true_mm[:, 0] = app[:, 0] + dx * scale_f
        true_mm[:, 1] = app[:, 1] + dy * scale_f
    else:
        true_mm = app

    tex_x = (true_mm[:, 0] * np.float32(texture_px_per_mm)).astype(np.float32)
    tex_y = (true_mm[:, 1] * np.float32(texture_px_per_mm)).astype(np.float32)
    inside = (
        (tex_x >= 0)
        & (tex_x <= tex.shape[1] - 1)
        & (tex_y >= 0)
        & (tex_y <= tex.shape[0] - 1)
        & (card_h_w > 0)
    )

    map_x = np.where(inside, tex_x, np.float32(-1)).reshape(h_img, w_img)
    map_y = np.where(inside, tex_y, np.float32(-1)).reshape(h_img, w_img)
    sampled = cv2.remap(
        tex, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    mask = inside.reshape(h_img, w_img)
    out[mask] = sampled[mask]

    # Vignette and sensor noise in a single float32 pass.
    # A physical card has thickness, and obliquely lit it casts a shadow of its
    # own edge onto whatever it sits on. The shadow falls only on the side away
    # from the light, so unlike almost everything else in this renderer it is
    # NOT common-mode: it biases one border and not its opposite.
    if card_thickness_mm > 1e-9 and light_elevation_deg < 89.0:
        reach = card_thickness_mm / math.tan(math.radians(light_elevation_deg))
        ax = math.radians(light_azimuth_deg)
        # Shadow extends away from the light, in card-plane mm.
        off = np.array([-math.cos(ax), -math.sin(ax)]) * reach
        shifted = np.empty_like(app)
        shifted[:, 0] = app[:, 0] - np.float32(off[0])
        shifted[:, 1] = app[:, 1] - np.float32(off[1])
        on_shifted = (
            (shifted[:, 0] >= 0) & (shifted[:, 0] <= STANDARD_CARD_W_MM)
            & (shifted[:, 1] >= 0) & (shifted[:, 1] <= STANDARD_CARD_H_MM)
        )
        shadow = on_shifted & (~inside)
        if shadow.any():
            m = shadow.reshape(h_img, w_img)
            out[m] = (out[m].astype(np.float32) * shadow_darkness).astype(np.uint8)

    work = out.astype(np.float32)
    if vignette > 0:
        rx = ((xs - w_img / 2.0) / (w_img / 2.0)) ** 2
        ry = ((ys - h_img / 2.0) / (h_img / 2.0)) ** 2
        r = np.sqrt(rx[None, :] + ry[:, None])
        gain = (1.0 - vignette * np.clip(r, 0, 1.5) ** 2).astype(np.float32)
        work *= gain[..., None]
    if noise_sigma > 0:
        work += rng.standard_normal(work.shape, dtype=np.float32) * np.float32(
            noise_sigma
        )
    out = np.clip(work, 0, 255).astype(np.uint8)

    gt = GroundTruth(
        left_mm=left_mm,
        top_mm=top_mm,
        right_mm=right_mm,
        bottom_mm=bottom_mm,
        tilt_deg=tilt_deg,
        azimuth_deg=azimuth_deg,
        slab_name=slab.name,
    )
    return out, gt, focal_px


@dataclass(frozen=True)
class CaseCard:
    """One card placed in a synthetic display-case scene."""

    left_mm: float
    top_mm: float
    right_mm: float
    bottom_mm: float
    rotation_deg: float = 0.0
    border_bgr: tuple[int, int, int] = (60, 200, 240)
    art_bgr: tuple[int, int, int] = (120, 70, 45)

    @property
    def h_ratio(self) -> float:
        return 100.0 * max(self.left_mm, self.right_mm) / (self.left_mm + self.right_mm)

    @property
    def v_ratio(self) -> float:
        return 100.0 * max(self.top_mm, self.bottom_mm) / (self.top_mm + self.bottom_mm)

    @property
    def worst_ratio(self) -> float:
        return max(self.h_ratio, self.v_ratio)


def render_case_scene(
    cards: list[CaseCard],
    columns: int = 4,
    px_per_mm: float = 9.0,
    gap_mm: float = 14.0,
    background_bgr: tuple[int, int, int] = (38, 40, 46),
    glare_spots: int = 0,
    blur_sigma: float = 0.0,
    noise_sigma: float = 2.0,
    seed: int = 0,
) -> tuple[np.ndarray, list[CaseCard]]:
    """Composite several cards into one wide frame, as a display case shot.

    Cards are laid out fronto-parallel with small individual rotations. This is
    a detection-and-throughput test, not an optics test -- the single-card
    renderer stays the authority on perspective and refraction.
    """
    rng = np.random.default_rng(seed)
    cw = int(round(STANDARD_CARD_W_MM * px_per_mm))
    ch = int(round(STANDARD_CARD_H_MM * px_per_mm))
    gap = int(round(gap_mm * px_per_mm))
    rows = int(math.ceil(len(cards) / columns))

    W = gap + columns * (cw + gap)
    H = gap + rows * (ch + gap)
    canvas = np.empty((H, W, 3), dtype=np.uint8)
    canvas[:, :] = background_bgr

    for i, spec in enumerate(cards):
        tex = make_card_texture(
            spec.left_mm,
            spec.top_mm,
            spec.right_mm,
            spec.bottom_mm,
            px_per_mm=px_per_mm,
            border_bgr=spec.border_bgr,
            art_bgr=spec.art_bgr,
            seed=seed + i,
        )
        pad = int(max(cw, ch) * 0.20)
        padded = np.empty((ch + 2 * pad, cw + 2 * pad, 3), dtype=np.uint8)
        padded[:, :] = background_bgr
        padded[pad : pad + ch, pad : pad + cw] = tex
        if abs(spec.rotation_deg) > 1e-6:
            centre = (padded.shape[1] / 2.0, padded.shape[0] / 2.0)
            M = cv2.getRotationMatrix2D(centre, spec.rotation_deg, 1.0)
            padded = cv2.warpAffine(
                padded,
                M,
                (padded.shape[1], padded.shape[0]),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=background_bgr,
            )

        # Composite through a mask of the card itself. Pasting the whole padded
        # box would overwrite a neighbouring card whenever the padding is wider
        # than the gap, silently clipping its border and corrupting the ground
        # truth the scan is being checked against.
        card_mask = np.zeros(padded.shape[:2], dtype=np.uint8)
        card_mask[pad : pad + ch, pad : pad + cw] = 255
        if abs(spec.rotation_deg) > 1e-6:
            card_mask = cv2.warpAffine(
                card_mask,
                M,
                (card_mask.shape[1], card_mask.shape[0]),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

        r, c = divmod(i, columns)
        cx = gap + c * (cw + gap) + cw // 2
        cy = gap + r * (ch + gap) + ch // 2
        y0 = cy - padded.shape[0] // 2
        x0 = cx - padded.shape[1] // 2
        y1, x1 = y0 + padded.shape[0], x0 + padded.shape[1]
        sy0, sx0 = max(0, y0), max(0, x0)
        sy1, sx1 = min(H, y1), min(W, x1)
        sub = padded[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0]
        sub_mask = card_mask[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] > 0
        canvas[sy0:sy1, sx0:sx1][sub_mask] = sub[sub_mask]

    work = canvas.astype(np.float32)
    for _ in range(glare_spots):
        gx = int(rng.integers(0, W))
        gy = int(rng.integers(0, H))
        rad = int(rng.integers(int(cw * 0.25), int(cw * 0.75)))
        overlay = np.zeros((H, W), dtype=np.float32)
        cv2.circle(overlay, (gx, gy), rad, 1.0, -1)
        overlay = cv2.GaussianBlur(overlay, (0, 0), rad * 0.35)
        work += (overlay[..., None] * 235.0)

    if blur_sigma > 0:
        work = cv2.GaussianBlur(work, (0, 0), blur_sigma)
    if noise_sigma > 0:
        work += rng.standard_normal(work.shape, dtype=np.float32) * np.float32(noise_sigma)
    return np.clip(work, 0, 255).astype(np.uint8), cards
