"""Visual output.

The overlay exists so a user can disagree with the tool. A number alone is
unfalsifiable by eye; a number with the detected border drawn on top of the
actual card lets anyone see in one second whether the detector locked onto the
right edge. If it did not, the measurement is wrong and the picture says so.

Drawn on the rectified card:
  solid  -- where the printed frame was actually detected
  dashed -- where that frame would sit if the card were perfectly centred
The gap between them is the miscentering, at true scale.
"""

from __future__ import annotations

import cv2
import numpy as np

from .grading import GradeBand
from .types import STANDARD_CARD_H_MM, STANDARD_CARD_W_MM, CenteringResult

_DETECTED = (70, 220, 90)
_IDEAL = (200, 200, 200)
_TEXT = (245, 245, 245)
_PANEL = (28, 28, 32)
_WARN = (60, 170, 250)


def _dashed_rect(
    img: np.ndarray, p0: tuple[int, int], p1: tuple[int, int], colour, thickness=1, dash=9
) -> None:
    x0, y0 = p0
    x1, y1 = p1
    for x in range(x0, x1, dash * 2):
        cv2.line(img, (x, y0), (min(x + dash, x1), y0), colour, thickness)
        cv2.line(img, (x, y1), (min(x + dash, x1), y1), colour, thickness)
    for y in range(y0, y1, dash * 2):
        cv2.line(img, (x0, y), (x0, min(y + dash, y1)), colour, thickness)
        cv2.line(img, (x1, y), (x1, min(y + dash, y1)), colour, thickness)


def annotate(
    result: CenteringResult,
    bands: dict[str, GradeBand] | None = None,
    panel_width: int = 420,
) -> np.ndarray:
    """Build the annotated result image."""
    if result.rectified is None:
        raise ValueError("result has no rectified image; measure with keep_rectified=True")

    card = result.rectified.copy()
    ppm = result.px_per_mm
    h, w = card.shape[:2]

    l, t, r, b = result.inner_rect_mm
    det = (int(round(l * ppm)), int(round(t * ppm)), int(round(r * ppm)), int(round(b * ppm)))

    # Ideal frame: same printed-area size, but centred within the cut.
    inner_w = r - l
    inner_h = b - t
    il = (STANDARD_CARD_W_MM - inner_w) / 2.0
    it = (STANDARD_CARD_H_MM - inner_h) / 2.0
    ideal = (
        int(round(il * ppm)),
        int(round(it * ppm)),
        int(round((il + inner_w) * ppm)),
        int(round((it + inner_h) * ppm)),
    )

    _dashed_rect(card, (ideal[0], ideal[1]), (ideal[2], ideal[3]), (10, 10, 10), 3, dash=9)
    _dashed_rect(card, (ideal[0], ideal[1]), (ideal[2], ideal[3]), _IDEAL, 2, dash=9)
    cv2.rectangle(card, (det[0], det[1]), (det[2], det[3]), (10, 10, 10), 4)
    cv2.rectangle(card, (det[0], det[1]), (det[2], det[3]), _DETECTED, 2)

    # Border width callouts, drawn in the border itself.
    def label(text: str, org: tuple[int, int]) -> None:
        cv2.putText(card, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(card, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.42, _TEXT, 1, cv2.LINE_AA)

    # Labels sit just inside the detected frame. Putting them in the border
    # itself clips as soon as the border is narrow, which is exactly the case
    # anyone using this tool cares most about.
    hl = result.horizontal
    vl = result.vertical
    label(f"{hl.low_mm.value:.2f}", (det[0] + 6, h // 2))
    label(f"{hl.high_mm.value:.2f}", (max(det[0] + 6, det[2] - 52), h // 2))
    label(f"{vl.low_mm.value:.2f}", (w // 2 - 18, det[1] + 20))
    label(f"{vl.high_mm.value:.2f}", (w // 2 - 18, max(det[1] + 40, det[3] - 8)))

    panel = np.zeros((h, panel_width, 3), dtype=np.uint8)
    panel[:, :] = _PANEL

    y = 34

    def line(text: str, colour=_TEXT, scale=0.5, gap=24, indent=16) -> None:
        nonlocal y
        if y > h - 12:
            return
        cv2.putText(
            panel, text, (indent, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA
        )
        y += gap

    line("CENTERING", _TEXT, 0.62, 32)

    for pair in (hl, vl):
        rp = pair.ratio_pct
        lo, hi = rp.interval()
        tag = "  <-- worst" if pair is result.worst_axis else ""
        line(f"{pair.axis}: {rp.value:.1f}/{100 - rp.value:.1f}{tag}", _DETECTED, 0.5, 20)
        line(f"   95% CI {lo:.1f}-{hi:.1f}  (wider {pair.skew_toward})", _TEXT, 0.42, 26)

    line("", gap=4)
    line(
        f"borders mm  L{hl.low_mm.value:.2f} R{hl.high_mm.value:.2f} "
        f"T{vl.low_mm.value:.2f} B{vl.high_mm.value:.2f}",
        _TEXT,
        0.4,
        24,
    )
    line(f"holder: {result.slab.name}   scale: {ppm:.1f} px/mm", _TEXT, 0.4, 22)
    if result.quality.refraction_applied:
        line(
            f"refraction corrected (max {result.quality.max_refraction_shift_mm:.3f} mm)",
            _TEXT,
            0.4,
            26,
        )

    if bands:
        line("", gap=6)
        line("GRADE CEILING (centering only)", _TEXT, 0.5, 26)
        for name, band in bands.items():
            txt = band.best if band.is_single else f"{band.worst}-{band.best}"
            line(f"   {name:<5} {txt}", _DETECTED, 0.48, 22)

    if result.quality.warnings:
        line("", gap=6)
        line("WARNINGS", _WARN, 0.5, 24)
        for wmsg in result.quality.warnings:
            words = wmsg.split()
            cur = ""
            for word in words:
                if len(cur) + len(word) + 1 > 46:
                    line(f"  {cur}", _WARN, 0.38, 17)
                    cur = word
                else:
                    cur = f"{cur} {word}".strip()
            if cur:
                line(f"  {cur}", _WARN, 0.38, 20)

    return np.hstack([card, panel])


def save_annotated(path: str, result: CenteringResult, bands=None) -> str:
    img = annotate(result, bands)
    cv2.imwrite(path, img)
    return path
