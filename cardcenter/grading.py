"""Turn a measured centering ratio into a grade *band*.

Two independent things stop us from naming a single grade, and the tool keeps
them separate because the user's response to each is different:

  MEASUREMENT uncertainty -- the confidence interval on our own ratio. Fixed by
  better capture: more light, less tilt, a tripod, a higher-resolution sensor.

  STANDARDS ambiguity -- reputable sources disagree about where the thresholds
  actually sit, and graders reserve explicit discretion. No amount of better
  photography fixes this. It is irreducible from outside the grading room.

Reporting one number would hide both. Reporting a band without saying which one
is binding would leave the user unable to act. So we report the band and name
the dominant cause.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from .types import Measured

Face = Literal["front", "back"]
_DATA = Path(__file__).parent / "data" / "standards.json"


@lru_cache(maxsize=1)
def load_standards() -> dict:
    with open(_DATA, "r", encoding="utf-8") as fh:
        return json.load(fh)


def available_graders() -> list[str]:
    return list(load_standards()["graders"].keys())


@dataclass(frozen=True)
class GradeBand:
    grader: str
    face: Face
    best: str
    worst: str
    ratio: Measured
    measurement_span: int
    standards_span: int
    limited_by: str
    grader_confidence: str
    notes: str

    @property
    def is_single(self) -> bool:
        return self.best == self.worst

    def describe(self) -> str:
        lo, hi = self.ratio.interval()
        head = (
            f"{self.grader} {self.face} centering ceiling: "
            + (self.best if self.is_single else f"{self.worst}-{self.best}")
        )
        body = (
            f"  measured {self.ratio.value:.1f}/{100 - self.ratio.value:.1f} "
            f"(95% CI {lo:.1f}-{hi:.1f})\n"
            f"  band limited by: {self.limited_by}\n"
            f"  table confidence: {self.grader_confidence}"
        )
        return head + "\n" + body


def _threshold_key(face: Face, variant: str) -> str:
    return f"{face}_{variant}"


def _tier_index_for(tiers: list[dict], ratio: float, face: Face, variant: str) -> int:
    """Index of the best tier whose threshold admits ``ratio``. Higher index = worse."""
    key = _threshold_key(face, variant)
    for i, tier in enumerate(tiers):
        if ratio <= tier[key] + 1e-9:
            return i
    return len(tiers) - 1


def grade_band(
    ratio: Measured,
    grader: str = "PSA",
    face: Face = "front",
    k_sigma: float = 1.96,
) -> GradeBand:
    """Map a worst-axis ratio to a band of plausible centering grades."""
    std = load_standards()
    graders = std["graders"]
    if grader not in graders:
        raise KeyError(
            f"unknown grader '{grader}'. Available: {', '.join(graders)}"
        )
    g = graders[grader]
    tiers = g["tiers"]

    lo, hi = ratio.interval(k_sigma)
    lo = max(50.0, lo)
    hi = max(50.0, hi)
    centre = max(50.0, ratio.value)

    # Best case: low end of our interval, judged by the most forgiving table.
    best_idx = _tier_index_for(tiers, lo, face, "lenient")
    # Worst case: high end of our interval, judged by the strictest table.
    worst_idx = _tier_index_for(tiers, hi, face, "strict")

    # Attribution. Hold the table fixed to isolate measurement span; hold the
    # ratio fixed to isolate standards span.
    meas_spans = [
        _tier_index_for(tiers, hi, face, v) - _tier_index_for(tiers, lo, face, v)
        for v in ("strict", "lenient")
    ]
    measurement_span = max(meas_spans)
    standards_span = _tier_index_for(tiers, centre, face, "strict") - _tier_index_for(
        tiers, centre, face, "lenient"
    )

    if measurement_span == 0 and standards_span == 0:
        limited_by = "neither; the measurement and the published tables agree"
    elif measurement_span > standards_span:
        limited_by = (
            "measurement uncertainty -- a steadier, better-lit, less-tilted "
            "capture would narrow this"
        )
    elif standards_span > measurement_span:
        limited_by = (
            "standards ambiguity -- sources disagree on this threshold, and "
            "better photography will not resolve it"
        )
    else:
        limited_by = "measurement uncertainty and standards ambiguity equally"

    return GradeBand(
        grader=grader,
        face=face,
        best=tiers[best_idx]["grade"],
        worst=tiers[worst_idx]["grade"],
        ratio=ratio,
        measurement_span=int(measurement_span),
        standards_span=int(standards_span),
        limited_by=limited_by,
        grader_confidence=g.get("confidence", "unknown"),
        notes=g.get("notes", ""),
    )


def all_grade_bands(
    ratio: Measured, face: Face = "front", k_sigma: float = 1.96
) -> dict[str, GradeBand]:
    return {
        name: grade_band(ratio, name, face, k_sigma) for name in available_graders()
    }


def caveat_text(grader: str) -> str:
    g = load_standards()["graders"].get(grader, {})
    lines = [g.get("notes", "")]
    if not g.get("subgrade_published", False):
        lines.append(
            "This grader does not publish a centering sub-grade, so centering "
            "only sets a ceiling on the overall grade. Corners, edges and "
            "surface can and often will land it lower."
        )
    if g.get("confidence") == "low":
        lines.append(
            "Threshold sourcing for this grader is weak. Treat the band as "
            "indicative and verify against current published standards."
        )
    return "\n".join(x for x in lines if x)
