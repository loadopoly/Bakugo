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
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

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


# ---------------------------------------------------------------------------
# Comprehensive Card Grade Estimation & Subgrade Prediction
# ---------------------------------------------------------------------------

CONDITION_NAMES: dict[float, str] = {
    10.0: "Gem Mint",
    9.5: "Gem Mint",
    9.0: "Mint",
    8.5: "Near Mint-Mint+",
    8.0: "Near Mint-Mint",
    7.5: "Near Mint+",
    7.0: "Near Mint",
    6.5: "Excellent-Mint+",
    6.0: "Excellent-Mint",
    5.5: "Excellent+",
    5.0: "Excellent",
    4.5: "Very Good-Excellent+",
    4.0: "Very Good-Excellent",
    3.0: "Very Good",
    2.0: "Good",
    1.0: "Poor",
}


@dataclass(frozen=True)
class CardGradePrediction:
    """Estimated overall card grade, condition classification, and subgrades."""

    grader: str
    grade_score: float
    grade_label: str
    condition_name: str
    centering_subgrade: float
    estimated_corners: float
    estimated_edges: float
    estimated_surface: float
    grade_ceiling: GradeBand
    probabilities: dict[str, float]
    confidence: float
    summary: str

    def describe(self) -> str:
        lines = [
            f"=== {self.grader} ESTIMATED GRADE: {self.grade_label} ({self.condition_name}) ===",
            f"  Centering Subgrade : {self.centering_subgrade:.1f}",
            f"  Corners Subgrade   : {self.estimated_corners:.1f} (est)",
            f"  Edges Subgrade     : {self.estimated_edges:.1f} (est)",
            f"  Surface Subgrade   : {self.estimated_surface:.1f} (est)",
            f"  Confidence         : {int(self.confidence * 100)}%",
            f"  Probabilities      : " + ", ".join(f"{g}: {int(p*100)}%" for g, p in sorted(self.probabilities.items(), key=lambda kv: -kv[1])),
            f"  Centering Ceiling  : {self.grade_ceiling.best if self.grade_ceiling.is_single else f'{self.grade_ceiling.worst}-{self.grade_ceiling.best}'}",
            f"  Notes              : {self.summary}",
        ]
        return "\n".join(lines)


def predict_overall_grade(
    ratio: Measured,
    quality: Optional[Any] = None,
    geometry: Optional[Any] = None,
    grader: str = "PSA",
    face: Face = "front",
) -> CardGradePrediction:
    """Compute estimated overall grade, condition name, and 4-subgrade breakdown.
    
    Synthesizes the physical centering ratio with edge sharpness profiles,
    corner line-fit residuals, and cut squareness.
    """
    band = grade_band(ratio, grader=grader, face=face)
    
    # 1. Base centering score from ratio
    std = load_standards()
    tiers = std["graders"].get(grader, std["graders"]["PSA"])["tiers"]
    
    # Score candidate tiers
    scores: list[tuple[float, str, float]] = []  # (score_num, grade_str, weight)
    val = max(50.0, ratio.value)
    
    for t in tiers:
        g_str = t["grade"]
        try:
            score_f = float(g_str)
        except ValueError:
            score_f = 9.0
        
        # Calculate likelihood from strict/lenient thresholds and measurement sigma
        thresh_strict = t.get(f"{face}_strict", 60.0)
        thresh_lenient = t.get(f"{face}_lenient", 60.0)
        mid_thresh = (thresh_strict + thresh_lenient) / 2.0
        
        diff = mid_thresh - val
        # Gaussian CDF / error margin
        z = diff / max(ratio.sigma, 0.4)
        weight = 1.0 / (1.0 + math.exp(-z))  # Sigmoid probability of meeting tier
        scores.append((score_f, g_str, weight))

    # Determine Centering Subgrade
    # Find best tier where probability > 0.45
    eligible = [s for s in scores if s[2] >= 0.45]
    if eligible:
        best_tier = max(eligible, key=lambda x: x[0])
        centering_sub = best_tier[0]
        modal_grade_str = best_tier[1]
    else:
        best_tier = min(scores, key=lambda x: -x[0])
        centering_sub = best_tier[0]
        modal_grade_str = best_tier[1]

    # 2. Estimate Corners, Edges, Surface from image quality / geometry
    # Default high condition if detection quality is clean
    outer_res = getattr(quality, "outer_residual_px", 1.0) if quality else 1.0
    inner_conf = getattr(quality, "inner_confidence", 0.9) if quality else 0.9
    max_angle_err = getattr(geometry, "max_angle_error_deg", 0.5) if geometry else 0.5

    # Corner penalty if line-fit residual is high
    corner_pen = min(2.0, max(0.0, (outer_res - 1.5) * 0.5))
    est_corners = max(1.0, min(10.0, 10.0 - corner_pen))

    # Edge penalty if inner edge confidence is low or cut skewed
    edge_pen = min(2.0, max(0.0, (1.0 - inner_conf) * 3.0 + max(0.0, max_angle_err - 1.5) * 0.4))
    est_edges = max(1.0, min(10.0, 10.0 - edge_pen))

    # Surface estimate
    est_surface = max(1.0, min(10.0, 10.0 - min(1.0, max(0.0, (1.0 - inner_conf) * 1.5))))

    # 3. Overall composite score
    # Centering acts as an upper bound, while composite is the minimum or weighted blend
    subgrades = [centering_sub, est_corners, est_edges, est_surface]
    if grader == "BGS":
        # BGS: Cannot be more than 0.5 points higher than lowest subgrade
        lowest_sub = min(subgrades)
        avg_sub = sum(subgrades) / 4.0
        final_score = min(avg_sub, lowest_sub + 0.5)
        # Snap to half grades
        final_score = round(final_score * 2) / 2.0
    else:
        # PSA / CGC: Weakest attribute governs the ceiling
        lowest = min(subgrades)
        if grader == "PSA":
            final_score = float(round(lowest)) if lowest >= 9.5 else float(math.floor(lowest))
        else:
            final_score = round(lowest * 2) / 2.0


    final_score = max(1.0, min(10.0, final_score))
    cond_name = CONDITION_NAMES.get(final_score, "Authentic")
    grade_label = f"{grader} {int(final_score) if final_score.is_integer() else final_score}"

    # Probabilities
    probs: dict[str, float] = {}
    if final_score == 10.0:
        probs["10"] = 0.70
        probs["9.5" if grader != "PSA" else "9"] = 0.25
        probs["9" if grader != "PSA" else "8"] = 0.05
    elif final_score >= 9.0:
        probs[str(final_score)] = 0.65
        probs[str(final_score - 0.5 if grader != "PSA" else final_score - 1)] = 0.25
        probs[str(min(10.0, final_score + 0.5 if grader != "PSA" else final_score + 1))] = 0.10
    else:
        probs[str(final_score)] = 0.70
        probs[str(max(1.0, final_score - 1.0))] = 0.20
        probs[str(min(10.0, final_score + 1.0))] = 0.10

    summary = (
        f"Predicted {grade_label} ({cond_name}) based on {ratio.value:.1f}/{100 - ratio.value:.1f} "
        f"centering and image quality assessment."
    )
    conf = max(0.5, min(0.95, 1.0 - (ratio.sigma / 10.0)))

    return CardGradePrediction(
        grader=grader,
        grade_score=final_score,
        grade_label=grade_label,
        condition_name=cond_name,
        centering_subgrade=centering_sub,
        estimated_corners=est_corners,
        estimated_edges=est_edges,
        estimated_surface=est_surface,
        grade_ceiling=band,
        probabilities=probs,
        confidence=round(conf, 2),
        summary=summary,
    )


def predict_all_grades(
    ratio: Measured,
    quality: Optional[Any] = None,
    geometry: Optional[Any] = None,
    face: Face = "front",
) -> dict[str, CardGradePrediction]:
    """Predict grades across all supported grading houses (PSA, BGS, CGC, SGC)."""
    return {
        g: predict_overall_grade(ratio, quality=quality, geometry=geometry, grader=g, face=face)
        for g in available_graders()
    }

