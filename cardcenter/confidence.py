"""
confidence.py -- auto-accept gate keyed to the interstitial-space / shot-noise
information floor computed in information.py.

The threshold is not a made-up similarity gap; it is the Cramer-Rao floor this
package already computes.

BASIS (all from cardcenter/information.py):
  - "Interstitial pixel space" is the sensor fill-factor dead area; it is folded
    into quantum_efficiency, and the binding limit is photon shot noise.
  - cramer_rao_ratio_pp(c, border_a_mm, border_b_mm, px_per_mm) -> the CR lower
    bound (sigma) on the centering ratio, in percentage points: the best any
    estimator can do on THIS frame.
  - shot_noise_consistency(c, background_level, sensor) -> (ratio, verdict):
    ratio ~1 means the frame is photon-limited (reaching the floor); >~1.6 means
    read noise / JPEG / demosaic dominate, so the CR floor is optimistic.

RULE. Auto-accept ("fire and forget") a grade/centering call only when the
measurement sits k CR-sigmas clear of the decision boundary AND the frame is
actually photon-limited. A wide margin computed on a compression-limited frame
is a lie -- the true sigma is larger than the CR floor there.

Multi-frame fusion enters exactly as information.py allows: it raises the
effective independent-row count (which tightens sigma), it never multiplies
Fisher information. `effective_rows` carries that; the gate does not invent it.

SCOPE. The CR math bounds the CENTERING ratio, so this is the rigorous gate for
the grade decision. For card *identification* there is no ratio-boundary; the
same `shot_noise_consistency` check still applies (don't trust an ID off a
non-photon-limited frame), and the printing is disambiguated by resolution +
candidate separation on top of that.

The one knob that still wants calibration from a labeled eval is `k_sigma`;
`max_shot_ratio` is reused directly from information.py's own shot-noise-limited
band, not guessed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Decision(str, Enum):
    ACCEPT = "accept"   # commit the grade silently
    REVIEW = "review"   # too close to the boundary, or frame not good enough
    REJECT = "reject"   # no usable information floor at all


@dataclass(frozen=True)
class GateConfig:
    k_sigma: float = 3.0          # decision must clear the boundary by this many CR-sigmas
    max_shot_ratio: float = 1.6   # information.py's own shot-noise-limited cutoff
    max_sigma_pp: float = 6.0     # above this the CR floor carries no decision power
    min_effective_rows: float = 2.0   # need at least this much fused evidence to ACCEPT


@dataclass(frozen=True)
class Gated:
    decision: Decision
    confidence: float     # margin/k_sigma squashed to 0..1 (ranking, not probability)
    margin_sigma: float   # |ratio - boundary| / sigma_cr -- the real number
    sigma_cr_pp: float    # the CR floor used
    shot_ratio: float     # how far the frame is from photon-limited
    reason: str


def gate(
    measured_ratio_pp: float,
    boundary_pp: float,
    sigma_cr_pp: float,
    shot_ratio: float,
    effective_rows: float,
    cfg: GateConfig = GateConfig(),
) -> Gated:
    """Decide accept / review / reject from the interstitial/shot-noise floor.

    measured_ratio_pp : fused worst-axis centering ratio (percentage points)
    boundary_pp       : grade boundary being decided against (e.g. 55.0)
    sigma_cr_pp       : cramer_rao_ratio_pp(...) -- the CR floor, in pp
    shot_ratio        : shot_noise_consistency(...)[0]
    effective_rows    : fused independent-row count (QUIPU boost scales THIS)
    """
    if not (sigma_cr_pp > 0.0) or sigma_cr_pp == float("inf") or sigma_cr_pp > cfg.max_sigma_pp:
        return Gated(Decision.REJECT, 0.0, 0.0, sigma_cr_pp, shot_ratio,
                     f"no usable information floor (sigma_cr={sigma_cr_pp:.2f}pp)")

    if shot_ratio > cfg.max_shot_ratio:
        return Gated(Decision.REVIEW, 0.0, 0.0, sigma_cr_pp, shot_ratio,
                     f"frame not photon-limited (noise {shot_ratio:.1f}x floor); "
                     "CR bound is optimistic here -- do not auto-accept")

    margin = abs(measured_ratio_pp - boundary_pp) / sigma_cr_pp
    conf = _squash(margin / cfg.k_sigma)

    if margin >= cfg.k_sigma and effective_rows >= cfg.min_effective_rows:
        return Gated(Decision.ACCEPT, conf, margin, sigma_cr_pp, shot_ratio,
                     f"{margin:.1f} sigma clear of {boundary_pp:.0f} at the shot-noise floor")

    if effective_rows < cfg.min_effective_rows:
        return Gated(Decision.REVIEW, conf, margin, sigma_cr_pp, shot_ratio,
                     "need another frame to fuse (too few effective rows)")

    return Gated(Decision.REVIEW, conf, margin, sigma_cr_pp, shot_ratio,
                 f"only {margin:.1f} sigma from the boundary -- too close to call")


def gate_from_channel(
    channel,                 # information.ChannelConditions
    sensor,                  # information.SensorModel
    background_level: float,
    measured_ratio_pp: float,
    boundary_pp: float,
    border_a_mm: float,
    border_b_mm: float,
    px_per_mm: float,
    effective_rows: float,
    cfg: GateConfig = GateConfig(),
) -> Gated:
    """Pull the interstitial/shot-noise floor straight from information.py and gate on it."""
    from .information import cramer_rao_ratio_pp, shot_noise_consistency

    sigma_cr_pp = cramer_rao_ratio_pp(channel, border_a_mm, border_b_mm, px_per_mm)
    shot_ratio, _verdict = shot_noise_consistency(channel, background_level, sensor)
    return gate(measured_ratio_pp, boundary_pp, sigma_cr_pp, shot_ratio, effective_rows, cfg)


def _squash(x: float) -> float:
    """0 at the boundary, ->1 well past k_sigma. Monotone, for the calibration curve."""
    return 0.0 if x <= 0.0 else round(1.0 - 2.0 ** (-x), 4)
