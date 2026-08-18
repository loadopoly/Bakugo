"""Pixel loss as probabilistic truth: the information-theoretic floor.

WHAT THE GARD-SHARD MECHANISM ACTUALLY IS
------------------------------------------
`gard_shard_model.py` is canonical UTF-8 JSON -> zlib -> AES-256-CBC ->
HMAC-SHA256, partitioned into shards. It is LOSSLESS DATA compression with
authenticated encryption. It is competent at that job and it has no relationship
to pixel loss: zlib does not discard information, and decompression returns the
input exactly. There is nothing in it to translate into a probability, because
nothing is lost.

But the instinct behind the request is right, and there is a rigorous version of
it. The imaging chain IS a lossy channel:

    true edge position  ->  optical blur  ->  pixel sampling  ->  noise  ->  data

and information theory says exactly how much survives. The quantity is Fisher
information, and it gives a hard lower bound -- the Cramer-Rao bound -- on the
variance of ANY unbiased estimator of the edge position. Not any estimator we
have written: any estimator that could be written.

THE DERIVATION
--------------
Model a card border as a step of contrast C convolved with a Gaussian PSF of
width sigma_p, sampled on a pixel grid of pitch Delta, with i.i.d. Gaussian
sensor noise sigma_n. The expected intensity along a row crossing the edge at
position theta is

    mu(x) = B + C * Phi((x - theta) / sigma_p)

so the sensitivity of each pixel to the edge position is

    d mu / d theta = -(C / sigma_p) * phi((x - theta) / sigma_p)

Fisher information for one row is the sum of squared sensitivities over noise
variance. In the continuum limit (pixels finer than the blur),

    I_row = (C^2 / sigma_n^2) * (1/Delta) * INT phi(u)^2 du / sigma_p
          = C^2 / (sigma_n^2 * Delta * 2 * sqrt(pi) * sigma_p)

using INT phi(u)^2 du = 1 / (2 sqrt(pi)). A card edge spans N rows, and if their
noise is independent the information adds, so I = N * I_row and

    sigma_theta  >=  (sigma_n / C) * sqrt( 2 sqrt(pi) sigma_p Delta / N )

WHY THIS IS WORTH HAVING
-------------------------
Three concrete uses, in order of importance.

  1. IT FALSIFIES ERROR BARS. If a detector reports an uncertainty below this
     bound, the uncertainty is impossible and something is wrong -- a bug, an
     unmodelled correlation, or double-counted rows. This is the strongest
     available check on the honesty of a measurement, because it does not depend
     on knowing the truth.

  2. IT SEPARATES "BAD ALGORITHM" FROM "BAD PHOTOGRAPH". If the achieved
     uncertainty sits near the bound, better code will not help and only a
     better capture will. If it sits far above, the capture is fine and the
     detector is leaving information on the floor.

  3. IT RANKS CAPTURE FIXES BY DERIVATION RATHER THAN FOLKLORE. Since
     sigma_theta scales as (sigma_n / C) * sqrt(sigma_p * Delta / N), doubling
     contrast halves the error, halving blur only buys sqrt(2), and using four
     times as many rows halves it. That tells a user what to actually change.

The bound assumes independent noise between rows, which glare and JPEG blocking
violate; correlated noise makes the true bound WORSE, so this remains a valid
floor rather than an optimistic one.

TEMPORAL-SPATIAL INTERACTIONS (from QUIPU)
------------------------------------------
QUIPU's ``temporal_spatiality`` layer coordinates cross-sense activity as a
rhythm: coherence of the joint vector, a relational gradient that washes
acceleration back down, a 1-D Weyl coordinate at the 7-D torus centroid, and a
boost in ``[0.5, 1.5]``. Those interactions belong on this channel because the
imaging chain is already a lossy, multi-sample process -- rows along an edge,
and frames of the same card -- whose independence is exactly what the CRB
assumes and what glare / JPEG / handshake violate.

Mapped honestly, not theatrically:

  * coherence / Weyl diagnose whether the channel observables (SNR, blur,
    coverage, shot-ratio, efficiency, multi-frame consistency) are aligned.
  * the synaptic wash (``boost < 1``) discounts independent-sample credit so
    correlated or disagreeing observations cannot claim ``1/sqrt(N)``.
  * ``boost > 1`` is reported as a cadence / fusion-confidence signal. It is
    never allowed to invent Fisher information or beat the single-shot CRB.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Mapping, Optional, Sequence

import cv2
import numpy as np

# INT phi(u)^2 du over the real line, phi the standard normal density.
_PHI_SQ_INTEGRAL = 1.0 / (2.0 * math.sqrt(math.pi))


@dataclass(frozen=True)
class ChannelConditions:
    """The imaging channel, measured from the actual photograph."""

    contrast: float  # step height across the edge, in grey levels
    noise_sigma: float  # per-pixel noise, grey levels
    psf_sigma_px: float  # optical + motion blur, pixels
    pixel_pitch_px: float = 1.0  # 1.0 unless the image was downsampled
    rows: int = 1  # independent samples along the edge

    @property
    def snr(self) -> float:
        return self.contrast / max(self.noise_sigma, 1e-9)


# ---------------------------------------------------------------------------
# Temporal-spatial overlay (QUIPU temporal_spatiality, pixel-space form)
# ---------------------------------------------------------------------------
#
# The six sense slots are filled from observables this module already
# measures. Weights match QUIPU exactly so a rhythm computed here is
# numerically comparable to one computed there.
#
#   vision     -- contrast / (contrast + noise)          edge visibility
#   touch      -- 1 / (1 + shot_ratio)                   photon corroboration
#   smell      -- 1 / (1 + extra blur above 0.4 px)      decay / wash
#   body       -- rows / (rows + 200)                    coverage / mass
#   brain      -- audit efficiency, else 0               estimator honesty
#   perception -- 1 / (1 + chi2/dof - 1)                 multi-frame agreement
#
# Boost never multiplies Fisher information. It only scales the *effective*
# independent-row count used when fusing repeated observations of the same
# edge -- the same job QUIPU's period_factor does for cadence.

_SENSE_WEIGHTS: dict[str, float] = {
    "vision": 0.22,
    "touch": 0.22,
    "smell": 0.18,
    "body": 0.12,
    "brain": 0.12,
    "perception": 0.14,
}

_TORUS_DIMS = 7
_BOOST_MIN = 0.5
_BOOST_MAX = 1.5
_BOOST_NEUTRAL = 1.0
_BODY_ROW_SCALE = 200.0
_SMELL_PSF_FLOOR = 0.4


@dataclass(frozen=True)
class TemporalSpatialRhythm:
    """QUIPU rhythm scalars, computed from pixel-space channel observables."""

    coherence: float
    gradient: float
    weyl: float
    boost: float
    period_factor: float
    lr_factor: float
    signals: dict[str, float]
    effective_rows: float

    def describe(self) -> str:
        return (
            f"temporal-spatial rhythm : coherence {self.coherence:.3f}, "
            f"gradient {self.gradient:.3f}, weyl {self.weyl:.3f} rad, "
            f"boost {self.boost:.3f} (period {self.period_factor:.3f}, "
            f"lr {self.lr_factor:.3f}), effective rows {self.effective_rows:.1f}"
        )


def measure_coherence(signals: Mapping[str, float]) -> float:
    """Joint activity times (1 - normalised dispersion), in ``[0, 1]``.

    Port of QUIPU ``temporal_spatiality.measure_coherence``. High only when
    every sense is active *and* they agree; one saturated channel with the
    rest silent scores near zero.
    """
    senses = list(_SENSE_WEIGHTS.keys())
    activities = [max(0.0, min(1.0, float(signals.get(s, 0.0)))) for s in senses]
    weights = [_SENSE_WEIGHTS[s] for s in senses]
    mean_a = sum(a * w for a, w in zip(activities, weights))
    if mean_a <= 0.0:
        return 0.0
    var = sum(w * (a - mean_a) ** 2 for a, w in zip(activities, weights))
    norm_disp = min(1.0, math.sqrt(var) / max(mean_a, 1e-9))
    return max(0.0, min(1.0, mean_a * (1.0 - norm_disp)))


def relational_gradient(
    signals: Mapping[str, float], decay: Optional[float] = None
) -> float:
    """Synaptic-wash damper: (touch + decay + spread) / 3, in ``[0, 1]``.

    Port of QUIPU ``temporal_spatiality.relational_gradient``. ``decay`` is
    the smell-mass complement; when omitted it is ``1 - smell``, which is
    the honest reading of extra blur as stale / washed signal.
    """
    touch = max(0.0, min(1.0, float(signals.get("touch", 0.0))))
    if decay is None:
        decay = 1.0 - max(0.0, min(1.0, float(signals.get("smell", 0.0))))
    decay = max(0.0, min(1.0, float(decay)))
    activities = [max(0.0, min(1.0, float(signals.get(s, 0.0)))) for s in _SENSE_WEIGHTS]
    spread = (max(activities) - min(activities)) if activities else 0.0
    return max(0.0, min(1.0, (touch + decay + spread) / 3.0))


def weyl_centroid(signals: Mapping[str, float], torus_dims: int = _TORUS_DIMS) -> float:
    """Circular mean of sense angles on the first n of ``torus_dims``, in ``[0, 2π]``.

    Port of QUIPU ``temporal_spatiality.weyl_centroid``. Senses sit on evenly
    spaced torus angles; activity weights the phasor. Empty activity returns 0.
    """
    dims = max(1, int(torus_dims))
    sx = 0.0
    sy = 0.0
    for i, sense in enumerate(_SENSE_WEIGHTS):
        theta = 2.0 * math.pi * (i / dims)
        a = max(0.0, min(1.0, float(signals.get(sense, 0.0))))
        sx += a * math.cos(theta)
        sy += a * math.sin(theta)
    if sx == 0.0 and sy == 0.0:
        return 0.0
    mean_angle = math.atan2(sy, sx)
    if mean_angle < 0.0:
        mean_angle += 2.0 * math.pi
    return mean_angle


def modulate(
    signals: Mapping[str, float],
    *,
    decay: Optional[float] = None,
    potential: float = 0.0,
) -> dict[str, object]:
    """Combine coherence, wash, and Weyl into the QUIPU rhythm dict.

    ``boost = clamp(1 + (coherence - gradient) * 0.5, 0.5, 1.5)``, with the
    recursive-strengthening floor ``1 + 0.25 * potential`` when potential
    is supplied. ``period_factor = 1 / boost``, ``lr_factor = boost``.
    """
    coh = measure_coherence(signals)
    grad = relational_gradient(signals, decay=decay)
    weyl = weyl_centroid(signals)
    raw_boost = _BOOST_NEUTRAL + (coh - grad) * 0.5
    floor = _BOOST_NEUTRAL + 0.25 * max(0.0, min(1.0, float(potential)))
    raw_boost = max(raw_boost, floor)
    boost = max(_BOOST_MIN, min(_BOOST_MAX, raw_boost))
    return {
        "coherence": round(coh, 4),
        "gradient": round(grad, 4),
        "weyl": round(weyl, 4),
        "boost": round(boost, 4),
        "period_factor": round(1.0 / boost, 4),
        "lr_factor": round(boost, 4),
        "signals": {
            k: round(max(0.0, min(1.0, float(signals.get(k, 0.0)))), 4)
            for k in _SENSE_WEIGHTS
        },
    }


def channel_sense_signals(
    c: ChannelConditions,
    *,
    efficiency: Optional[float] = None,
    shot_ratio: Optional[float] = None,
    frame_chi2_dof: Optional[float] = None,
) -> dict[str, float]:
    """Project a measured imaging channel onto the six QUIPU sense slots."""
    vision = c.contrast / (c.contrast + max(c.noise_sigma, 1e-9))
    if shot_ratio is None:
        touch = min(1.0, c.snr / (c.snr + 25.0))
    else:
        touch = 1.0 / (1.0 + max(0.0, float(shot_ratio)))
    smell = 1.0 / (1.0 + max(0.0, c.psf_sigma_px - _SMELL_PSF_FLOOR))
    body = float(max(1, c.rows)) / (float(max(1, c.rows)) + _BODY_ROW_SCALE)
    brain = 0.0 if efficiency is None else max(0.0, min(1.0, float(efficiency)))
    if frame_chi2_dof is None:
        perception = 0.0
    else:
        perception = 1.0 / (1.0 + max(0.0, float(frame_chi2_dof) - 1.0))
    return {
        "vision": max(0.0, min(1.0, vision)),
        "touch": max(0.0, min(1.0, touch)),
        "smell": max(0.0, min(1.0, smell)),
        "body": max(0.0, min(1.0, body)),
        "brain": brain,
        "perception": max(0.0, min(1.0, perception)),
    }


def temporal_spatial_rhythm(
    c: ChannelConditions,
    *,
    efficiency: Optional[float] = None,
    shot_ratio: Optional[float] = None,
    frame_chi2_dof: Optional[float] = None,
    potential: float = 0.0,
) -> TemporalSpatialRhythm:
    """Rhythm for one measured channel, plus the wash-adjusted row count.

    ``effective_rows = rows * boost``. Because boost is at most 1.5 this
    never more than modestly credits extra independence, and because it
    bottoms at 0.5 a washed / disagreeing channel cannot claim the full
    ``1/sqrt(N)`` the CRB would otherwise grant.
    """
    signals = channel_sense_signals(
        c,
        efficiency=efficiency,
        shot_ratio=shot_ratio,
        frame_chi2_dof=frame_chi2_dof,
    )
    decay = 1.0 - signals["smell"]
    rhythm = modulate(signals, decay=decay, potential=potential)
    boost = float(rhythm["boost"])
    return TemporalSpatialRhythm(
        coherence=float(rhythm["coherence"]),
        gradient=float(rhythm["gradient"]),
        weyl=float(rhythm["weyl"]),
        boost=boost,
        period_factor=float(rhythm["period_factor"]),
        lr_factor=float(rhythm["lr_factor"]),
        signals=dict(rhythm["signals"]),  # type: ignore[arg-type]
        effective_rows=float(max(1, c.rows)) * boost,
    )


def apply_rhythm_to_channel(
    c: ChannelConditions, rhythm: TemporalSpatialRhythm
) -> ChannelConditions:
    """Return a copy whose ``rows`` are the wash-adjusted effective count.

    Used only for *fusion* bounds (multi-frame / multi-row credit). The
    single-shot CRB still uses the raw measured channel.
    """
    return replace(c, rows=max(1, int(round(rhythm.effective_rows))))


def fuse_channel_conditions(
    channels: Sequence[ChannelConditions],
) -> Optional[ChannelConditions]:
    """Inverse-variance blend of repeated channel measurements of one edge.

    Contrast, noise and blur are combined with weights ``rows / noise^2``.
    The fused row count is the sum -- independence is then discounted by
    :func:`temporal_spatial_rhythm` via ``effective_rows``, not here.
    """
    usable = [c for c in channels if c.noise_sigma > 0 and c.rows > 0]
    if not usable:
        return None
    weights = [float(c.rows) / (c.noise_sigma**2) for c in usable]
    wsum = sum(weights)
    if wsum <= 0:
        return None

    def _wavg(getter) -> float:
        return sum(getter(c) * w for c, w in zip(usable, weights)) / wsum

    return ChannelConditions(
        contrast=_wavg(lambda c: c.contrast),
        noise_sigma=math.sqrt(
            sum((c.noise_sigma**2) * w for c, w in zip(usable, weights)) / wsum
        ),
        psf_sigma_px=_wavg(lambda c: c.psf_sigma_px),
        pixel_pitch_px=_wavg(lambda c: c.pixel_pitch_px),
        rows=sum(max(1, c.rows) for c in usable),
    )


def fisher_information_edge(c: ChannelConditions) -> float:
    """Fisher information about edge position, in 1/px^2."""
    if c.contrast <= 0 or c.noise_sigma <= 0 or c.psf_sigma_px <= 0:
        return 0.0
    per_row = (c.contrast**2 / c.noise_sigma**2) * _PHI_SQ_INTEGRAL / (
        c.psf_sigma_px * c.pixel_pitch_px
    )
    return per_row * max(1, c.rows)


def cramer_rao_edge_px(c: ChannelConditions) -> float:
    """Lower bound on the standard error of ANY unbiased edge estimator, px."""
    info = fisher_information_edge(c)
    if info <= 0:
        return float("inf")
    return 1.0 / math.sqrt(info)


def cramer_rao_ratio_pp(
    c: ChannelConditions, border_a_mm: float, border_b_mm: float, px_per_mm: float
) -> float:
    """The same bound expressed on the centering ratio, in percentage points.

    Four edges bound the two opposing borders, and the ratio's sensitivity to
    each border width follows the same partials used elsewhere:
    d(w/(w+n))/dw = n/(w+n)^2.
    """
    sigma_px = cramer_rao_edge_px(c)
    if not math.isfinite(sigma_px):
        return float("inf")
    sigma_mm = sigma_px / max(px_per_mm, 1e-9)
    # Each border is bounded by two edges (card cut and printed frame).
    border_sigma = sigma_mm * math.sqrt(2.0)
    a, b = border_a_mm, border_b_mm
    total = a + b
    if total <= 0:
        return float("inf")
    wider, narrower = (a, b) if a >= b else (b, a)
    return 100.0 * math.hypot(narrower * border_sigma, wider * border_sigma) / total**2


def measure_channel(
    rect_bgr: np.ndarray, side: str, px_per_mm: float, depth_mm: float
) -> ChannelConditions:
    """Estimate contrast, noise and blur from the image itself.

    Contrast is the median step across the detected border; noise is the median
    absolute deviation within the flat border region; blur is recovered from the
    width of the intensity transition, since a step convolved with a Gaussian
    has a 10-90% rise of about 2.563 * sigma.
    """
    if rect_bgr.ndim == 3:
        gray = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    else:
        gray = rect_bgr.astype(np.float64)

    h, w = gray.shape
    d = depth_mm * px_per_mm
    band = max(4, int(2.5 * px_per_mm))

    if side in ("left", "right"):
        lo, hi = int(h * 0.2), int(h * 0.8)
        strip = gray[lo:hi, :]
        profile = strip.mean(axis=0)
        pos = d if side == "left" else w - 1 - d
        rows = hi - lo
        along_axis = 0
    else:
        lo, hi = int(w * 0.2), int(w * 0.8)
        strip = gray[:, lo:hi]
        profile = strip.mean(axis=1)
        pos = d if side == "top" else h - 1 - d
        rows = hi - lo
        along_axis = 1

    i = int(round(pos))
    a0, a1 = max(0, i - band), max(1, i - 2)
    b0, b1 = min(len(profile) - 1, i + 2), min(len(profile), i + band)
    if a1 <= a0 or b1 <= b0:
        return ChannelConditions(1.0, 1.0, 1.0, 1.0, 1)

    outer = float(np.median(profile[a0:a1]))
    inner = float(np.median(profile[b0:b1]))
    contrast = abs(inner - outer)

    # PER-PIXEL noise, measured on the un-averaged strip. Taking it from the
    # row-averaged profile instead was a bug: averaging N rows suppresses noise
    # by sqrt(N), and the Fisher information then multiplies by N again, so the
    # row count was counted twice and the bound came out independent of the
    # actual noise level. Caught because the floor did not move when the
    # synthetic noise was raised 25-fold.
    flat_block = (
        strip[:, a0:a1] if along_axis == 0 else strip[a0:a1, :]
    )
    if flat_block.size >= 16:
        resid = flat_block - np.median(flat_block, axis=along_axis, keepdims=True)
        noise = float(np.median(np.abs(resid))) * 1.4826
    else:
        flat = profile[a0:a1]
        noise = float(np.median(np.abs(flat - np.median(flat)))) * 1.4826
    noise = max(noise, 0.5)  # quantisation floor: never claim sub-level noise

    # 10-90% rise width -> Gaussian sigma.
    lo_v, hi_v = min(outer, inner), max(outer, inner)
    if contrast > 2 * noise:
        t10, t90 = lo_v + 0.1 * contrast, lo_v + 0.9 * contrast
        seg = profile[a0:b1]
        above10 = np.where(seg >= t10)[0]
        above90 = np.where(seg >= t90)[0]
        if len(above10) and len(above90):
            rise = abs(int(above90[0]) - int(above10[0]))
            psf = max(0.4, rise / 2.563)
        else:
            psf = 1.0
    else:
        psf = 1.0

    return ChannelConditions(
        contrast=max(contrast, 1e-6),
        noise_sigma=noise,
        psf_sigma_px=psf,
        pixel_pitch_px=1.0,
        rows=max(1, rows),
    )


@dataclass(frozen=True)
class InformationAudit:
    bound_px: float
    bound_ratio_pp: float
    reported_ratio_pp: float
    efficiency: float
    physically_possible: bool
    channel: ChannelConditions
    advice: tuple[str, ...]
    rhythm: Optional[TemporalSpatialRhythm] = None
    fused_bound_px: Optional[float] = None

    def describe(self) -> str:
        lines = [
            f"edge localisation floor : {self.bound_px:.3f} px "
            f"(SNR {self.channel.snr:.0f}, blur {self.channel.psf_sigma_px:.2f} px, "
            f"{self.channel.rows} rows)",
            f"floor on the ratio      : +/-{self.bound_ratio_pp:.3f} pp",
            f"reported                : +/-{self.reported_ratio_pp:.3f} pp",
        ]
        if not self.physically_possible:
            lines.append(
                "  IMPOSSIBLE: the reported uncertainty is below the "
                "information-theoretic floor. The error bar is wrong."
            )
        else:
            lines.append(
                f"  efficiency {self.efficiency * 100:.0f}% of the theoretical limit"
            )
        if self.rhythm is not None:
            lines.append(f"  {self.rhythm.describe()}")
            if self.fused_bound_px is not None and math.isfinite(self.fused_bound_px):
                lines.append(
                    f"  wash-adjusted fusion floor : {self.fused_bound_px:.3f} px "
                    f"(effective rows {self.rhythm.effective_rows:.1f})"
                )
        for a in self.advice:
            lines.append(f"  -> {a}")
        return "\n".join(lines)


def audit_measurement(
    channel: ChannelConditions,
    reported_ratio_sigma_pp: float,
    border_a_mm: float,
    border_b_mm: float,
    px_per_mm: float,
    *,
    shot_ratio: Optional[float] = None,
    frame_chi2_dof: Optional[float] = None,
    potential: float = 0.0,
) -> InformationAudit:
    """Compare a reported uncertainty against what the physics allows."""
    bound_px = cramer_rao_edge_px(channel)
    bound_pp = cramer_rao_ratio_pp(channel, border_a_mm, border_b_mm, px_per_mm)

    possible = reported_ratio_sigma_pp >= bound_pp * 0.98  # 2% numerical slack
    efficiency = (
        min(1.0, bound_pp / reported_ratio_sigma_pp)
        if reported_ratio_sigma_pp > 0
        else 0.0
    )

    rhythm = temporal_spatial_rhythm(
        channel,
        efficiency=efficiency,
        shot_ratio=shot_ratio,
        frame_chi2_dof=frame_chi2_dof,
        potential=potential,
    )
    fused_bound_px = cramer_rao_edge_px(apply_rhythm_to_channel(channel, rhythm))

    advice: list[str] = []
    regime = "photon-limited"
    if not possible:
        regime = "impossible"
        advice.append(
            "the reported error bar is smaller than any estimator can achieve "
            "on this image; treat the measurement as unvalidated"
        )
    elif efficiency > 0.5:
        advice.append(
            "the detector is near the information limit. Better code will not "
            "help; only a better photograph will."
        )
    elif efficiency > 0.05:
        advice.append(
            f"the detector is using about {efficiency * 100:.0f}% of the "
            "available photon information; there is some headroom in the algorithm"
        )
    else:
        # This is the usual regime on a real card, and it is not a criticism of
        # the detector. The bound describes an ideal straight step edge. A
        # printed card border is not one: the ink boundary wanders, the die cut
        # wanders, and the reported sigma is measured from that real scatter.
        # When the gap is this large the measurement is limited by the CARD, not
        # by the camera, and no amount of light or lens will close it.
        regime = "specimen-limited"
        advice.append(
            f"reported uncertainty is {1 / max(efficiency, 1e-12):.0f}x the "
            "photon floor. The limit here is the card itself -- real ink and cut "
            "boundaries wander by more than the sensor noise -- not the capture "
            "or the algorithm. More light will not narrow this."
        )

    if rhythm.boost < 0.85:
        advice.append(
            f"temporal-spatial wash is active (boost {rhythm.boost:.2f}, "
            f"coherence {rhythm.coherence:.2f} vs gradient {rhythm.gradient:.2f}); "
            "do not credit the full 1/sqrt(N) from extra rows or frames"
        )
    elif rhythm.coherence > 0.6 and rhythm.boost > 1.05:
        advice.append(
            f"channel observables are coherent (weyl {rhythm.weyl:.2f} rad, "
            f"boost {rhythm.boost:.2f}); extra frames can tighten the fusion "
            "bound, but not the single-shot CRB"
        )

    # Capture advice is only actionable when photons are actually the binding
    # constraint. Telling someone to add light when they are specimen-limited is
    # advice that cannot work.
    if regime == "specimen-limited":
        return InformationAudit(
            bound_px=bound_px,
            bound_ratio_pp=bound_pp,
            reported_ratio_pp=reported_ratio_sigma_pp,
            efficiency=efficiency,
            physically_possible=possible,
            channel=channel,
            advice=tuple(advice),
            rhythm=rhythm,
            fused_bound_px=fused_bound_px,
        )

    # Rank the capture fixes by their actual exponents in the bound.
    if channel.snr < 25:
        advice.append(
            f"contrast-to-noise is {channel.snr:.0f}. Error scales as 1/SNR, so "
            "more light is the single largest available improvement -- doubling "
            "it halves the error bar."
        )
    if channel.psf_sigma_px > 1.6:
        advice.append(
            f"blur is {channel.psf_sigma_px:.1f} px. Error scales as "
            "sqrt(blur), so steadying the shot helps, but only by sqrt(2) per "
            "halving -- less than light does."
        )
    if channel.rows < 200:
        advice.append(
            f"only {channel.rows} rows contribute. Error scales as "
            "1/sqrt(rows), so filling more of the frame with the card is worth "
            "roughly as much as the same factor in light."
        )
    return InformationAudit(
        bound_px=bound_px,
        bound_ratio_pp=bound_pp,
        reported_ratio_pp=reported_ratio_sigma_pp,
        efficiency=efficiency,
        physically_possible=possible,
        channel=channel,
        advice=tuple(advice),
        rhythm=rhythm,
        fused_bound_px=fused_bound_px,
    )


def audit_result(
    result,
    side: Optional[str] = None,
    *,
    shot_ratio: Optional[float] = None,
    frame_chi2_dof: Optional[float] = None,
    potential: float = 0.0,
) -> InformationAudit:
    """Audit a CenteringResult against the channel in its own rectified image."""
    if result.rectified is None:
        raise ValueError("audit needs the rectified image; measure with keep_rectified=True")
    pair = result.worst_axis
    side = side or (pair.low_name)
    depth = pair.low_mm.value if side == pair.low_name else pair.high_mm.value
    channel = measure_channel(result.rectified, side, result.px_per_mm, depth)
    return audit_measurement(
        channel,
        pair.ratio_pct.sigma,
        pair.low_mm.value,
        pair.high_mm.value,
        result.px_per_mm,
        shot_ratio=shot_ratio,
        frame_chi2_dof=frame_chi2_dof,
        potential=potential,
    )


# ---------------------------------------------------------------------------
# The actual quantum floor: photon shot noise
# ---------------------------------------------------------------------------
#
# WHY ENTANGLEMENT DOES NOT ENTER HERE
#
# Li et al., Optica (2026), generate polarization-entangled photon pairs by
# using concentrated sunlight as the PUMP for spontaneous parametric
# down-conversion in a ppKTP crystal inside a Sagnac interferometer, reaching
# concurrence 0.905 +/- 0.053 and Bell S = 2.54 +/- 0.22. Good work, and the
# point of it is energy efficiency: replacing a laser pump.
#
# It does not reach this problem, for four mechanical reasons:
#
#   1. The entanglement is CREATED in the crystal. Sunlight is the pump and is
#      annihilated in down-conversion. Light reflecting off a cardboard card has
#      undergone no nonlinear process, so there is no entanglement present to be
#      lost, degraded, or recovered.
#   2. Reflected imaging light is in a thermal/chaotic state. It does show
#      photon bunching (Hanbury Brown-Twiss), but that is a classical intensity
#      correlation, not entanglement, and it carries no extra positional
#      information about an edge.
#   3. A CMOS sensor is a photon-counting INTENSITY detector. It discards phase
#      and polarization. Demonstrating entanglement requires coincidence
#      detection across two separated detectors with independent basis choice --
#      two detectors and a coincidence window, not one sensor.
#   4. "Interstitial pixel space" is the fill-factor dead area between
#      photodiodes plus the microlens array. Photons landing there are absorbed
#      or redirected. That is ordinary optical loss and it is already in the
#      model below as quantum efficiency.
#
# What IS quantum about this measurement, and genuinely so, is photon shot
# noise: arrivals are Poisson because light is quantised. That sets the Standard
# Quantum Limit for intensity imaging, and it is the correct floor to compute.
# The Gaussian bound above is an approximation to it; this is the real thing.


@dataclass(frozen=True)
class SensorModel:
    """Enough of a camera to convert grey levels into photoelectrons.

    Defaults are a typical small-pixel phone sensor. They are estimates; the
    ``shot_noise_consistency`` check below tests them against the image's own
    noise rather than asking you to trust them.
    """

    full_well_e: float = 6000.0
    bit_depth: int = 8
    iso: float = 100.0
    base_iso: float = 100.0
    quantum_efficiency: float = 0.65

    @property
    def electrons_per_level(self) -> float:
        levels = float(2**self.bit_depth - 1)
        gain = max(1.0, self.iso / self.base_iso)
        return self.full_well_e / levels / gain

    def electrons(self, level: float) -> float:
        return max(0.0, level) * self.electrons_per_level

    def shot_noise_levels(self, level: float) -> float:
        """Expected noise in grey levels if the pixel is shot-noise limited."""
        e = self.electrons(level)
        if e <= 0:
            return 0.0
        return math.sqrt(e) / self.electrons_per_level


def shot_noise_fisher_edge(
    c: ChannelConditions, background_level: float, sensor: SensorModel
) -> float:
    """Fisher information for a POISSON channel, in 1/px^2.

    For photon counting the likelihood is Poisson, not Gaussian, so the
    information is sum (d mu / d theta)^2 / mu with mu in photoelectrons --
    the variance is the mean rather than a fixed sigma^2. This is the Standard
    Quantum Limit for intensity-based edge localisation.
    """
    if c.contrast <= 0 or c.psf_sigma_px <= 0:
        return 0.0
    c_e = sensor.electrons(c.contrast)
    mu_e = sensor.electrons(max(background_level, c.contrast * 0.5))
    if mu_e <= 0:
        return 0.0
    per_row = (c_e**2 / mu_e) * _PHI_SQ_INTEGRAL / (
        c.psf_sigma_px * c.pixel_pitch_px
    )
    return per_row * max(1, c.rows)


def quantum_floor_px(
    c: ChannelConditions, background_level: float, sensor: SensorModel
) -> float:
    """Standard Quantum Limit on edge localisation, in pixels."""
    info = shot_noise_fisher_edge(c, background_level, sensor)
    return float("inf") if info <= 0 else 1.0 / math.sqrt(info)


def shot_noise_consistency(
    c: ChannelConditions, background_level: float, sensor: SensorModel
) -> tuple[float, str]:
    """Is this image actually shot-noise limited? Returns (ratio, verdict).

    Compares the measured per-pixel noise against what Poisson statistics alone
    would produce. A ratio near 1 means photons are the binding constraint. Much
    above 1 means read noise, JPEG quantisation or demosaic dominate -- and in
    that case the sensor is not even reaching its own classical limit, so
    anything quantum is many rungs further away.
    """
    predicted = sensor.shot_noise_levels(background_level)
    if predicted <= 0:
        return float("inf"), "cannot predict shot noise at this exposure"
    ratio = c.noise_sigma / predicted
    if ratio < 0.7:
        verdict = (
            "measured noise is BELOW the shot-noise prediction, which is "
            "impossible for an honest sensor model -- the image has been "
            "denoised, or the sensor parameters are wrong"
        )
    elif ratio < 1.6:
        verdict = "shot-noise limited: photons are the binding noise source"
    elif ratio < 5.0:
        verdict = (
            f"noise is {ratio:.1f}x the shot-noise floor; read noise and "
            "compression dominate, not photon statistics"
        )
    else:
        verdict = (
            f"noise is {ratio:.0f}x the shot-noise floor. This image is nowhere "
            "near photon-limited; it is limited by processing and compression."
        )
    return ratio, verdict


@dataclass(frozen=True)
class VarianceBudget:
    """Where the uncertainty actually comes from, and what fixing each part buys.

    Variances add, so sigma_total^2 = sigma_photon^2 + sigma_everything_else^2.
    That decomposition is what turns "could a better sensor help" from an
    opinion into arithmetic.
    """

    total_pp: float
    photon_pp: float
    residual_pp: float
    photon_share: float
    gain_from_perfect_sensor_pct: float
    shot_ratio: float
    shot_verdict: str

    def describe(self) -> str:
        lines = [
            f"total reported uncertainty : +/-{self.total_pp:.4f} pp",
            f"  photon shot-noise part   : +/-{self.photon_pp:.6f} pp "
            f"({self.photon_share * 100:.4f}% of the variance)",
            f"  everything else          : +/-{self.residual_pp:.4f} pp",
            "",
            f"sensor noise vs shot-noise floor: {self.shot_ratio:.1f}x",
            f"  {self.shot_verdict}",
            "",
            "If photon noise were driven to ZERO -- a perfect detector, or any "
            "quantum-enhanced scheme reaching Heisenberg scaling -- the reported",
            f"uncertainty would improve by {self.gain_from_perfect_sensor_pct:.5f}%.",
        ]
        if self.gain_from_perfect_sensor_pct < 1.0:
            lines.append("")
            lines.append(
                "That is the whole case. The measurement is limited by the "
                "physical card -- ink boundaries and die cuts wander by far more "
                "than the light does -- so improving the light, the sensor, or "
                "the quantum state of the illumination cannot move it."
            )
        return "\n".join(lines)


def variance_budget(
    reported_ratio_sigma_pp: float,
    c: ChannelConditions,
    background_level: float,
    border_a_mm: float,
    border_b_mm: float,
    px_per_mm: float,
    sensor: Optional[SensorModel] = None,
) -> VarianceBudget:
    """Decompose the reported uncertainty into photon and non-photon parts."""
    sensor = sensor or SensorModel()
    floor_px = quantum_floor_px(c, background_level, sensor)

    if math.isfinite(floor_px):
        sigma_mm = floor_px / max(px_per_mm, 1e-9) * math.sqrt(2.0)
        a, b = border_a_mm, border_b_mm
        total = a + b
        wider, narrower = (a, b) if a >= b else (b, a)
        photon_pp = (
            100.0 * math.hypot(narrower * sigma_mm, wider * sigma_mm) / total**2
            if total > 0
            else float("inf")
        )
    else:
        photon_pp = float("inf")

    tot = max(reported_ratio_sigma_pp, 1e-12)
    share = min(1.0, (photon_pp / tot) ** 2) if math.isfinite(photon_pp) else 1.0
    residual = math.sqrt(max(0.0, tot**2 - min(photon_pp, tot) ** 2))
    gain_pct = (1.0 - math.sqrt(max(0.0, 1.0 - share))) * 100.0

    ratio, verdict = shot_noise_consistency(c, background_level, sensor)
    return VarianceBudget(
        total_pp=tot,
        photon_pp=photon_pp,
        residual_pp=residual,
        photon_share=share,
        gain_from_perfect_sensor_pct=gain_pct,
        shot_ratio=ratio,
        shot_verdict=verdict,
    )
