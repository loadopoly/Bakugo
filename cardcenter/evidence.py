"""Multi-view evidence: combining views without lying about what they agree on.

THE MEASUREMENT THAT SHAPED THIS MODULE
----------------------------------------
A static card is the favourable case for multi-observer fusion: nothing changes
between frames, so the simultaneity problem that sank the two-view lightning
work simply does not arise. Fisher information adds across independent views and
the Cramer-Rao bound should fall as 1/sqrt(N).

Tested on two real views of the same card from one capture burst:

    view A   54.1 / 45.9      reported sigma 1.67
    view B   66.6 / 33.4      reported sigma 1.67
    inverse-variance combination -> 54.18 with sigma 0.288

The two views disagree by 12.5 percentage points while each claims +/-1.67, and
naive combination reports 0.288 -- a SIX-FOLD tightening onto an answer that
cannot be right, because at most one of the inputs is. chi2/dof came to 16.8.

This is the central failure mode of multi-view fusion and it gets worse, not
better, with more views: every additional disagreeing measurement shrinks the
reported interval. So the module below is built around detecting disagreement
first and combining second.

WHAT THAT IMPLIES, CONCRETELY
------------------------------
1. Never report a combined estimate without a consistency test. chi2/dof is the
   test, and above ~3 the views are measuring different things.
2. When views disagree, the right output is not an average. It is either the
   single most reliable view, or a refusal -- an average of two incompatible
   readings is a number with no referent.
3. Additional views are only worth collecting while they change the answer.
   Wald's SPRT gives the optimal stopping rule for that, and it is dramatically
   cheaper than a fixed frame count near a decision boundary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence

import numpy as np

from .types import Measured

# chi2/dof above this means the views are inconsistent, not merely noisy.
MAX_CONSISTENCY = 3.0
# Below this, disagreement is small enough that pooling is honest.
GOOD_CONSISTENCY = 1.5


class Verdict(str, Enum):
    ABOVE = "above"          # confidently better than the threshold
    BELOW = "below"          # confidently worse
    UNDECIDED = "undecided"  # need more evidence
    INCONSISTENT = "inconsistent"  # views disagree; do not pool


@dataclass(frozen=True)
class Fusion:
    """Result of combining several views of one card."""

    combined: Optional[Measured]
    n_views: int
    consistency: float          # chi2/dof
    inflated_sigma: float       # sigma after PDG scaling
    naive_sigma: float          # what pooling would have claimed
    values: tuple[float, ...]
    trustworthy: bool
    reason: str

    @property
    def inflation_factor(self) -> float:
        return self.inflated_sigma / max(self.naive_sigma, 1e-12)

    def describe(self) -> str:
        if self.combined is None:
            return f"no fusion: {self.reason}"
        lo, hi = self.combined.interval()
        lines = [
            f"{self.n_views} views -> {self.combined.value:.1f}/"
            f"{100 - self.combined.value:.1f}  95% CI {lo:.1f}-{hi:.1f}",
            f"  consistency chi2/dof {self.consistency:.1f}"
            + ("  (views agree)" if self.consistency <= GOOD_CONSISTENCY else ""),
        ]
        if self.inflation_factor > 1.05:
            lines.append(
                f"  sigma inflated {self.inflation_factor:.1f}x for disagreement "
                f"(naive pooling would have claimed +/-{self.naive_sigma:.2f})"
            )
        if not self.trustworthy:
            lines.append(f"  NOT TRUSTWORTHY: {self.reason}")
        return "\n".join(lines)


def fuse(measurements: Sequence[Measured]) -> Fusion:
    """Inverse-variance combination with a mandatory consistency test.

    Uses the Particle Data Group's scale-factor convention: when observations
    scatter by more than their own error bars allow, inflate the combined
    uncertainty by sqrt(chi2/dof) rather than letting it shrink as 1/sqrt(N).
    Above MAX_CONSISTENCY the result is marked untrustworthy outright, because
    at that point the inputs are not repeated measurements of one quantity.
    """
    ms = [m for m in measurements if m.sigma > 0]
    if not ms:
        return Fusion(None, 0, 0.0, 0.0, 0.0, (), False, "no usable measurements")
    if len(ms) == 1:
        return Fusion(ms[0], 1, 0.0, ms[0].sigma, ms[0].sigma,
                      (ms[0].value,), True, "single view")

    v = np.array([m.value for m in ms])
    s = np.array([m.sigma for m in ms])
    w = 1.0 / s**2
    mean = float((w * v).sum() / w.sum())
    naive = float(math.sqrt(1.0 / w.sum()))
    chi2 = float((w * (v - mean) ** 2).sum() / (len(v) - 1))
    scale = math.sqrt(max(1.0, chi2))
    inflated = naive * scale

    ok = chi2 <= MAX_CONSISTENCY
    reason = (
        "views agree"
        if ok
        else f"views disagree (chi2/dof {chi2:.1f}); spread {v.min():.1f}-{v.max():.1f} "
        f"exceeds their own error bars -- at most one can be right"
    )
    return Fusion(
        Measured(mean, inflated), len(ms), chi2, inflated, naive,
        tuple(float(x) for x in v), ok, reason,
    )


def best_single(measurements: Sequence[Measured]) -> Optional[Measured]:
    """Fall back to the most precise single view.

    When views disagree there is no honest average, but there is usually one
    view that was captured better than the others. Reporting it -- with its own
    unreduced uncertainty -- is more defensible than pooling incompatible
    readings.
    """
    ms = [m for m in measurements if m.sigma > 0]
    return min(ms, key=lambda m: m.sigma) if ms else None


# ---------------------------------------------------------------------------
# Sequential testing: stop when the answer is decided, not at a frame count
# ---------------------------------------------------------------------------


@dataclass
class SequentialBoundaryTest:
    """Wald's SPRT for 'is this card above or below a grading threshold'.

    A fixed number of frames is the wrong stopping rule in both directions: a
    card at 70/30 against a 55/45 boundary is decided on the first frame and
    further capture is wasted, while a card at 54.9/45.1 may never be decided
    and the user should be told so rather than handed a coin flip.

    SPRT is the optimal test in the sense that no other test with the same error
    rates uses fewer samples on average. The log-likelihood ratio accumulates
    per view, and the test stops when it crosses either boundary.

    ``threshold`` is the grading boundary (e.g. 55.0 for a PSA 10 ceiling);
    ``margin`` is how far either side counts as decisively above or below.
    """

    threshold: float
    margin: float = 1.5
    alpha: float = 0.05  # false 'above'
    beta: float = 0.05   # false 'below'
    llr: float = 0.0
    n: int = 0
    _history: list[float] = field(default_factory=list)

    @property
    def upper(self) -> float:
        return math.log((1 - self.beta) / self.alpha)

    @property
    def lower(self) -> float:
        return math.log(self.beta / (1 - self.alpha))

    def update(self, m: Measured) -> Verdict:
        """Add one view's measurement and re-test."""
        if m.sigma <= 0:
            return self.verdict
        self.n += 1
        self._history.append(m.value)
        # Gaussian likelihood ratio for mean above vs below the threshold,
        # separated by +/- margin.
        hi = self.threshold + self.margin
        lo = self.threshold - self.margin
        ll_hi = -((m.value - hi) ** 2) / (2 * m.sigma**2)
        ll_lo = -((m.value - lo) ** 2) / (2 * m.sigma**2)
        self.llr += ll_hi - ll_lo
        return self.verdict

    @property
    def verdict(self) -> Verdict:
        if self.n == 0:
            return Verdict.UNDECIDED
        if self.llr >= self.upper:
            return Verdict.ABOVE
        if self.llr <= self.lower:
            return Verdict.BELOW
        return Verdict.UNDECIDED

    @property
    def decided(self) -> bool:
        return self.verdict in (Verdict.ABOVE, Verdict.BELOW)

    def expected_remaining(self) -> Optional[int]:
        """Rough estimate of how many more views are needed, or None if stalled.

        Extrapolates the current LLR drift. A card sitting exactly on the
        boundary produces no drift, and this returns None -- which is the honest
        answer: more frames will not decide it.
        """
        if self.decided or self.n < 2:
            return None
        drift = self.llr / self.n
        # A card on the boundary produces a tiny residual drift from noise
        # rather than from real separation. Extrapolating it yields a large but
        # meaningless "views remaining" figure, so require the drift to be big
        # enough that the boundary is reachable in a plausible number of views.
        min_drift = (self.upper - abs(self.llr)) / 60.0
        if abs(drift) < max(1e-6, min_drift):
            return None
        target = self.upper if drift > 0 else self.lower
        remaining = (target - self.llr) / drift
        return int(math.ceil(remaining)) if 0 < remaining < 500 else None

    def describe(self) -> str:
        v = self.verdict
        lines = [
            f"SPRT vs {self.threshold:.0f}/{100 - self.threshold:.0f}: "
            f"{v.value.upper()} after {self.n} view(s)"
        ]
        lines.append(f"  log-likelihood ratio {self.llr:+.2f} "
                     f"(decide at {self.lower:+.2f} / {self.upper:+.2f})")
        if not self.decided:
            rem = self.expected_remaining()
            if rem is None:
                lines.append(
                    "  no drift -- this card sits on the boundary and more "
                    "views will not decide it. Report the interval, not a side."
                )
            else:
                lines.append(f"  about {rem} more view(s) should decide it")
        return "\n".join(lines)


def information_value(m: Measured, threshold: float) -> float:
    """How much a further view is worth, given where this card sits.

    Fisher information for a binary decision peaks at the boundary: a card
    measured far from the threshold is already decided and further capture
    changes nothing, while a card near it is exactly where evidence pays. This
    returns a 0..1 weight usable to decide whether to keep capturing.
    """
    if m.sigma <= 0:
        return 0.0
    z = abs(m.value - threshold) / m.sigma
    # Standard normal density, normalised to 1 at the boundary.
    return float(math.exp(-0.5 * z * z))
