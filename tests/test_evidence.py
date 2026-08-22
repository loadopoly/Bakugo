"""Tests for multi-view fusion and sequential stopping.

Anchored on a real measurement: two views of the same card from one capture
burst gave 54.1 and 66.6 while each claimed +/-1.67. Naive pooling reported
+/-0.288 -- a six-fold tightening onto an answer at most one input supports.
"""

from __future__ import annotations

import pytest

from cardcenter.evidence import (
    GOOD_CONSISTENCY,
    MAX_CONSISTENCY,
    SequentialBoundaryTest,
    Verdict,
    best_single,
    fuse,
    information_value,
)
from cardcenter.types import Measured


# --- fusion ---------------------------------------------------------------


def test_agreeing_views_tighten_the_estimate() -> None:
    """The favourable case: a static card, views that actually agree."""
    f = fuse([Measured(54.1, 1.6), Measured(54.9, 1.6), Measured(53.8, 1.6)])
    assert f.trustworthy
    assert f.consistency < GOOD_CONSISTENCY
    assert f.combined.sigma < 1.6  # genuinely tighter than one view


def test_the_real_disagreement_is_caught() -> None:
    """REGRESSION FROM REAL DATA. Two views of one card, 54.1 and 66.6, each
    claiming +/-1.67. Pooling would report +/-0.288."""
    f = fuse([Measured(54.1, 1.668), Measured(66.6, 1.668)])
    assert not f.trustworthy
    assert f.consistency > MAX_CONSISTENCY
    assert f.inflation_factor > 3.0
    assert "at most one can be right" in f.reason


def test_disagreement_inflates_rather_than_shrinks() -> None:
    """The failure mode is that MORE disagreeing views report MORE confidence.
    Inflation by sqrt(chi2/dof) is what prevents that."""
    f = fuse([Measured(50.0, 1.0), Measured(70.0, 1.0), Measured(60.0, 1.0)])
    assert f.inflated_sigma > f.naive_sigma
    assert f.inflated_sigma > 1.0  # worse than any single view, which is correct


def test_single_view_is_passed_through_unchanged() -> None:
    f = fuse([Measured(58.0, 1.2)])
    assert f.trustworthy and f.combined.sigma == pytest.approx(1.2)


def test_best_single_prefers_the_most_precise_view() -> None:
    b = best_single([Measured(54.0, 2.0), Measured(66.0, 0.8), Measured(60.0, 3.0)])
    assert b.sigma == pytest.approx(0.8)


def test_no_usable_measurements_yields_no_fusion() -> None:
    f = fuse([Measured(54.0, 0.0)])
    assert f.combined is None and not f.trustworthy


# --- sequential stopping --------------------------------------------------


def test_card_far_from_boundary_decides_immediately() -> None:
    """A 68/32 card against a 55/45 threshold needs one view, not twenty."""
    t = SequentialBoundaryTest(threshold=55.0)
    t.update(Measured(68.0, 1.5))
    assert t.decided and t.verdict is Verdict.ABOVE
    assert t.n == 1


def test_card_below_boundary_decides_the_other_way() -> None:
    t = SequentialBoundaryTest(threshold=55.0)
    t.update(Measured(51.0, 1.0))
    assert t.verdict is Verdict.BELOW


def test_card_on_the_boundary_is_never_decided_and_says_so() -> None:
    """The honest answer for a card sitting exactly on a threshold is that more
    frames will not help -- not a coin flip dressed as a verdict."""
    t = SequentialBoundaryTest(threshold=55.0)
    for v in (55.1, 54.9, 55.0, 55.2, 54.8, 55.1):
        t.update(Measured(v, 1.5))
    assert not t.decided
    assert t.verdict is Verdict.UNDECIDED
    assert "will not decide it" in t.describe()


def test_undecided_card_estimates_remaining_views() -> None:
    t = SequentialBoundaryTest(threshold=55.0)
    for v in (56.2, 56.4):
        t.update(Measured(v, 2.5))
    if not t.decided:
        assert t.expected_remaining() is None or t.expected_remaining() > 0


# --- boundary-adaptive sampling ------------------------------------------


def test_information_peaks_at_the_boundary() -> None:
    """Fisher information for a binary decision is maximal where you are least
    certain, which is why capture effort belongs there."""
    at = information_value(Measured(55.0, 1.5), 55.0)
    near = information_value(Measured(57.0, 1.5), 55.0)
    far = information_value(Measured(70.0, 1.5), 55.0)
    assert at == pytest.approx(1.0)
    assert at > near > far
    assert far < 0.01
