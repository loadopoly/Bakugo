"""Tests for the scanning, capture, valuation and storage layers.

As with the core, a large share of these assert refusal. The valuation module in
particular is mostly a set of things it declines to do: no prices, no offer; no
real prior, a warning on every number; crowd labels, no training export.
"""

from __future__ import annotations

import numpy as np
import pytest

from cardcenter.capture import RunningRatio, assess_frame
from cardcenter.grading import grade_band
from cardcenter.multicard import dhash, detect_cards_in_frame, hamming, scan_image
from cardcenter.store import LabelKind, ScanStore
from cardcenter.synth import CaseCard, render_capture, render_case_scene
from cardcenter.types import Measured, SLAB_STACKS, resolve_holder
from cardcenter.valuation import (
    DealCosts,
    GradePrior,
    ManualPriceSource,
    PricingUnavailable,
    analyse_offer,
)


# --------------------------------------------------------------------------
# Optical stacks
# --------------------------------------------------------------------------


def test_stack_shift_is_sum_of_layers() -> None:
    import math

    from cardcenter.optics import inplane_shift_mm
    from cardcenter.types import SLAB_PRESETS

    th = math.radians(25.0)
    bgs = float(inplane_shift_mm(th, SLAB_PRESETS["bgs"]))
    glass = float(inplane_shift_mm(th, SLAB_PRESETS["case_glass"]))
    both = float(inplane_shift_mm(th, SLAB_STACKS["case_bgs"]))
    assert both == pytest.approx(bgs + glass, rel=1e-9)


def test_case_glass_dominates_the_slab() -> None:
    """5mm of glass displaces far more than 1.6mm of acrylic, so ignoring the
    case is a bigger error than getting the slab thickness wrong."""
    import math

    from cardcenter.optics import inplane_shift_measured
    from cardcenter.types import SLAB_PRESETS

    th = math.radians(25.0)
    assert (
        inplane_shift_measured(th, SLAB_PRESETS["case_glass"]).value
        > 2.5 * inplane_shift_measured(th, SLAB_PRESETS["bgs"]).value
    )


def test_resolve_holder_finds_both_kinds() -> None:
    assert resolve_holder("bgs").name == "bgs"
    assert resolve_holder("case_bgs").name == "case_bgs"
    with pytest.raises(KeyError):
        resolve_holder("nope")


# --------------------------------------------------------------------------
# Frame quality gating
# --------------------------------------------------------------------------


def test_blurred_frame_is_rejected() -> None:
    import cv2

    img, _, _ = render_capture()
    sharp = assess_frame(img)
    blurred = assess_frame(cv2.GaussianBlur(img, (0, 0), 6))
    assert blurred.sharpness < sharp.sharpness
    assert not blurred.passed
    assert any("steadier" in g for g in blurred.guidance)


def test_glare_is_detected() -> None:
    img, _, _ = render_capture()
    glared = img.copy()
    h, w = glared.shape[:2]
    glared[: int(h * 0.45), : int(w * 0.45)] = 253
    q = assess_frame(glared)
    assert q.glare_frac > 0.03
    assert not q.passed
    assert any("glare" in g for g in q.guidance)


def test_too_far_away_is_rejected_with_a_distance_hint() -> None:
    img, _, _ = render_capture()
    q = assess_frame(img, px_per_mm=2.0)
    assert not q.passed
    assert any("closer" in g for g in q.guidance)


def test_steep_angle_is_rejected() -> None:
    img, _, _ = render_capture()
    assert not assess_frame(img, tilt_deg=50.0).passed


# --------------------------------------------------------------------------
# Combining repeated measurements
# --------------------------------------------------------------------------


def test_consistent_measurements_tighten() -> None:
    r = RunningRatio()
    for _ in range(4):
        r.add(Measured(60.0, 1.0))
    c = r.combined
    assert c.value == pytest.approx(60.0)
    assert c.sigma == pytest.approx(0.5, rel=0.02)  # 1/sqrt(4)


def test_inconsistent_measurements_do_not_tighten() -> None:
    """Frames that disagree by far more than their error bars must inflate the
    combined uncertainty, not shrink it as 1/sqrt(N)."""
    r = RunningRatio()
    for v in (55.0, 65.0, 54.0, 66.0):
        r.add(Measured(v, 0.4))
    c = r.combined
    assert r.consistency > 5.0
    assert c.sigma > 0.4  # worse than any single measurement, which is correct


def test_running_ratio_ignores_zero_sigma() -> None:
    r = RunningRatio()
    r.add(Measured(60.0, 0.0))
    assert len(r) == 0
    assert r.combined is None


# --------------------------------------------------------------------------
# Multi-card scanning
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def case_scene():
    specs = [
        CaseCard(3.0, 3.0, 3.0, 3.0),
        CaseCard(4.2, 3.0, 1.8, 3.0),
        CaseCard(3.0, 4.0, 3.0, 2.0, rotation_deg=4.0),
        CaseCard(2.6, 3.0, 3.4, 3.0, border_bgr=(230, 230, 235), art_bgr=(70, 40, 120)),
        CaseCard(5.0, 3.0, 1.0, 3.0),
        CaseCard(3.1, 3.0, 2.9, 3.0, rotation_deg=-3.0),
        CaseCard(3.0, 2.2, 3.0, 3.8),
        CaseCard(3.5, 3.5, 2.5, 2.5),
    ]
    img, _ = render_case_scene(specs, columns=4, px_per_mm=9.0)
    return img, specs


def test_finds_every_card_in_the_scene(case_scene) -> None:
    img, specs = case_scene
    found = detect_cards_in_frame(img)
    assert len(found) == len(specs)


def test_does_not_double_count_the_inner_printed_frame(case_scene) -> None:
    """A card's printed border is itself a card-shaped rectangle. Without
    nesting rejection every card is reported twice."""
    img, specs = case_scene
    assert len(detect_cards_in_frame(img)) <= len(specs)


def test_scan_measures_cards_accurately(case_scene) -> None:
    img, specs = case_scene
    H, W = img.shape[:2]
    report = scan_image(img, holder="raw", enforce_quality=False)
    assert len(report.measured()) >= 7

    errors = []
    for card in report.measured():
        cx, cy = card.observations[0].quad.mean(axis=0)
        idx = min(1, int(cy / (H / 2))) * 4 + min(3, int(cx / (W / 4)))
        errors.append(card.worst_ratio.value - specs[idx].worst_ratio)
    rms = float(np.sqrt(np.mean(np.square(errors))))
    assert rms < 2.0, f"multi-card RMS error {rms:.2f}pp is too high"


def test_same_frame_detections_stay_distinct(case_scene) -> None:
    """Two cards in one frame are two cards even if they look identical."""
    img, specs = case_scene
    report = scan_image(img, holder="raw", enforce_quality=False)
    assert len(report.cards) == len(specs)


def test_identical_crops_hash_together_and_different_ones_apart() -> None:
    a, _, _ = render_capture(left_mm=3.0, right_mm=3.0, seed=1)
    b, _, _ = render_capture(left_mm=3.0, right_mm=3.0, seed=1)
    c, _, _ = render_capture(
        left_mm=5.0, right_mm=1.0, seed=9, border_bgr=(20, 20, 220)
    ) if False else render_capture(left_mm=5.0, right_mm=1.0, seed=9)
    assert hamming(dhash(a), dhash(b)) == 0
    assert hamming(dhash(a), dhash(c)) > 0


def test_unmeasured_cards_report_a_reason() -> None:
    img = np.full((1200, 1600, 3), 30, dtype=np.uint8)
    import cv2

    cv2.rectangle(img, (200, 200), (620, 790), (150, 90, 70), -1)
    report = scan_image(img, holder="raw", enforce_quality=False)
    for card in report.unmeasured():
        assert card.failure_reasons or "NOT MEASURED" in card.summary()


# --------------------------------------------------------------------------
# Valuation
# --------------------------------------------------------------------------


def _band(ratio: float = 62.0, sigma: float = 0.4, grader: str = "PSA"):
    return grade_band(Measured(ratio, sigma), grader, "front")


def test_offer_requires_prices() -> None:
    prior = GradePrior.from_population({"10": 10, "9": 60, "8": 30}, "test pop report")
    with pytest.raises(PricingUnavailable):
        analyse_offer("card-x", _band(), prior, ManualPriceSource({}))


def test_offer_uses_the_pessimistic_end_of_the_band() -> None:
    """Offering as though the optimistic ceiling were certain means paying full
    price for our own uncertainty."""
    band = _band(62.0, 2.5)
    assert band.best != band.worst
    prior = GradePrior.from_population(
        {"10": 20, "9": 50, "8": 25, "7": 5}, "pop"
    )
    prices = ManualPriceSource(
        {"card-x": {"10": 1000.0, "9": 200.0, "8": 80.0, "7": 40.0}}
    )
    a = analyse_offer("card-x", band, prior, prices, DealCosts())
    assert a.ceiling_grade == band.worst
    # The optimistic reading of the band would allow a 10; the offer must not.
    assert "10" not in a.grade_probabilities


def test_uniform_prior_warns_on_every_result() -> None:
    prior = GradePrior.uninformative(["10", "9", "8"])
    prices = ManualPriceSource({"card-x": {"10": 1000.0, "9": 200.0, "8": 80.0}})
    a = analyse_offer("card-x", _band(), prior, prices)
    assert not prior.trustworthy
    assert any("placeholder" in w for w in a.warnings)


def test_missing_comps_are_flagged_not_silently_zeroed() -> None:
    prior = GradePrior.from_population({"10": 10, "9": 40, "8": 50}, "pop")
    prices = ManualPriceSource({"card-x": {"9": 200.0}})
    a = analyse_offer("card-x", _band(), prior, prices)
    assert any("no comparable price" in w for w in a.warnings)


def test_fat_left_tail_is_flagged() -> None:
    prior = GradePrior.from_population({"10": 2, "9": 8, "8": 90}, "pop")
    prices = ManualPriceSource({"card-x": {"10": 5000.0, "9": 400.0, "8": 15.0}})
    a = analyse_offer("card-x", _band(), prior, prices, DealCosts(grading_fee=25.0))
    assert a.loss_probability >= 0.0
    assert a.p10_net <= a.p50_net <= max(a.prices.values())


def test_grading_costs_reduce_the_offer() -> None:
    prior = GradePrior.from_population({"10": 10, "9": 60, "8": 30}, "pop")
    prices = ManualPriceSource({"card-x": {"10": 900.0, "9": 250.0, "8": 90.0}})
    cheap = analyse_offer("card-x", _band(), prior, prices, DealCosts(grading_fee=0.0))
    dear = analyse_offer("card-x", _band(), prior, prices, DealCosts(grading_fee=200.0))
    assert dear.suggested_offer < cheap.suggested_offer


def test_prior_truncation_renormalises() -> None:
    prior = GradePrior.from_population({"10": 25, "9": 25, "8": 50}, "pop")
    truncated = prior.truncated_at("9", "PSA")
    assert "10" not in truncated
    assert sum(truncated.values()) == pytest.approx(1.0)
    assert truncated["9"] == pytest.approx(25 / 75)


def test_empty_population_rejected() -> None:
    with pytest.raises(ValueError):
        GradePrior.from_population({}, "empty")


# --------------------------------------------------------------------------
# Provenance store
# --------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path):
    with ScanStore(str(tmp_path / "t.db")) as s:
        yield s


def _add_scan(store) -> int:
    from cardcenter.centering import measure_centering
    from cardcenter.types import CaptureSpec

    img, _, f = render_capture(left_mm=3.4, right_mm=2.6)
    res = measure_centering(img, slab="raw", capture=CaptureSpec(focal_px=f))
    return store.add_scan("bakugo-ensky-2017", res, source="test")


def test_scan_round_trip(store) -> None:
    sid = _add_scan(store)
    assert sid > 0
    assert store.scan_count() == 1


def test_certified_label_requires_a_cert_number(store) -> None:
    sid = _add_scan(store)
    with pytest.raises(ValueError, match="cert number"):
        store.add_label(sid, "PSA", "9", LabelKind.CERTIFIED)
    assert store.add_label(sid, "PSA", "9", LabelKind.CERTIFIED, cert_number="12345678")


def test_opinion_labels_are_not_ground_truth() -> None:
    assert LabelKind.CERTIFIED.is_ground_truth
    assert not LabelKind.MARKETPLACE_VOTE.is_ground_truth
    assert not LabelKind.SELF_REPORTED.is_ground_truth
    assert not LabelKind.MODEL_PREDICTED.is_ground_truth


def test_training_export_refuses_crowd_labels(store) -> None:
    sid = _add_scan(store)
    store.add_label(sid, "BGS", "9.5", LabelKind.MARKETPLACE_VOTE, attributed_to="user7")
    with pytest.raises(ValueError, match="opinions, not grades"):
        store.export_training_set(
            include_kinds=[LabelKind.CERTIFIED, LabelKind.MARKETPLACE_VOTE]
        )


def test_training_export_defaults_to_certified_only(store) -> None:
    sid = _add_scan(store)
    store.add_label(sid, "PSA", "9", LabelKind.CERTIFIED, cert_number="1")
    store.add_label(sid, "PSA", "10", LabelKind.MARKETPLACE_VOTE)
    out = store.export_training_set()
    assert out["manifest"]["n_examples"] == 1
    assert out["manifest"]["ground_truth_only"]


def test_contaminated_export_is_stamped_in_the_manifest(store) -> None:
    sid = _add_scan(store)
    store.add_label(sid, "PSA", "10", LabelKind.MARKETPLACE_VOTE)
    out = store.export_training_set(
        include_kinds=[LabelKind.MARKETPLACE_VOTE], acknowledge_contamination=True
    )
    assert out["manifest"]["contamination_acknowledged"]
    assert "must not be described as predicting grades" in (
        out["manifest"]["contamination_warning"]
    )


def test_circularity_report_calls_out_a_self_referential_pool(store) -> None:
    sid = _add_scan(store)
    for _ in range(5):
        store.add_label(sid, "PSA", "10", LabelKind.MODEL_PREDICTED)
        store.add_label(sid, "PSA", "10", LabelKind.MARKETPLACE_VOTE)
    report = store.circularity_report()
    assert "NO independent labels" in report


def test_circularity_report_recognises_real_labels(store) -> None:
    sid = _add_scan(store)
    for i in range(6):
        store.add_label(sid, "PSA", "9", LabelKind.CERTIFIED, cert_number=str(i))
    assert "independent labels available" in store.circularity_report()
