"""Tests for card identification from sparse OCR.

The load-bearing property here is not the hit rate, it is that a WRONG name is
never reported confidently. A wrong species routes to a wrong price, so every
test below is really asking the same question: when the evidence is weak, does
it refuse?
"""

from __future__ import annotations

import numpy as np
import pytest

from cardcenter.recognise import (
    MIN_TOKEN_LEN,
    Recognition,
    Species,
    _allowed_edits,
    load_species,
    match_species,
    prepare_for_text,
    read_dex_number,
    recognise_card,
    tokenise,
)

VOCAB = (
    Species("Sableye", 302),
    Species("Genesect", 649),
    Species("Thundurus", 642),
    Species("Seel", 86),
    Species("Goodra", 706),
    Species("Zubat", 41),
    Species("Golbat", 42),
)


class FakeEngine:
    """Returns canned OCR text so tests never need tesseract installed."""

    psm = 11
    timeout_s = 1

    def __init__(self, text: str) -> None:
        self._text = text

    @property
    def name(self) -> str:
        return "fake"

    def read_text(self, image: np.ndarray) -> str:
        return self._text


def _card() -> np.ndarray:
    return np.full((400, 280, 3), 180, dtype=np.uint8)


# --------------------------------------------------------------------------
# Tokenising and the dex number
# --------------------------------------------------------------------------


def test_tokenise_splits_on_non_letters() -> None:
    assert tokenise("Sableye  HP70 -- No. 0302!") == ["Sableye", "HP", "No"]


def test_read_dex_number_accepts_a_printed_number() -> None:
    assert read_dex_number("NO. 0706 Dragon Pokemon") == 706
    assert read_dex_number("No, 0128 Wild Bull") == 128


def test_read_dex_number_rejects_other_numbers_on_the_card() -> None:
    """HP, damage and set totals are numbers too, and must not be mistaken
    for a dex entry."""
    assert read_dex_number("HP 130") is None
    assert read_dex_number("069/086") is None
    assert read_dex_number("NO. 9999") is None  # outside the dex


# --------------------------------------------------------------------------
# The edit budget -- the Seel regression
# --------------------------------------------------------------------------


def test_short_names_must_be_read_exactly() -> None:
    """'Seel' from readings of 'Sees'/'seal'/'peel' produced 22 false
    identifications on the real corpus. Short names get no tolerance."""
    assert _allowed_edits("seel") == 0
    for junk in ("Sees", "seal", "peel", "feel"):
        sp, _, _, _ = match_species([junk], VOCAB)
        assert sp is None, f"{junk!r} must not resolve to a species"


def test_a_short_name_read_exactly_still_resolves() -> None:
    sp, _, edits, _ = match_species(["Seel"], VOCAB)
    assert sp is not None and sp.name == "Seel" and edits == 0


def test_long_names_tolerate_ocr_damage() -> None:
    for token, expect in (("Genesec", "Genesect"), ("Thundurns", "Thundurus")):
        sp, _, _, _ = match_species([token], VOCAB)
        assert sp is not None and sp.name == expect


def test_garbage_tokens_resolve_to_nothing() -> None:
    junk = tokenise("wes 2G i (y ZF ZZ ees SLID aa Gs LIE ge Ly Lee FZ")
    sp, _, _, _ = match_species(junk, VOCAB)
    assert sp is None


def test_boilerplate_is_ignored() -> None:
    sp, _, _, _ = match_species(["Pokemon", "Trainer", "Weakness"], VOCAB)
    assert sp is None


def test_tokens_below_the_length_floor_are_ignored() -> None:
    sp, _, _, _ = match_species(["Mew", "abc"], VOCAB)
    assert sp is None


def test_a_token_between_two_species_resolves_to_neither() -> None:
    """An equidistant reading must refuse rather than pick one.

    Both candidates are six letters, so each is inside the one-edit budget and
    the tie is real rather than an artefact of the budget.
    """
    vocab = (Species("Golbat", 42), Species("Zolbat", 9999))
    sp, _, _, alts = match_species(["Xolbat"], vocab)
    assert sp is None
    assert set(alts) == {"Golbat", "Zolbat"}


# --------------------------------------------------------------------------
# End to end, including the dex cross-check
# --------------------------------------------------------------------------


def test_recognise_card_reports_a_corroborated_name() -> None:
    eng = FakeEngine("BASIC Goodra HP150 NO. 0706 Dragon Pokemon")
    r = recognise_card(_card(), vocabulary=VOCAB, engine=eng)
    assert r.resolved and r.name == "Goodra" and r.dex == 706
    assert not any("uncorroborated" in w for w in r.warnings)


def test_recognise_card_flags_an_uncorroborated_name() -> None:
    r = recognise_card(_card(), vocabulary=VOCAB, engine=FakeEngine("Sableye attack"))
    assert r.resolved and r.name == "Sableye"
    assert any("uncorroborated" in w for w in r.warnings)


def test_recognise_card_refuses_when_name_and_dex_disagree() -> None:
    """Measured on the corpus: a card read as 'Garbodor' printed dex 562, and
    Garbodor is 569. Two independent readings disagreeing means neither is
    trustworthy, and pricing the wrong card is the cost of guessing."""
    eng = FakeEngine("Genesect NO. 0302 something")
    r = recognise_card(_card(), vocabulary=VOCAB, engine=eng)
    assert not r.resolved
    assert any("disagree" in w for w in r.warnings)


def test_recognise_card_refuses_on_garbage() -> None:
    r = recognise_card(_card(), vocabulary=VOCAB, engine=FakeEngine("wes 2G ZF ees"))
    assert not r.resolved
    assert "UNIDENTIFIED" in r.describe()


def test_recognise_card_refuses_with_an_empty_vocabulary() -> None:
    r = recognise_card(_card(), vocabulary=(), engine=FakeEngine("Sableye"))
    assert not r.resolved
    assert any("vocabulary is empty" in w for w in r.warnings)


# --------------------------------------------------------------------------
# The shipped vocabulary
# --------------------------------------------------------------------------


def test_shipped_vocabulary_loads_and_is_consistent() -> None:
    v = load_species()
    assert len(v) > 200
    names = [s.name for s in v]
    assert len(names) == len(set(names)), "duplicate species names"
    dexes = [s.dex for s in v if s.dex is not None]
    assert len(dexes) == len(set(dexes)), "duplicate dex numbers"
    assert all(1 <= d <= 1025 for d in dexes)
    by_name = {s.name: s for s in v}
    # Spot-check entries this corpus actually exercised.
    assert by_name["Tauros"].dex == 128
    assert by_name["Goodra"].dex == 706
    assert by_name["Genesect"].dex == 649


def test_prepare_for_text_returns_grayscale_within_bounds() -> None:
    out = prepare_for_text(np.full((3000, 2100, 3), 120, dtype=np.uint8),
                           max_long_side=1700)
    assert out.ndim == 2
    assert max(out.shape) <= 1700
