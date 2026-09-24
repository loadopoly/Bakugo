"""Sleeved cards on a light counter: the outline one band off the card.

The three stills in tests/fixtures/sleeve are the exact photos the phone sent
in the 2.20.0 field session (Freeze, 3840x2160), cards in penny sleeves on a
light wood counter:

* meganium_frame_left: the outline's left side on the printed frame, 2.4 mm
  inside the card (the silver border outside it), its right side and bottom
  on the sleeve's margin. 2.20.0 reported 68.4/31.6 (bottom 4.43 mm).
* meganium_frame_top: the same card, the outline's top on the printed frame.
  2.20.0 refused it (top border confidence 0.22).
* terapagos_shadow: the outline's bottom on the sleeve's seam, 3.2 mm of
  sleeve margin and the card's shadow inside it. 2.20.0 refused it ("a 3.81 mm
  edge shadow on the bottom is too wide to subtract").

The bands of the expected values are read off the stills by eye at 36 px/mm
(the card's edge to the printed frame), +/-0.4 mm.
"""

from pathlib import Path
import shutil

import cv2
import numpy as np
import pytest

from cardcenter.recognise import (
    NAME_BAND, Species, Word, load_species, match_species, match_words, species_key,
)
from cardcenter.serve import _identify_payload, _measure_payload

SLEEVE = Path(__file__).parent / "fixtures" / "sleeve"


def _measure(name):
    return _measure_payload((SLEEVE / name).read_bytes(), "raw", "main")


def test_outline_on_the_printed_frame_is_moved_out_to_the_card():
    d = _measure("meganium_frame_left.jpg")
    b = d["borders"]
    # 2.20.0: left on the frame, bottom on the sleeve, 68.4 reported
    assert 2.0 <= b["left"] <= 3.2 and 2.0 <= b["right"] <= 3.2
    assert 1.7 <= b["top"] <= 2.6 and 2.3 <= b["bottom"] <= 3.3
    assert 53.0 <= d["ratio"] <= 60.0
    assert any("printed frame" in w for w in d["warnings"])


def test_same_card_from_another_shot_agrees_vertically():
    """56.6 and 52.6 on 2.21.0 -- four points apart, where 2.20.0 had 68.4
    and a refusal. Not yet as close as two photos of one card should be."""
    a = _measure("meganium_frame_left.jpg")
    b = _measure("meganium_frame_top.jpg")
    va = 100 * max(a["borders"]["top"], a["borders"]["bottom"]) / (a["borders"]["top"] + a["borders"]["bottom"])
    vb = 100 * max(b["borders"]["top"], b["borders"]["bottom"]) / (b["borders"]["top"] + b["borders"]["bottom"])
    assert abs(va - vb) <= 5.0


def test_sleeve_margin_and_shadow_are_not_an_unmeasurable_shadow():
    d = _measure("terapagos_shadow.jpg")
    b = d["borders"]
    assert 2.0 <= b["bottom"] <= 3.2 and 2.0 <= b["top"] <= 3.2
    assert any("shadow" in w and "subtracted" in w for w in d["warnings"])


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="needs tesseract")
@pytest.mark.parametrize("name,species", [
    ("meganium_frame_left.jpg", "Meganium"),     # 2.20.0: "Seaking", from "soaking"
    ("terapagos_shadow.jpg", "Terapagos"),       # 2.20.0: not in the vocabulary
])
def test_sleeved_field_stills_are_named(name, species):
    d = _identify_payload((SLEEVE / name).read_bytes(), "dev_test", False)
    assert d["name"] == species


# --------------------------------------------------------------------------
# Naming: the vocabulary and where on the card a word sits
# --------------------------------------------------------------------------

def test_vocabulary_is_the_national_dex():
    v = load_species()
    names = {s.name for s in v}
    assert len(v) == 1025
    assert {"Meganium", "Terapagos", "Pecharunt", "Mr. Mime", "Farfetch'd"} <= names


def test_species_key_drops_punctuation_and_accents():
    assert species_key("Mr. Mime") == "mrmime"
    assert species_key("Farfetch'd") == "farfetchd"
    assert species_key("Flabébé") == "flabebe"
    assert species_key("Porygon-Z") == "porygonz"


VOCAB = (Species("Meganium", 154), Species("Seaking", 119), Species("Bayleef", 153),
         Species("Mr. Mime", 122), Species("Terapagos", 1024))


def test_a_flavour_text_word_one_letter_off_is_not_a_name():
    # the Meganium still: the name unread, "soaking" in the flavour text
    words = [Word("Anyone", 0.87, 0.01, (1,)), Word("soaking", 0.90, 0.01, (2,))]
    assert match_words(words, VOCAB)[0] is None


def test_the_name_bar_beats_the_rest_of_the_card():
    words = [Word("Meganiun", 0.05, 0.03, (1,)), Word("Seaking", 0.6, 0.01, (5,))]
    sp, token, edits, _ = match_words(words, VOCAB)
    assert sp.name == "Meganium" and edits == 1


def test_evolves_from_names_the_previous_stage_not_the_card():
    words = [Word("Evolves", 0.09, 0.01, (2,)), Word("from", 0.09, 0.01, (2,)),
             Word("Bayleef", 0.09, 0.01, (2,))]
    assert match_words(words, VOCAB)[0] is None


def test_a_name_in_two_words_is_read():
    assert match_species(["Mr.", "Mime"], VOCAB)[0].name == "Mr. Mime"
    words = [Word("Mr.", 0.05, 0.03, (1,)), Word("Mime", 0.05, 0.03, (1,))]
    assert match_words(words, VOCAB)[0].name == "Mr. Mime"


def test_name_band_covers_a_margin():
    # a caller that rectified with a margin says where the name bar is
    words = [Word("Meganiun", 0.20, 0.03, (1,))]
    assert match_words(words, VOCAB)[0] is None
    assert match_words(words, VOCAB, name_band=(0.0, 0.24))[0].name == "Meganium"
    assert NAME_BAND < 0.2
