"""2.32.0 fixes from the 2026-10-21 field screenshots (Fletchinder)."""
from cardcenter import serve
from cardcenter.recognise import Species, Word, prefix_candidates, NAME_BAND

VOCAB = [Species("Fletchling", 661), Species("Fletchinder", 662), Species("Talonflame", 663)]


def _band(text):
    return Word(text, 0.0, 0.0, ("band", 0))


def test_prefix_names_both_candidates():
    tok, names = prefix_candidates([_band("Fletch")], VOCAB, (0.0, NAME_BAND))
    assert tok == "Fletch"
    assert set(names) == {"Fletchling", "Fletchinder"}


def test_short_or_exact_tokens_give_no_prefix():
    assert prefix_candidates([_band("Flet")], VOCAB) is None          # under 5 letters
    assert prefix_candidates([_band("Talonflame")], VOCAB) is None    # a full name is not a prefix


def test_word_outside_name_band_ignored():
    w = Word("Fletch", 0.9, 0.02, (1,))
    assert prefix_candidates([w], VOCAB, (0.0, NAME_BAND)) is None


def test_page_sizes_detail_to_uplink():
    page = serve.PAGE
    assert "function detailBudgetBytes" in page
    assert "Date.now() - lastDetailAt < detailEveryMs" in page
    assert "Ray-tracing Snell" not in page
