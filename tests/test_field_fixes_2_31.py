"""2.31.0 fixes from the 2026-10-15 field screenshots."""
from cardcenter.ar import _detection_guidance
from cardcenter.detect import DetectionError
from cardcenter import serve


def test_border_failure_names_the_sides():
    exc = DetectionError("border detection failed:\n  - side 'top': no edge\n  - side 'left': weak")
    text = _detection_guidance(exc)
    assert "top, left sides" in text
    assert not text.rstrip().endswith(":")


def test_single_line_error_passes_through():
    assert _detection_guidance(DetectionError("empty colour-distance field")) == "empty colour-distance field"


def test_page_has_no_all_caps_status_defaults():
    page = serve.PAGE
    assert 'id="hud-status">Searching<' in page
    assert "SPRT IDLE" not in page
    assert "viewport-fit=cover" not in page
    # zoom row sits inside the fixed dock, so the buttons cannot cover it
    dock = page.index('<div class="native-dock">')
    assert page.index('id="zoom-bar"') > dock
