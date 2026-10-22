"""2.33.0: the phone's tracked outline as a locate hint; auto-capture page logic."""
import json

import cv2
import numpy as np

from cardcenter import serve
from cardcenter.framing import _locate, locate_from_hint


def _card_photo():
    img = np.full((1200, 900, 3), (196, 186, 170), np.uint8)
    q = np.array([[220, 260], [680, 280], [670, 920], [200, 900]], np.int32)
    cv2.fillConvexPoly(img, q, (60, 60, 160))
    inner = np.array([[245, 290], [655, 308], [646, 893], [226, 874]], np.int32)
    cv2.fillConvexPoly(img, inner, (40, 140, 220))
    return img, q.astype(np.float64)


def test_hint_matches_full_search():
    img, _ = _card_photo()
    full = _locate(img)
    assert full is not None
    hint = full[0] + np.array([[8, -6], [-5, 7], [6, 4], [-7, -5]], float)
    fast = locate_from_hint(img, hint)
    assert fast is not None
    assert np.abs(fast[0] - full[0]).max() < 3.0


def test_bad_hint_falls_back():
    img, _ = _card_photo()
    assert locate_from_hint(img, [[0, 0], [10, 0], [10, 10], [0, 10]]) is None     # too small
    assert locate_from_hint(img, "nonsense") is None
    far = np.array([[700, 1000], [880, 1000], [880, 1190], [700, 1190]], float)  # no card there
    r = locate_from_hint(img, far)
    assert r is None or np.abs(r[0] - far).max() < 200


def test_quad_hint_field_parsing():
    img, q = _card_photo()
    ok, buf = cv2.imencode(".png", img)
    b = buf.tobytes()
    assert serve._quad_hint({"quad_hint": json.dumps(q.tolist()).encode()}, b) is not None
    assert serve._quad_hint({"quad_hint": b"[[1,2],[3,4]]"}, b) is None
    assert serve._quad_hint({"quad_hint": json.dumps((q + 5000).tolist()).encode()}, b) is None
    assert serve._quad_hint({}, b) is None


def test_page_has_fast_capture():
    page = serve.PAGE
    assert "function maybeAutoCapture" in page
    assert "Photo blurred -- not sent" in page
    assert "quad_hint" in page
    assert ".hud-spirit{position:absolute;bottom:10px;left:10px;" in page
