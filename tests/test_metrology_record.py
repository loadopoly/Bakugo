"""The standard metrology record (cardcenter/schemas/metrology_record.v1.schema.json)."""

import json
import os
import tempfile
import urllib.request
from http.server import HTTPServer
from threading import Thread
from types import SimpleNamespace

import cv2
import pytest

jsonschema = pytest.importorskip("jsonschema")

from cardcenter import metrology_record as mr
from cardcenter.serve import Handler
from cardcenter.synth import render_capture


@pytest.fixture(scope="module")
def server():
    db = tempfile.mktemp(suffix=".db")
    os.environ["CARDCENTER_DB"] = db
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def _measure(server, holder):
    img, _, _ = render_capture(left_mm=2.5, right_mm=3.5, top_mm=3.0, bottom_mm=3.0)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    b = "----RecordBoundary"
    body = b"".join([
        f'--{b}\r\nContent-Disposition: form-data; name="holder"\r\n\r\n{holder}\r\n'.encode(),
        f'--{b}\r\nContent-Disposition: form-data; name="lens"\r\n\r\nmain\r\n'.encode(),
        f'--{b}\r\nContent-Disposition: form-data; name="image"; filename="c.jpg"\r\nContent-Type: image/jpeg\r\n\r\n'.encode(),
        buf.tobytes(), f"\r\n--{b}--\r\n".encode()])
    req = urllib.request.Request(f"{server}/measure", data=body, method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={b}"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def test_the_schema_is_a_valid_json_schema():
    jsonschema.Draft202012Validator.check_schema(mr.schema())


@pytest.mark.parametrize("holder", ["raw", "toploader"])
def test_measure_returns_a_record_that_validates_and_keeps_the_uncertainties(server, holder):
    data = _measure(server, holder)
    assert data["ok"] is True
    rec = data["record"]
    jsonschema.Draft202012Validator(mr.schema()).validate(rec)
    assert rec["schema"] == "bakugo.metrology_record/1" and rec["record_id"].startswith("mr_")
    assert abs(rec["centering"]["worst"]["ratio"]["value"] - data["ratio"]) < 0.06      # the same measurement, unrounded
    assert all(rec["centering"]["borders_mm"][s]["sigma"] > 0 for s in ("left", "right", "top", "bottom"))
    assert rec["capture"]["lens"]["distortion"] == {"status": "not_corrected"}               # said, not hidden
    assert rec["grading"]["caveat"].startswith("Estimates from a photo")
    if holder == "raw":
        assert rec["holder"]["layers"] == []
    else:
        (layer,) = rec["holder"]["layers"]
        assert layer["refractive_index"] == {"value": 1.58, "sigma": 0.03} and layer["source"] == "preset"
    # the same photo measured by the same release has the same id
    assert _measure(server, holder)["record"]["record_id"] == rec["record_id"]


def test_the_schema_is_served(server):
    with urllib.request.urlopen(f"{server}/schema/metrology-record") as r:
        assert r.headers["Content-Type"].startswith("application/schema+json")
        assert json.loads(r.read())["title"].startswith("Bakugo metrology record")


def _corner(**kw):
    base = dict(resolvable=True, whitening_resolvable=True, whitening_mm=0.0, whitening_len_mm=0.0, loss_mm=0.0, sigma_mm=0.02)
    base.update(kw)
    return SimpleNamespace(**base)


def _edge(**kw):
    base = dict(resolvable=True, whitening_resolvable=True, whitening_frac=0.0, nick_count=0, nick_max_mm=0.0, noise_mm=0.01)
    base.update(kw)
    return SimpleNamespace(**base)


def test_only_resolved_wear_becomes_a_defect_with_its_magnitude():
    view = SimpleNamespace(
        corners={"top_left": _corner(whitening_mm=0.12, whitening_len_mm=0.8),
                 "top_right": _corner(resolvable=False, whitening_mm=0.5),          # glare: says nothing
                 "bottom_right": _corner(whitening_resolvable=False, whitening_mm=0.3),
                 "bottom_left": _corner(loss_mm=0.2)},
        edges={"left": _edge(whitening_frac=0.25), "top": _edge(nick_count=2, nick_max_mm=0.15), "right": _edge()})
    d = {x["defect_id"]: x for x in mr._defects(view, "t")}
    assert set(d) == {"corner_tl:whitening", "corner_bl:corner_loss", "edge_left:whitening", "edge_top:nick"}
    assert d["corner_tl:whitening"]["magnitude"]["depth_mm"] == {"value": 0.12, "sigma": 0.02}
    assert d["edge_left:whitening"]["magnitude"]["length_mm"]["value"] == pytest.approx(0.25 * 88.9, abs=1e-3)
    assert all(x["label"]["source"] == "model" for x in d.values())


# ---------------------------------------------------------------------------- edge probe

import numpy as np

from cardcenter.centering import measure_centering
from cardcenter.edge_probe import probe_edges
from cardcenter.framing import frame_card_for_measure
from cardcenter.types import resolve_holder


def _framed(img):
    fr = frame_card_for_measure(img, fov_deg=68.0, max_side=2400)
    res = measure_centering(fr.image, slab=resolve_holder("raw"), capture=fr.capture, card_quad=fr.quad,
                            quad_residual_px=fr.residual_px)
    return fr.image, res.corners_px


def _radial(img, k1):
    """Radial distortion about the image centre (k1 > 0 bows edges outward in this probe's convention)."""
    h, w = img.shape[:2]
    cx, cy, f = w / 2, h / 2, max(w, h)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    x, y = (xx - cx) / f, (yy - cy) / f
    s = 1 + k1 * (x * x + y * y)
    return cv2.remap(img, (x * s * f + cx).astype(np.float32), (y * s * f + cy).astype(np.float32), cv2.INTER_LINEAR)


def _red_scaled(img, scale):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, scale)
    out = img.copy()
    out[..., 2] = cv2.warpAffine(img[..., 2], M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out


@pytest.fixture(scope="module")
def card():
    img, _, _ = render_capture(left_mm=2.5, right_mm=3.5, top_mm=3.0, bottom_mm=3.0)
    return img


def test_a_flat_card_has_straight_edges_and_no_colour_split(card):
    p = mr._edge_probe(probe_edges(*_framed(card)))
    assert p["bow"] == "none" and p["bowed"] == []
    for v in p["sides"].values():
        assert v["points"] >= 30 and abs(v["sagitta_mm"]["value"]) < 0.03
        assert abs(v["red_minus_green_px"]) < 0.1 and abs(v["blue_minus_green_px"]) < 0.1


@pytest.mark.parametrize("k1, pattern", [(0.08, "all_outward"), (-0.08, "all_inward")])
def test_radial_lens_distortion_bows_every_edge_the_same_way(card, k1, pattern):
    p = mr._edge_probe(probe_edges(*_framed(_radial(card, k1))))
    assert p["bow"] == pattern
    assert all(abs(v["sagitta_mm"]["value"]) > 0.08 for v in p["sides"].values())


def test_a_red_channel_scaled_outward_shows_as_a_positive_red_minus_green(card):
    small = probe_edges(*_framed(_red_scaled(card, 1.0015)))
    large = probe_edges(*_framed(_red_scaled(card, 1.003)))
    for s in ("top", "right", "bottom", "left"):
        assert 0.25 < small[s]["red_minus_green_px"] < large[s]["red_minus_green_px"]
        assert abs(large[s]["blue_minus_green_px"]) < 0.1


def _side(sag, sig=0.01):
    return {"points": 40, "sagitta_mm": sag, "sagitta_sigma_mm": sig, "line_rms_mm": 0.02,
            "red_minus_green_px": 0.0, "blue_minus_green_px": 0.0}


@pytest.mark.parametrize("sags, pattern", [
    ({"top": 0.0, "right": 0.01, "bottom": -0.01, "left": 0.0}, "none"),
    ({"top": 0.0, "right": 0.12, "bottom": 0.0, "left": 0.11}, "left_right"),       # face curved across the width
    ({"top": -0.2, "right": 0.0, "bottom": -0.15, "left": 0.0}, "top_bottom"),
    ({"top": 0.1, "right": 0.12, "bottom": 0.1, "left": 0.11}, "all_outward"),
    ({"top": 0.1, "right": -0.12, "bottom": 0.0, "left": 0.0}, "mixed"),
    ({"top": 0.2, "right": 0.0, "bottom": 0.0, "left": 0.0}, "mixed"),
])
def test_the_bow_pattern_names_lens_or_holder(sags, pattern):
    assert mr._edge_probe({s: _side(v) for s, v in sags.items()})["bow"] == pattern


def test_a_bow_inside_its_own_noise_does_not_count():
    assert mr._edge_probe({s: _side(0.1, sig=0.05) for s in ("top", "right", "bottom", "left")})["bow"] == "none"


# ---------------------------------------------------------------------------- the same card, bare and in holders

def test_measure_keeps_the_card_ref_and_stores_the_record(server):
    img, _, _ = render_capture(left_mm=2.5, right_mm=3.5, top_mm=3.0, bottom_mm=3.0, seed=7)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    b = "----RefBoundary"
    body = b"".join([
        f'--{b}\r\nContent-Disposition: form-data; name="card_ref"\r\n\r\nprobopass-1\r\n'.encode(),
        f'--{b}\r\nContent-Disposition: form-data; name="image"; filename="c.jpg"\r\nContent-Type: image/jpeg\r\n\r\n'.encode(),
        buf.tobytes(), f"\r\n--{b}--\r\n".encode()])
    req = urllib.request.Request(f"{server}/measure", data=body, method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={b}"})
    with urllib.request.urlopen(req) as r:
        data = json.loads(r.read())
    rec = data["record"]
    jsonschema.Draft202012Validator(mr.schema()).validate(rec)
    assert rec["subject"]["card_ref"] == "probopass-1"
    assert rec["provenance"]["device_id"] == data["device_id"] and rec["provenance"]["scan_id"] == data["scan_id"]
    assert rec["edge_probe"]["bow"] == "none"
    stored = mr.load_records(os.environ["CARDCENTER_DB"])
    assert any(s["record_id"] == rec["record_id"] for s in stored)


def test_a_card_ref_that_is_not_a_plain_label_is_dropped():
    assert mr.clean_card_ref("probopass-1") == "probopass-1"
    assert mr.clean_card_ref("a b") is None and mr.clean_card_ref("x" * 65) is None and mr.clean_card_ref(None) is None


def _rec(card_ref, holder, borders, sigma=0.02, device="dev_a", ratio=None):
    l, r, t, btm = borders
    ratio = ratio if ratio is not None else 100 * max(l, r) / (l + r)
    m = lambda v: {"value": v, "sigma": sigma}
    return {"subject": {"card_ref": card_ref, "face": "front"}, "provenance": {"device_id": device},
            "holder": {"preset": holder},
            "centering": {"borders_mm": {"left": m(l), "right": m(r), "top": m(t), "bottom": m(btm)},
                          "worst": {"ratio": {"value": ratio, "sigma": 0.5}}}}


def test_holder_study_finds_the_bias_and_the_scatter_beyond_the_sigmas():
    recs = []
    rng = np.random.default_rng(3)
    for card in ("c1", "c2", "c3"):
        recs.append(_rec(card, "raw", (2.5, 3.5, 3.0, 3.0)))
        for _ in range(4):        # in a toploader: left reads 0.05 mm wide, with 0.04 mm of scatter the sigmas do not explain
            recs.append(_rec(card, "toploader", (2.55 + rng.normal(0, 0.04), 3.5, 3.0, 3.0)))
    recs.append(_rec("c9", "psa", (2.5, 3.5, 3.0, 3.0)))        # no bare measurement of c9: not paired
    recs.append(_rec("c1", "toploader", (9.0, 3.5, 3.0, 3.0), device="dev_b"))   # another device's c1 is another card
    out = mr.holder_study(recs)
    assert set(out) == {"toploader"}
    t = out["toploader"]
    assert t["cards"] == 3 and t["measurements"] == 12
    assert t["bias_mm"]["left"] == pytest.approx(0.05, abs=0.03) and t["bias_mm"]["right"] == 0.0
    assert 0.0 < t["extra_sigma_mm"]["left"] < 0.08 and t["extra_sigma_mm"]["right"] == 0.0
    assert t["ratio_bias_pp"] < 0                                   # a wider left border pulls the ratio toward 50
