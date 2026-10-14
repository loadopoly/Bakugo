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
