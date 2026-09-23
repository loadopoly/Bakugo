"""In-situ learning: identification records, the second vote, feedback,
per-device priors, owner inbox, and the service-worker static routes."""

import json
import os
import tempfile
import urllib.error
import urllib.request
from http.server import HTTPServer
from pathlib import Path
from threading import Thread

import cv2
import pytest

from cardcenter import insitu
from cardcenter.confidence import Decision, IdentificationVote, gate_identification
from cardcenter.insitu import FeedbackError, InSituStore, write_inbox
from cardcenter.serve import Handler, device_id_for_token
from cardcenter.synth import render_capture

OWNER = "owner-token-" + "x" * 20


# ------------------------------------------------------------------ store

@pytest.fixture()
def store(tmp_path):
    s = InSituStore(str(tmp_path / "app.db"))
    yield s
    s.close()


def test_feedback_is_scoped_to_the_device(store):
    ident = store.record_identification("dev-a", {"name": "Lechonk", "dex": 915})
    assert store.get_identification(ident, "dev-a")["name"] == "Lechonk"
    assert store.get_identification(ident, "dev-b") is None
    with pytest.raises(FeedbackError):
        store.add_feedback("dev-b", "confirm", identification_id=ident)


def test_confirm_and_correct_rules(store):
    ident = store.record_identification("d", {"name": "Lechonk", "dex": 915})
    fb = store.add_feedback("d", "confirm", identification_id=ident)
    assert fb["name"] == "Lechonk" and fb["number"] == 915
    with pytest.raises(FeedbackError):
        store.add_feedback("d", "confirm", identification_id=ident, name="Pikachu")
    with pytest.raises(FeedbackError):
        store.add_feedback("d", "correct", identification_id=ident)
    unnamed = store.record_identification("d", {"name": None})
    with pytest.raises(FeedbackError):
        store.add_feedback("d", "confirm", identification_id=unnamed)
    with pytest.raises(FeedbackError):
        store.add_feedback("d", "correct", identification_id=unnamed, name="x\nDROP")
    with pytest.raises(FeedbackError):
        store.add_feedback("d", "delete")
    r = store.add_feedback("d", "reject", identification_id=unnamed)
    assert r["learned"] == {}


def test_device_prior_breaks_ties_only_with_clear_history(store):
    alts = ["Lechonk", "Lokix"]
    assert store.break_tie("d", alts) is None
    for _ in range(2):
        store.add_feedback("d", "correct", name="Lechonk")
    assert store.break_tie("d", alts) is None                     # 2 < TIE_MIN_COUNT
    store.add_feedback("d", "correct", name="Lechonk")
    name, p = store.break_tie("d", alts)
    assert name == "Lechonk" and 0.5 < p < 1.0
    store.add_feedback("d", "correct", name="Lokix")
    assert store.break_tie("d", alts) is None                     # 4:2 odds < 3:1
    assert store.break_tie("other-device", alts) is None           # not shared


def test_write_inbox(tmp_path, monkeypatch):
    side = write_inbox({"id": "abc", "name": "Lechonk"}, {"name": "Lechonk", "action": "confirm"},
                       {"amount": "2"}, b"\xff\xd8\xff\xe0jpeg", "dev", inbox=str(tmp_path))
    body = json.loads(side.read_text())
    assert body["confirmed_by"] == "owner-device:dev" and body["image"] == "abc.jpg"
    assert (side.parent / "abc.jpg").read_bytes().startswith(b"\xff\xd8")
    monkeypatch.delenv("CARDCENTER_INSITU_INBOX", raising=False)
    assert write_inbox({"id": "x"}, {}, None, b"", "d", inbox="") is None


def test_owner_token_needs_length_and_match(monkeypatch):
    monkeypatch.setenv("CARDCENTER_OWNER_TOKEN", "short")
    assert not insitu.owner_token_matches("short")
    monkeypatch.setenv("CARDCENTER_OWNER_TOKEN", OWNER)
    assert insitu.owner_token_matches(OWNER)
    assert not insitu.owner_token_matches(OWNER + "y")
    assert not insitu.owner_token_matches(None)


# ------------------------------------------------------------------- gate

def test_identification_gate():
    agree = IdentificationVote(True, "Lechonk", 0.93, 0.2, 5, "m")
    assert gate_identification("Lechonk", True, agree, 1.1).decision is Decision.ACCEPT
    assert gate_identification("lechonk", False, agree, 1.1).decision is Decision.ACCEPT
    assert gate_identification("Lechonk", True, agree, 2.5).decision is Decision.REVIEW
    assert gate_identification("Lechonk", True, agree, 1.1, edges_resolvable=False).decision is Decision.REVIEW
    assert gate_identification(None, False, agree, 1.1).decision is Decision.REVIEW
    assert gate_identification("Pikachu", True, agree, 1.1).decision is Decision.REVIEW
    weak = IdentificationVote(True, "Lechonk", 0.6, 0.2, 5, "m")
    assert gate_identification("Lechonk", True, weak, 1.1).decision is Decision.REVIEW
    small = IdentificationVote(True, "Lechonk", 0.99, 0.5, 1, "m")
    assert gate_identification("Lechonk", True, small, 1.1).decision is Decision.REVIEW
    off = IdentificationVote(False, reason="no model")
    r = gate_identification("Lechonk", True, off, 1.1)
    assert r.decision is Decision.REVIEW and "no model" in r.reason


# ----------------------------------------------------------------- server

@pytest.fixture()
def server(tmp_path, monkeypatch):
    db = tmp_path / "cardcenter.db"
    monkeypatch.setenv("CARDCENTER_DB", str(db))
    monkeypatch.setenv("CARDCENTER_OWNER_TOKEN", OWNER)
    monkeypatch.setenv("CARDCENTER_INSITU_INBOX", str(tmp_path / "inbox"))
    monkeypatch.delenv("CARDCENTER_EMBED_DIR", raising=False)
    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", db, tmp_path
    httpd.shutdown()
    httpd.server_close()


def _req(url, data=None, headers=None, method=None):
    body = json.dumps(data).encode() if isinstance(data, dict) else data
    h = dict(headers or {})
    if isinstance(data, dict):
        h["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, headers=h, method=method)
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, r.headers, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers, e.read()


def _token(base):
    _, headers, _ = _req(f"{base}/config")
    return headers["Set-Cookie"].split(";", 1)[0].split("=", 1)[1]


def _multipart(image_bytes, boundary="----B0undary"):
    return (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"s.jpg\"\r\n"
            f"Content-Type: image/jpeg\r\n\r\n").encode() + image_bytes + \
        f"\r\n--{boundary}--\r\n".encode(), f"multipart/form-data; boundary={boundary}"


def test_identify_records_and_reports_information(server):
    base, db, _ = server
    tok = _token(base)
    img, _, _ = render_capture(seed=2)
    ok, jpg = cv2.imencode(".jpg", img)
    body, ctype = _multipart(jpg.tobytes())
    status, _, raw = _req(f"{base}/identify", body,
                          {"Content-Type": ctype, "Authorization": f"Bearer {tok}"})
    d = json.loads(raw)
    assert status == 200 and d["ok"], d
    assert d["identification_id"] and d["decision"] == "review"
    assert d["information"]["resolvable"] is True and len(d["quad"]) == 4
    assert "embed_crop" not in d                                   # no model installed
    with InSituStore(str(db)) as st:
        rec = st.get_identification(d["identification_id"], device_id_for_token(tok))
        assert rec is not None and rec["resolvable"] == 1


def test_vote_and_feedback_endpoints(server):
    base, db, tmp = server
    tok = _token(base)
    dev = device_id_for_token(tok)
    auth = {"Authorization": f"Bearer {tok}"}
    with InSituStore(str(db)) as st:
        ident = st.record_identification(dev, {"name": "Lechonk", "dex": 915, "corroborated": True,
                                               "shot_ratio": 1.2, "resolvable": True})
    vote = {"available": True, "name": "Lechonk", "similarity": 0.95, "margin": 0.3,
            "index_size": 4, "model_id": "m"}
    s, _, raw = _req(f"{base}/identify/vote", {"identification_id": ident, "vote": vote}, auth)
    assert json.loads(raw)["decision"] == "accept"
    # another device cannot vote on it or give feedback on it
    other = {"Authorization": f"Bearer {_token(base)}"}
    s, _, raw = _req(f"{base}/identify/vote", {"identification_id": ident, "vote": vote}, other)
    assert json.loads(raw)["ok"] is False
    # feedback with a price, no owner token -> no inbox
    s, _, raw = _req(f"{base}/feedback", {"identification_id": ident, "action": "confirm",
                                          "price": {"text": "3 for $1", "source": "shelf_sign",
                                                    "scope": "box", "scope_key": "bin 4"}}, auth)
    r = json.loads(raw)
    assert r["ok"] and r["learned"] == {"device_prior": "Lechonk"} and r["inbox"] is False
    assert r["price"]["quantity"] == 3 and r["price"]["scope"] == "box"
    s, _, raw = _req(f"{base}/feedback", {"identification_id": ident, "action": "confirm",
                                          "price": {"text": "cheap"}}, auth)
    assert json.loads(raw)["ok"] is False
    s, _, raw = _req(f"{base}/feedback", b"[1]", {**auth, "Content-Type": "application/json"})
    assert json.loads(raw)["ok"] is False
    assert not (tmp / "inbox").exists()


def test_owner_confirmation_reaches_the_private_inbox(server):
    base, db, tmp = server
    tok = _token(base)
    img, _, _ = render_capture(seed=3)
    _, jpg = cv2.imencode(".jpg", img)
    body, ctype = _multipart(jpg.tobytes())
    owner = {"Authorization": f"Bearer {tok}", "X-Bakugo-Owner": OWNER}
    _, _, raw = _req(f"{base}/identify", body, {**owner, "Content-Type": ctype})
    ident = json.loads(raw)["identification_id"]
    _, _, raw = _req(f"{base}/feedback", {"identification_id": ident, "action": "correct",
                                          "name": "Lechonk", "number": 182, "franchise": "pokemon",
                                          "price": {"amount": "2", "source": "verbal", "kind": "paid"}},
                     owner)
    r = json.loads(raw)
    assert r["ok"] and r["inbox"] is True
    sides = list((tmp / "inbox").rglob("*.json"))
    assert len(sides) == 1
    body = json.loads(sides[0].read_text())
    assert body["label"]["name"] == "Lechonk" and body["price"]["kind"] == "paid"
    assert (sides[0].parent / body["image"]).read_bytes() == jpg.tobytes()


def test_wrong_owner_token_keeps_photo_out_of_inbox(server):
    base, _, tmp = server
    tok = _token(base)
    img, _, _ = render_capture(seed=4)
    _, jpg = cv2.imencode(".jpg", img)
    body, ctype = _multipart(jpg.tobytes())
    bad = {"Authorization": f"Bearer {tok}", "X-Bakugo-Owner": OWNER + "z"}
    _, _, raw = _req(f"{base}/identify", body, {**bad, "Content-Type": ctype})
    ident = json.loads(raw)["identification_id"]
    _, _, raw = _req(f"{base}/feedback", {"identification_id": ident, "action": "correct",
                                          "name": "Lechonk"}, bad)
    assert json.loads(raw)["inbox"] is False
    assert not (tmp / "inbox").exists()


def test_service_worker_and_embed_routes(server, tmp_path, monkeypatch):
    base, _, _ = server
    s, h, raw = _req(f"{base}/sw.js")
    assert s == 200 and h["Service-Worker-Allowed"] == "/"
    assert h["Content-Type"].startswith("text/javascript")
    js = raw.decode()
    assert "const ort = null;" in js and '"available": false' in js
    s, _, raw = _req(f"{base}/embed/status")
    assert json.loads(raw)["available"] is False
    assert _req(f"{base}/embed/model.onnx")[0] == 404

    emb = tmp_path / "embed"
    (emb / "ort").mkdir(parents=True)
    (emb / "model.json").write_text(json.dumps({"model_id": "m1", "input_size": [32, 32]}))
    (emb / "model.onnx").write_bytes(b"onnx")
    (emb / "ort" / "ort.wasm.bundle.min.mjs").write_text("export {}")
    (emb / "ort" / "ort-wasm-simd-threaded.wasm").write_bytes(b"\0asm")
    (emb / "secret.txt").write_text("no")
    monkeypatch.setenv("CARDCENTER_EMBED_DIR", str(emb))
    js = _req(f"{base}/sw.js")[2].decode()
    assert "import * as ort from '/embed/ort/ort.wasm.bundle.min.mjs';" in js
    s, h, raw = _req(f"{base}/embed/ort/ort-wasm-simd-threaded.wasm")
    assert s == 200 and h["Content-Type"] == "application/wasm"
    for bad in ("secret.txt", "../model.json", "ort/../../secret.txt", "model.json/"):
        assert _req(f"{base}/embed/{bad}")[0] == 404
    (emb / "model.json").write_text(json.dumps({"model_id": "bad id!", "input_size": [32, 32]}))
    st = json.loads(_req(f"{base}/embed/status")[2])
    assert st["available"] is False and "model_id" in st["reason"]


def test_identify_includes_embed_crop_when_model_installed(server, tmp_path, monkeypatch):
    base, _, _ = server
    emb = tmp_path / "embed2"
    (emb / "ort").mkdir(parents=True)
    (emb / "model.json").write_text(json.dumps({"model_id": "m1", "input_size": [48, 40]}))
    for f in ("model.onnx", "ort/ort.wasm.bundle.min.mjs", "ort/ort-wasm-simd-threaded.wasm"):
        (emb / f).write_bytes(b"x")
    monkeypatch.setenv("CARDCENTER_EMBED_DIR", str(emb))
    img, _, _ = render_capture(seed=5)
    _, jpg = cv2.imencode(".jpg", img)
    body, ctype = _multipart(jpg.tobytes())
    _, _, raw = _req(f"{base}/identify", body, {"Content-Type": ctype})
    d = json.loads(raw)
    import base64

    import numpy as np

    crop = cv2.imdecode(np.frombuffer(base64.b64decode(d["embed_crop"]), np.uint8), cv2.IMREAD_COLOR)
    assert crop.shape[:2] == (48, 40)
