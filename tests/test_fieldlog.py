"""The server keeps the exact inputs it measured, so a field miss can be
replayed instead of rebuilt from a screenshot."""

from __future__ import annotations

import json
import urllib.request

import cv2
import numpy as np

from cardcenter import fieldlog
from cardcenter.synth import render_capture


def test_record_keeps_the_newest(tmp_path, monkeypatch):
    monkeypatch.setenv("CARDCENTER_FIELD_DIR", str(tmp_path))
    monkeypatch.setenv("CARDCENTER_FIELD_LOG", "3")
    for i in range(5):
        assert fieldlog.record("measure", b"\xff\xd8jpeg%d" % i, {"i": i}) is not None
    metas = sorted(tmp_path.glob("*.json"))
    assert len(metas) == 3 and len(list(tmp_path.glob("*.jpg"))) == 3
    assert [json.loads(p.read_text())["i"] for p in metas] == [2, 3, 4]


def test_off_means_off(tmp_path, monkeypatch):
    monkeypatch.setenv("CARDCENTER_FIELD_DIR", str(tmp_path))
    monkeypatch.setenv("CARDCENTER_FIELD_LOG", "0")
    assert fieldlog.record("measure", b"x", {}) is None
    assert not list(tmp_path.iterdir())


def test_measure_and_identify_keep_the_still_and_the_outcome(tmp_path, monkeypatch):
    from test_serve_api import _multipart, _new_token, _open, test_server  # noqa: F401
    from http.server import HTTPServer
    from threading import Thread
    from cardcenter.serve import Handler

    monkeypatch.setenv("CARDCENTER_FIELD_DIR", str(tmp_path))
    monkeypatch.setenv("CARDCENTER_FIELD_LOG", "50")
    monkeypatch.setenv("CARDCENTER_DB", str(tmp_path / "t.db"))
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        img, _, _ = render_capture(left_mm=3.4, right_mm=2.6, image_size=(900, 1200))
        jpg = cv2.imencode(".jpg", img)[1].tobytes()
        token = _new_token(base)
        body, ctype = _multipart({"holder": "raw", "lens": "main"}, jpg)
        for path in ("/measure", "/identify"):
            req = urllib.request.Request(base + path, data=body, method="POST",
                                         headers={"Content-Type": ctype,
                                                  "Cookie": f"bakugo_device={token}"})
            status, _, _ = _open(req)
            assert status == 200
        blank = cv2.imencode(".jpg", np.full((600, 450, 3), 128, np.uint8))[1].tobytes()
        body, ctype = _multipart({"holder": "raw"}, blank)
        req = urllib.request.Request(base + "/measure", data=body, method="POST",
                                     headers={"Content-Type": ctype,
                                              "Cookie": f"bakugo_device={token}"})
        _open(req)
    finally:
        srv.shutdown()
        srv.server_close()
    metas = [json.loads(p.read_text()) for p in sorted(tmp_path.glob("*.json"))]
    kinds = [m["kind"] for m in metas]
    assert kinds.count("measure") == 2 and kinds.count("identify") == 1
    ok = [m for m in metas if m["kind"] == "measure" and m["outcome"].get("ok")]
    bad = [m for m in metas if m["kind"] == "measure" and not m["outcome"].get("ok")]
    assert ok and ok[0]["outcome"]["ratio"] and bad and bad[0]["outcome"]["error"]
    # the image kept is the one that was sent, byte for byte
    first = sorted(tmp_path.glob("*_measure.jpg"))[0]
    assert first.read_bytes() == jpg
