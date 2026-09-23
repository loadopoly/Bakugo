"""Tests for cardcenter.serve HTTP & AR API, CORS, tenant identity and request limits."""

import io
import json
import os
import socket
import tempfile
import urllib.error
import urllib.request
from http.server import HTTPServer
from threading import Thread

import cv2
import numpy as np
import pytest

from cardcenter import __version__
from cardcenter import serve as serve_mod
from cardcenter.serve import Handler, device_id_for_token
from cardcenter.synth import render_capture


@pytest.fixture(scope="module")
def test_server():
    """Spin up a real test HTTP server on an ephemeral port."""
    tmp_db = tempfile.mktemp(suffix=".db")
    os.environ["CARDCENTER_DB"] = tmp_db

    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    base_url = f"http://127.0.0.1:{port}"

    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    yield base_url

    server.shutdown()
    server.server_close()
    if os.path.exists(tmp_db):
        try:
            os.remove(tmp_db)
        except OSError:
            pass


def _open(req):
    """urlopen that returns (status, headers, body) for error statuses too."""
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, resp.headers, resp.read()
    except urllib.error.HTTPError as err:
        return err.code, err.headers, err.read()


def _new_token(base_url):
    """Ask the server for a device token the way a browser would (cookie)."""
    status, headers, body = _open(urllib.request.Request(f"{base_url}/config"))
    assert status == 200
    cookie = headers.get("Set-Cookie", "")
    assert cookie.startswith("bakugo_device=")
    assert "HttpOnly" in cookie and "SameSite=Lax" in cookie
    token = cookie.split(";", 1)[0].split("=", 1)[1]
    assert json.loads(body)["device_id"] == device_id_for_token(token)
    return token


def _multipart(fields, image_bytes, boundary="----TestBoundary12345678"):
    body = io.BytesIO()
    for name, value in fields.items():
        body.write(f"--{boundary}\r\n".encode())
        body.write(f'Content-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    body.write(f"--{boundary}\r\n".encode())
    body.write(b'Content-Disposition: form-data; name="image"; filename="shot.jpg"\r\n')
    body.write(b"Content-Type: image/jpeg\r\n\r\n")
    body.write(image_bytes)
    body.write(f"\r\n--{boundary}--\r\n".encode())
    return body.getvalue(), f"multipart/form-data; boundary={boundary}"


def _card_jpeg(quality=85):
    img, _, _ = render_capture(left_mm=2.5, right_mm=3.5, top_mm=3.0, bottom_mm=3.0)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    assert ok
    return buf.tobytes()


def test_cors_preflight_allowed_origin(test_server):
    """An allowed origin gets its own origin echoed back, never '*'."""
    req = urllib.request.Request(f"{test_server}/measure", method="OPTIONS")
    req.add_header("Origin", "capacitor://localhost")
    req.add_header("Access-Control-Request-Method", "POST")
    req.add_header("Access-Control-Request-Headers", "Authorization, Content-Type")

    status, headers, _ = _open(req)
    assert status == 204
    assert headers.get("Access-Control-Allow-Origin") == "capacitor://localhost"
    assert headers.get("Access-Control-Allow-Credentials") == "true"
    assert "POST" in headers.get("Access-Control-Allow-Methods", "")
    assert "Authorization" in headers.get("Access-Control-Allow-Headers", "")


def test_cors_rejects_other_origins(test_server):
    for method, path in (("OPTIONS", "/measure"), ("GET", "/config")):
        req = urllib.request.Request(f"{test_server}{path}", method=method)
        req.add_header("Origin", "https://evil.example")
        _, headers, _ = _open(req)
        assert headers.get("Access-Control-Allow-Origin") is None
        assert headers.get("X-Device-Token") is None


def test_cross_origin_client_gets_bearer_token(test_server):
    req = urllib.request.Request(f"{test_server}/config")
    req.add_header("Origin", "https://localhost")
    status, headers, body = _open(req)
    assert status == 200
    token = headers.get("X-Device-Token")
    assert token and token.startswith("bk1_")
    assert json.loads(body)["device_id"] == device_id_for_token(token)

    # The token is accepted back as a Bearer credential and not re-issued.
    req = urllib.request.Request(f"{test_server}/config")
    req.add_header("Origin", "https://localhost")
    req.add_header("Authorization", f"Bearer {token}")
    _, headers, body = _open(req)
    assert headers.get("X-Device-Token") is None
    assert headers.get("Set-Cookie") is None
    assert json.loads(body)["device_id"] == device_id_for_token(token)


def test_same_origin_does_not_expose_token_header(test_server):
    req = urllib.request.Request(f"{test_server}/config")
    req.add_header("Origin", test_server)
    _, headers, _ = _open(req)
    assert headers.get("Set-Cookie")
    assert headers.get("X-Device-Token") is None


def test_health_and_config(test_server):
    """Test /health and /config discovery endpoints."""
    status, headers, body = _open(urllib.request.Request(f"{test_server}/health"))
    assert status == 200
    data = json.loads(body)
    assert data["status"] == "healthy"
    assert data["version"] == __version__
    assert "db" in data
    assert headers.get("X-Content-Type-Options") == "nosniff"

    token = _new_token(test_server)
    req = urllib.request.Request(f"{test_server}/config")
    req.add_header("Cookie", f"bakugo_device={token}")
    status, headers, body = _open(req)
    assert status == 200
    assert headers.get("Set-Cookie") is None
    data = json.loads(body)
    assert data["ok"] is True
    assert data["version"] == __version__
    assert len(data["holders"]) > 0
    assert "PSA" in data["graders"]

    status, _, body = _open(urllib.request.Request(f"{test_server}/manifest.json"))
    assert status == 200
    data = json.loads(body)
    assert data["short_name"] == "Bakugo"
    assert data["display"] == "standalone"


def test_ar_push_streaming_lifecycle(test_server):
    """Test real-time AR frame push and session lifecycle over HTTP."""
    token = _new_token(test_server)

    req = urllib.request.Request(f"{test_server}/ar/reset", method="POST")
    req.add_header("Cookie", f"bakugo_device={token}")
    status, _, body = _open(req)
    assert status == 200
    assert json.loads(body)["ok"] is True

    payload, ctype = _multipart({"holder": "raw", "lens": "main"}, _card_jpeg(80))
    req = urllib.request.Request(f"{test_server}/ar/push", data=payload, method="POST")
    req.add_header("Content-Type", ctype)
    req.add_header("Cookie", f"bakugo_device={token}")
    status, _, body = _open(req)
    assert status == 200
    data = json.loads(body)
    assert data["ok"] is True
    assert "tracking" in data
    assert "headline" in data
    assert data["seen_frames"] >= 1


def test_measure_and_tenant_isolation(test_server):
    """Scans are scoped to the server-issued token, not to client-sent ids."""
    token_a = _new_token(test_server)
    token_b = _new_token(test_server)
    assert token_a != token_b
    device_a = device_id_for_token(token_a)

    payload, ctype = _multipart({"holder": "raw", "lens": "main"}, _card_jpeg())
    req = urllib.request.Request(f"{test_server}/measure", data=payload, method="POST")
    req.add_header("Content-Type", ctype)
    req.add_header("Cookie", f"bakugo_device={token_a}")
    status, _, body = _open(req)
    assert status == 200
    data = json.loads(body)
    assert data["ok"] is True
    assert data["ratio"] > 0
    assert data["overlay"] != ""
    assert data.get("saved_local") is True
    assert data["device_id"] == device_a

    def my_scans(**headers):
        req = urllib.request.Request(f"{test_server}/my-scans?device_id={device_a}")
        for k, v in headers.items():
            req.add_header(k.replace("_", "-"), v)
        status, _, body = _open(req)
        assert status == 200
        return json.loads(body)

    # Owner sees the scan.
    assert len(my_scans(Cookie=f"bakugo_device={token_a}")["scans"]) == 1
    assert len(my_scans(Authorization=f"Bearer {token_a}")["scans"]) == 1

    # Another device sees nothing, even when it names device A's id in the
    # header and the query string.
    got = my_scans(Cookie=f"bakugo_device={token_b}", X_Device_ID=device_a)
    assert got["scans"] == []
    assert got["device_id"] == device_id_for_token(token_b)

    # No credential at all: a fresh identity with no scans.
    got = my_scans(X_Device_ID=device_a)
    assert got["scans"] == []
    assert got["device_id"] != device_a

    # A malformed token is ignored rather than used as an id.
    got = my_scans(Cookie="bakugo_device=anonymous")
    assert got["scans"] == []


def test_trusted_device_header_opt_in(test_server, monkeypatch):
    monkeypatch.setenv("CARDCENTER_TRUST_DEVICE_HEADER", "1")
    req = urllib.request.Request(f"{test_server}/config")
    req.add_header("X-Device-ID", "lan_device_1")
    _, _, body = _open(req)
    assert json.loads(body)["device_id"] == "lan_device_1"


def test_body_size_limit(test_server, monkeypatch):
    monkeypatch.setattr(serve_mod, "MAX_BODY_BYTES", 1024)
    # The server must answer 413 from the header alone, without reading the body.
    host, port = test_server.rsplit("/", 1)[1].split(":")
    with socket.create_connection((host, int(port)), timeout=5) as s:
        s.sendall(
            b"POST /measure HTTP/1.1\r\nHost: x\r\nContent-Type: image/jpeg\r\n"
            b"Content-Length: 999999999\r\n\r\n"
        )
        reply = s.recv(4096).decode("latin-1")
    assert reply.startswith("HTTP/1.0 413")


def test_bad_content_length(test_server):
    host, port = test_server.rsplit("/", 1)[1].split(":")
    for value in (b"-5", b"abc"):
        with socket.create_connection((host, int(port)), timeout=5) as s:
            s.sendall(b"POST /measure HTTP/1.1\r\nHost: x\r\nContent-Length: " + value + b"\r\n\r\n")
            assert s.recv(4096).decode("latin-1").startswith("HTTP/1.0 400")


def test_image_pixel_limit(test_server, monkeypatch):
    monkeypatch.setattr(serve_mod, "MAX_IMAGE_PIXELS", 10_000)
    ok, buf = cv2.imencode(".png", np.zeros((200, 200, 3), np.uint8))
    assert ok
    for path in ("/measure", "/identify", "/ar/push"):
        req = urllib.request.Request(f"{test_server}{path}", data=buf.tobytes(), method="POST")
        req.add_header("Content-Type", "image/png")
        status, _, body = _open(req)
        assert status == 200
        data = json.loads(body)
        assert data["ok"] is False
        assert "too large" in data["error"]


def test_image_dims_from_headers():
    img = np.zeros((123, 456, 3), np.uint8)
    for ext in (".jpg", ".png", ".webp"):
        ok, buf = cv2.imencode(ext, img)
        assert ok
        assert serve_mod._image_dims(buf.tobytes()) == (456, 123), ext
    assert serve_mod._image_dims(b"BM" + b"\0" * 100) is None
    with pytest.raises(serve_mod.DetectionError):
        serve_mod.decode_image(b"BM" + b"\0" * 100)


def test_unknown_post_route_is_404(test_server):
    req = urllib.request.Request(f"{test_server}/nope", data=b"x", method="POST")
    status, _, _ = _open(req)
    assert status == 404


def test_marketplace_disabled_by_default(test_server, monkeypatch):
    monkeypatch.delenv("CARDCENTER_ENABLE_MARKETPLACE", raising=False)
    req = urllib.request.Request(
        f"{test_server}/marketplace/tokenize", data=b'{"wallet": "0xabc"}', method="POST"
    )
    req.add_header("Content-Type", "application/json")
    assert _open(req)[0] == 404
    assert _open(urllib.request.Request(f"{test_server}/marketplace/assets"))[0] == 404


def test_marketplace_requires_wallet_when_enabled(test_server, monkeypatch):
    monkeypatch.setenv("CARDCENTER_ENABLE_MARKETPLACE", "1")
    req = urllib.request.Request(f"{test_server}/marketplace/tokenize", data=b"{}", method="POST")
    req.add_header("Content-Type", "application/json")
    status, _, body = _open(req)
    assert status == 200
    assert json.loads(body) == {"ok": False, "error": "wallet is required"}


def test_quipu_hidden_by_default(test_server, monkeypatch):
    monkeypatch.delenv("CARDCENTER_EXPOSE_QUIPU", raising=False)
    assert _open(urllib.request.Request(f"{test_server}/quipu"))[0] == 404


def test_internal_errors_are_generic(test_server, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("secret path /data/cardcenter.db and SQL details")

    monkeypatch.setattr(serve_mod, "_measure_payload", boom)
    req = urllib.request.Request(f"{test_server}/measure", data=_card_jpeg(), method="POST")
    req.add_header("Content-Type", "image/jpeg")
    status, _, body = _open(req)
    assert status == 500
    data = json.loads(body)
    assert data["error"] == "internal error"
    assert "secret" not in body.decode()
    assert data["ref"]


def test_ar_sessions_are_bounded(monkeypatch):
    monkeypatch.setattr(serve_mod, "MAX_AR_SESSIONS", 3)
    serve_mod._AR_SESSIONS.clear()
    for i in range(10):
        serve_mod._get_or_create_ar_session(f"dev_{i}")
    assert list(serve_mod._AR_SESSIONS) == ["dev_7", "dev_8", "dev_9"]
    serve_mod._AR_SESSIONS.clear()


def test_bounded_server_limits_connections():
    srv = serve_mod.BoundedThreadingHTTPServer(("127.0.0.1", 0), Handler, max_connections=2)
    try:
        assert srv.daemon_threads is True
        assert srv._slots.acquire(timeout=0) and srv._slots.acquire(timeout=0)
        assert not srv._slots.acquire(timeout=0)
    finally:
        srv.server_close()


def _strict(text: str):
    """json.loads the way a browser's JSON.parse does: no NaN or Infinity."""
    def refuse(tok):
        raise ValueError(f"non-JSON token {tok}")
    return json.loads(text, parse_constant=refuse)


def test_ar_push_answer_is_strict_json_on_a_soft_frame(test_server):
    """Field report (2.16.0, Android Chrome): PUSH FAILED on every frame, inset
    reading 'HTTP 200 not JSON'. A soft frame leaves one side with no finite
    information floor; its advice gain was float('inf'), written as the bare
    token Infinity, which JSON.parse rejects."""
    token = _new_token(test_server)
    img, _, _ = render_capture(left_mm=3.4, right_mm=2.6, distance_mm=300.0, focal_px=900.0,
                               image_size=(540, 632), noise_sigma=2.0)
    img = cv2.GaussianBlur(img, (0, 0), 1.5)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
    assert ok
    payload, ctype = _multipart({"holder": "raw", "lens": "main"}, buf.tobytes())
    for _ in range(4):
        req = urllib.request.Request(f"{test_server}/ar/push", data=payload, method="POST")
        req.add_header("Content-Type", ctype)
        req.add_header("Cookie", f"bakugo_device={token}")
        status, _, body = _open(req)
        assert status == 200
        data = _strict(body.decode())
        assert data["ok"] is True


def test_dumps_never_writes_non_json_tokens():
    from cardcenter.serve import _dumps

    out = _dumps({"a": float("inf"), "b": [float("nan"), 1.5, -float("inf")],
                  "c": {"d": np.float64("inf"), "e": np.float32(2.0)}, "f": (1, float("nan"))})
    data = _strict(out)
    assert data == {"a": None, "b": [None, 1.5, None], "c": {"d": None, "e": 2.0}, "f": [1, None]}
