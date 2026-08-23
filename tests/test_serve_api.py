"""Tests for cardcenter.serve HTTP & AR API, CORS preflights, and multi-tenant scoping."""

import io
import json
import os
import tempfile
import urllib.request
import urllib.parse
from http.server import HTTPServer
from threading import Thread

import cv2
import pytest

from cardcenter import __version__
from cardcenter.serve import Handler
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


def test_cors_options_preflight(test_server):
    """Ensure OPTIONS pre-flight requests return 204 and complete CORS headers."""
    req = urllib.request.Request(f"{test_server}/measure", method="OPTIONS")
    req.add_header("Origin", "capacitor://localhost")
    req.add_header("Access-Control-Request-Method", "POST")
    req.add_header("Access-Control-Request-Headers", "X-Device-ID, Content-Type")

    with urllib.request.urlopen(req) as resp:
        assert resp.status == 204
        headers = dict(resp.headers)
        assert headers.get("Access-Control-Allow-Origin") == "*"
        assert "POST" in headers.get("Access-Control-Allow-Methods", "")
        assert "X-Device-ID" in headers.get("Access-Control-Allow-Headers", "")


def test_health_and_config(test_server):
    """Test /health and /config discovery endpoints."""
    # /health
    req = urllib.request.Request(f"{test_server}/health")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        assert data["status"] == "healthy"
        assert data["version"] == __version__
        assert "db" in data

    # /config
    req = urllib.request.Request(f"{test_server}/config")
    req.add_header("X-Device-ID", "dev_test_config")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        assert data["ok"] is True
        assert data["version"] == __version__
        assert len(data["holders"]) > 0
        assert "PSA" in data["graders"]

    # /manifest.json
    req = urllib.request.Request(f"{test_server}/manifest.json")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        assert data["short_name"] == "Bakugo"
        assert data["display"] == "standalone"


def test_ar_push_streaming_lifecycle(test_server):
    """Test real-time AR frame push and session lifecycle over HTTP."""
    # 1. Reset / initialize AR session
    req = urllib.request.Request(f"{test_server}/ar/reset", method="POST")
    req.add_header("X-Device-ID", "device_ar_user")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        assert data["ok"] is True

    # 2. Push synthetic frame
    img, _, _ = render_capture(left_mm=2.5, right_mm=3.5, top_mm=3.0, bottom_mm=3.0)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    assert ok

    # Build multipart payload
    boundary = "----TestBoundary12345678"
    body = io.BytesIO()
    body.write(f"--{boundary}\r\n".encode())
    body.write(b'Content-Disposition: form-data; name="holder"\r\n\r\nraw\r\n')
    body.write(f"--{boundary}\r\n".encode())
    body.write(b'Content-Disposition: form-data; name="lens"\r\n\r\nmain\r\n')
    body.write(f"--{boundary}\r\n".encode())
    body.write(b'Content-Disposition: form-data; name="image"; filename="frame.jpg"\r\n')
    body.write(b"Content-Type: image/jpeg\r\n\r\n")
    body.write(buf.tobytes())
    body.write(f"\r\n--{boundary}--\r\n".encode())

    req = urllib.request.Request(f"{test_server}/ar/push", data=body.getvalue(), method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    req.add_header("X-Device-ID", "device_ar_user")

    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        assert data["ok"] is True
        assert "tracking" in data
        assert "headline" in data
        assert "seen_frames" in data
        assert data["seen_frames"] >= 1


def test_measure_and_tenant_isolation(test_server):
    """Test full /measure endpoint and multi-tenant scan separation."""
    img, _, _ = render_capture(left_mm=2.5, right_mm=3.5, top_mm=3.0, bottom_mm=3.0)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    assert ok

    # User A measures a card
    boundary = "----UserABoundary"
    body = io.BytesIO()
    body.write(f"--{boundary}\r\n".encode())
    body.write(b'Content-Disposition: form-data; name="holder"\r\n\r\nraw\r\n')
    body.write(f"--{boundary}\r\n".encode())
    body.write(b'Content-Disposition: form-data; name="lens"\r\n\r\nmain\r\n')
    body.write(f"--{boundary}\r\n".encode())
    body.write(b'Content-Disposition: form-data; name="image"; filename="shot.jpg"\r\n')
    body.write(b"Content-Type: image/jpeg\r\n\r\n")
    body.write(buf.tobytes())
    body.write(f"\r\n--{boundary}--\r\n".encode())

    req = urllib.request.Request(f"{test_server}/measure", data=body.getvalue(), method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    req.add_header("X-Device-ID", "tenant_device_AAA")

    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        assert data["ok"] is True
        assert data["ratio"] > 0
        assert data["overlay"] != ""
        assert data.get("saved_local") is True

    # User A checks /my-scans -> has 1 scan
    req_a = urllib.request.Request(f"{test_server}/my-scans")
    req_a.add_header("X-Device-ID", "tenant_device_AAA")
    with urllib.request.urlopen(req_a) as resp:
        scans_a = json.loads(resp.read().decode())["scans"]
        assert len(scans_a) == 1

    # User B checks /my-scans -> has 0 scans (tenant isolated)
    req_b = urllib.request.Request(f"{test_server}/my-scans")
    req_b.add_header("X-Device-ID", "tenant_device_BBB")
    with urllib.request.urlopen(req_b) as resp:
        scans_b = json.loads(resp.read().decode())["scans"]
        assert len(scans_b) == 0
