"""The outline follows the card while an answer is on its way.

A fake camera plays a view sliding over a card on a textured counter. The
stub server finds the card in each pushed frame (the bright rectangle) and
answers 0.7 s later, as shop LTE does. The outline on screen is compared with
where the card is in the live preview at that moment, with the page's motion
compensation on and off.

Skipped without playwright/chromium, like test_ar_loop.
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import pytest

sync_playwright = pytest.importorskip("playwright.sync_api").sync_playwright

from cardcenter.serve import PAGE  # noqa: E402

VW, VH, FPS, SECONDS = 480, 270, 20, 4.8
LATENCY_S = 0.7


def _scene(seed=3):
    rng = np.random.default_rng(seed)
    base = rng.uniform(30, 120, (70, 110)).astype(np.float32)
    bg = cv2.resize(base, (1100, 700), interpolation=cv2.INTER_CUBIC)
    for _ in range(90):                    # things on a counter: edges to match on
        x, y = rng.integers(0, 1060), rng.integers(0, 660)
        w, h = rng.integers(8, 40), rng.integers(8, 40)
        bg[y:y + h, x:x + w] = rng.uniform(20, 150)
    bg = np.clip(bg, 0, 150).astype(np.uint8)
    img = cv2.merge([bg, (bg * 0.9).astype(np.uint8), (bg * 0.8).astype(np.uint8)])
    cx, cy = 550, 350
    img[cy - 42:cy + 42, cx - 30:cx + 30] = (235, 235, 235)          # the card
    img[cy - 30:cy + 5, cx - 24:cx + 24] = (60, 180, 90)            # its art
    return img, (cx, cy)


def _window(t):
    """Top-left of the camera's view of the scene at time t (a hand sliding)."""
    # +/-22 px across a ~200 px wide view: the card moves a fifth of the view
    # peak to peak, more than the 4-9% seen in the field
    return (int(round(550 - VW / 2 + 22 * math.sin(2 * math.pi * t / 2.4))),
            int(round(350 - VH / 2 + 10 * math.sin(2 * math.pi * t / 3.1))))


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    scene, _ = _scene()
    path = tmp_path_factory.mktemp("cam") / "slide.y4m"
    with open(path, "wb") as f:
        f.write(f"YUV4MPEG2 W{VW} H{VH} F{FPS}:1 Ip A1:1 C420jpeg\n".encode())
        for i in range(int(FPS * SECONDS)):
            x, y = _window(i / FPS)
            frame = scene[y:y + VH, x:x + VW]
            f.write(b"FRAME\n")
            f.write(cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420).tobytes())
    return str(path)


@pytest.fixture()
def stub():
    state = {}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                body = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/config":
                self._json({"ok": True, "device_id": "d", "ip": "127.0.0.1", "version": "t",
                            "holders": [{"id": "raw", "label": "Raw"}]})
            else:
                self._json({"ok": False}, 404)

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            if self.path != "/ar/push":
                return self._json({"ok": True})
            m = re.search(rb"\xff\xd8\xff.*", body, re.S)
            img = cv2.imdecode(np.frombuffer(m.group(0), np.uint8), cv2.IMREAD_GRAYSCALE)
            time.sleep(LATENCY_S)
            ys, xs = np.nonzero(img > 200)
            if len(xs) < 20:
                return self._json({"ok": True, "tracking": False, "quad": None, "guidance": [],
                                   "measured_frames": 0, "seen_frames": 1, "ratio": None,
                                   "settled": False, "verdict": "UNDECIDED"})
            x0, x1, y0, y1 = float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())
            self._json({"ok": True, "tracking": True, "quad": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                        "guidance": [], "measured_frames": 0, "seen_frames": 1, "ratio": None,
                        "settled": False, "verdict": "UNDECIDED"})

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    state["url"] = f"http://127.0.0.1:{srv.server_address[1]}"
    yield state
    srv.shutdown()
    srv.server_close()


@pytest.fixture()
def page(stub, video):
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(args=["--use-fake-ui-for-media-stream",
                                              "--use-fake-device-for-media-stream",
                                              f"--use-file-for-fake-video-capture={video}"])
        except Exception as exc:
            pytest.skip(f"chromium unavailable: {exc}")
        ctx = browser.new_context(viewport={"width": 412, "height": 915},
                                  permissions=["camera"], base_url=stub["url"])
        pg = ctx.new_page()
        pg.goto("/")
        yield pg
        browser.close()


PROBE = """() => {
  const crop = coverCrop();
  if (!crop || !targetQuad) return null;
  const c = document.createElement('canvas');
  c.width = crop.w; c.height = crop.h;
  const x = c.getContext('2d');
  x.drawImage(arVideo, crop.x, crop.y, crop.w, crop.h, 0, 0, crop.w, crop.h);
  const d = x.getImageData(0, 0, crop.w, crop.h).data;
  let x0 = 1e9, x1 = -1, y0 = 1e9, y1 = -1;
  for (let i = 0, p = 0; i < d.length; i += 4, p++) {
    if (0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2] > 200) {
      const px = p % crop.w, py = (p / crop.w) | 0;
      if (px < x0) x0 = px; if (px > x1) x1 = px; if (py < y0) y0 = py; if (py > y1) y1 = py;
    }
  }
  if (x1 < 0) return null;
  const q = targetQuad;
  return { err: Math.hypot((q[0][0] + q[2][0]) / 2 - (crop.x + (x0 + x1) / 2),
                           (q[0][1] + q[2][1]) / 2 - (crop.y + (y0 + y1) / 2)),
           card: x1 - x0 };
}"""


def _errors(page, seconds=5.0):
    out, t_end = [], time.time() + seconds
    while time.time() < t_end:
        r = page.evaluate(PROBE)
        if r:
            out.append(r["err"] / max(r["card"], 1))
        page.wait_for_timeout(90)
    return np.array(out)


def test_the_outline_stays_on_the_card_between_answers(page):
    page.wait_for_function("() => typeof arStats !== 'undefined' && arStats.ok > 2", timeout=30000)
    page.evaluate("MOTION_COMP = false")
    page.wait_for_timeout(1500)
    off = _errors(page)
    page.evaluate("MOTION_COMP = true")
    page.wait_for_timeout(1500)
    on = _errors(page)
    assert len(off) > 20 and len(on) > 20
    # error in card widths: without compensation the outline trails the card
    # by most of a card width at the fastest; with it, a fraction
    assert np.median(off) > 0.25, np.median(off)
    assert np.median(on) < 0.5 * np.median(off), (np.median(on), np.median(off))
    assert np.percentile(on, 90) < 0.3, np.percentile(on, 90)
    print("median error, card widths: off %.2f on %.2f (p90 on %.2f)" % (np.median(off), np.median(on), np.percentile(on, 90)))
    assert page.evaluate("arStats.shift") != ""
