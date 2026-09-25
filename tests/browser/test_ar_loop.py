"""The live-AR loop, in a real browser with a fake camera.

Pins the failure mode where the preview runs but nothing is ever posted to
/ar/push: the HUD sits on "SEARCHING", the debug inset stays an unpainted
300x150 canvas, and the empty `catch` in arTick hides the reason.

Needs playwright with chromium (`pip install playwright && playwright
install chromium`); skipped otherwise, so it does not block the suite on a
machine without a browser.
"""

from __future__ import annotations

import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

sync_playwright = pytest.importorskip("playwright.sync_api").sync_playwright

from cardcenter.serve import PAGE  # noqa: E402

QUAD = [[100.0, 120.0], [300.0, 120.0], [300.0, 400.0], [100.0, 400.0]]


def _jpeg_size(blob: bytes) -> tuple[int, int]:
    i = 2
    while i < len(blob) - 9:
        if blob[i] != 0xFF:
            i += 1
            continue
        marker, seg = blob[i + 1], int.from_bytes(blob[i + 2:i + 4], "big")
        if marker in (0xC0, 0xC1, 0xC2):
            return (int.from_bytes(blob[i + 7:i + 9], "big"),
                    int.from_bytes(blob[i + 5:i + 7], "big"))
        i += 2 + seg
    return (0, 0)


@pytest.fixture()
def stub():
    state = {"pushes": [], "mode": "ok"}

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
            if self.path in ("/measure", "/identify"):
                pushes_before = len(state["pushes"])
                if state["mode"] == "still_hang":
                    time.sleep(4)
                else:
                    time.sleep(1.5)            # a slow shop uplink
                m = re.search(rb"\xff\xd8\xff.*", body, re.S)
                w, h = _jpeg_size(m.group(0) if m else b"")
                fields = dict(re.findall(rb'name="(crop_x|crop_y|full_w|full_h)"\r\n\r\n(\d+)', body))
                state.setdefault(self.path, []).append({
                    "w": w, "h": h, "bytes": len(body),
                    "fields": {k.decode(): int(v) for k, v in fields.items()},
                    "pushes_during": len(state["pushes"]) - pushes_before})
                if self.path == "/measure":
                    return self._json({"ok": False, "error": "stub: not measured"})
                return self._json({"ok": True, "identified": False, "engine": "stub",
                                   "tokens_considered": 0, "identification_id": None})
            if self.path != "/ar/push":
                return self._json({"ok": True})
            if state["mode"] == "hang":
                time.sleep(9)
            m = re.search(rb"\xff\xd8\xff.*", body, re.S)
            w, h = _jpeg_size(m.group(0) if m else b"")
            extra = {k.decode(): float(v) for k, v in re.findall(
                rb'name="(aim_x|aim_y|source_scale)"\r\n\r\n([0-9.]+)', body)}
            state["pushes"].append({"w": w, "h": h, **extra})
            if state["mode"] == "infinity":
                # what a 2.16.0 server sent for a soft frame
                page = (b'{"ok": true, "tracking": true, "headline": "tracking", "quad": '
                        + json.dumps(QUAD).encode()
                        + b', "guidance": ["hold steadier or let it refocus"], "measured_frames": 0,'
                          b' "seen_frames": 1, "ratio": null, "settled": false, "verdict": "UNDECIDED",'
                          b' "information": {"advice": [{"key": "contrast", "gain": Infinity}],'
                          b' "sigma_cr_pp": NaN, "x": [-Infinity, 1]}}')
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(page)))
                self.end_headers()
                self.wfile.write(page)
                return
            if state["mode"] == "html_error":
                page = b"<html>502 Bad Gateway</html>"
                self.send_response(502)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(page)))
                self.end_headers()
                self.wfile.write(page)
                return
            if state["mode"] == "settled":
                return self._json({"ok": True, "tracking": True, "headline": "66.8", "quad": QUAD,
                                   "guidance": [], "measured_frames": 4, "seen_frames": 9,
                                   "ratio": 66.8, "settled": True, "grade_estimate": "PSA 8",
                                   "grade_confidence": 0.52, "grade_ceiling": "6-10",
                                   "verdict": "ABOVE"})
            self._json({"ok": True, "tracking": True, "headline": "tracking", "quad": QUAD,
                        "guidance": [], "measured_frames": 0, "seen_frames": 1, "ratio": None,
                        "settled": False, "verdict": "UNDECIDED"})

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    state["url"] = f"http://127.0.0.1:{srv.server_address[1]}"
    yield state
    srv.shutdown()
    srv.server_close()


@pytest.fixture()
def page(stub):
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(args=["--use-fake-ui-for-media-stream",
                                              "--use-fake-device-for-media-stream"])
        except Exception as exc:                      # chromium not installed
            pytest.skip(f"chromium unavailable: {exc}")
        ctx = browser.new_context(viewport={"width": 412, "height": 915},
                                  permissions=["camera"], base_url=stub["url"])
        pg = ctx.new_page()
        pg.warnings = []
        pg.on("console", lambda m: pg.warnings.append(m.text))
        pg.goto("/")
        yield pg
        browser.close()


def _wait_pushes(page, n=3):
    page.wait_for_function(
        f"() => typeof arStats !== 'undefined' && arStats.pushes > {n}", timeout=20000)


def test_loop_starts_and_posts_frames(page, stub):
    _wait_pushes(page)
    stats = page.evaluate("({pushes: arStats.pushes, ok: arStats.ok, fail: arStats.fail, "
                          "status: arStats.status, error: arStats.error})")
    assert stats["ok"] >= 3 and stats["fail"] == 0 and stats["status"] == 200
    assert stats["error"] == ""
    assert page.evaluate("document.querySelector('#hud-status').textContent") == "TRACKING"


def test_posted_frame_is_the_region_the_user_frames(page, stub):
    _wait_pushes(page)
    vw, vh = (int(x) for x in page.evaluate("arStats.video").split("x"))
    assert vw > vh                                   # landscape camera, portrait viewport
    crop = page.evaluate("pushCrop")
    assert crop["h"] == vh and crop["w"] < vw        # cover-cropped, not the whole frame
    first = stub["pushes"][0]
    # 540 across, taller than wide: the same detail the uncropped frame had
    assert first["w"] == 540 and first["h"] > first["w"]


def test_quad_maps_back_through_the_crop(page):
    _wait_pushes(page)
    crop = page.evaluate("pushCrop")
    off = page.evaluate("({w: offscreenCanvas.width, h: offscreenCanvas.height})")
    quad = page.evaluate("targetQuad")
    assert quad is not None
    expect_x = crop["x"] + QUAD[0][0] * crop["w"] / off["w"]
    expect_y = crop["y"] + QUAD[0][1] * crop["h"] / off["h"]
    assert abs(quad[0][0] - expect_x) < 1.5 and abs(quad[0][1] - expect_y) < 1.5
    assert quad[0][0] > QUAD[0][0]                   # offset by the crop origin


def test_debug_inset_shows_the_posted_frame(page):
    _wait_pushes(page)
    inset = page.evaluate("""() => {
        const c = document.querySelector('#ar-debug');
        const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
        let lit = 0;
        for (let i = 0; i < d.length; i += 4) if (d[i] + d[i+1] + d[i+2] > 30) lit++;
        return {w: c.width, h: c.height, lit: lit / (d.length / 4)};
    }""")
    assert (inset["w"], inset["h"]) != (300, 150)    # not the unpainted default
    assert inset["lit"] > 0.2


def test_zoom_crops_the_posted_frame_when_the_camera_has_no_zoom(page, stub):
    """The fake camera exposes no zoom capability, so the control falls back to
    a capture crop: same 540 px budget, spent on a smaller region."""
    _wait_pushes(page)
    wide = page.evaluate("pushCrop")
    assert page.evaluate("zoomIsOptical") is False
    page.evaluate("setZoom(3)")
    page.wait_for_function("() => pushCrop && pushCrop.w < %d" % (wide["w"] - 10), timeout=10000)
    tight = page.evaluate("pushCrop")
    assert tight["w"] == pytest.approx(wide["w"] / 3, rel=0.15)
    assert tight["h"] == pytest.approx(wide["h"] / 3, rel=0.15)
    # still centred on the reticle, and still 540 across
    assert tight["x"] > wide["x"] and tight["y"] > wide["y"]
    assert page.evaluate("offscreenCanvas.width") == 540
    assert "crop" in page.evaluate("document.querySelector('#zoom-read').textContent")
    page.evaluate("setZoom(1)")


def test_still_capture_falls_back_to_the_video_frame(page, stub):
    """Without ImageCapture the measurement still gets a frame, and says which."""
    _wait_pushes(page)
    page.evaluate("window.ImageCapture = undefined")
    page.evaluate("""() => { window.__sent = null;
                             window.sendStillPhoto = (b) => { window.__sent = b.size; }; }""")
    page.evaluate("triggerARFreeze()")
    page.wait_for_function("() => window.__sent > 0", timeout=10000)
    assert page.evaluate("arStats.still").startswith("frame ")
    assert page.evaluate("arStats.still").split(" ")[1] == page.evaluate("arStats.video")
    # the sharpest of a burst, not whichever frame the button caught
    assert "best of 4" in page.evaluate("arStats.still")


def test_camera_picker_lists_the_cameras(page):
    page.wait_for_function(
        "() => document.querySelector('#setting-camera').options.length > 1", timeout=10000)
    labels = page.evaluate(
        "[...document.querySelector('#setting-camera').options].map(o => o.textContent)")
    assert labels[0].startswith("Default")
    assert len(labels) > 1


def test_failures_are_reported_not_swallowed(page, stub):
    _wait_pushes(page)
    stub["mode"] = "html_error"
    page.evaluate("arStats.fail = 0; arStats.error = ''")
    page.wait_for_function("() => typeof arStats !== 'undefined' && arStats.fail > 0",
                           timeout=15000)
    state = page.evaluate("({error: arStats.error, status: arStats.status})")
    assert state["status"] == 502 and "502" in state["error"]
    assert any("ar/push" in w for w in page.warnings)


def test_a_hung_request_does_not_kill_the_loop(page, stub):
    """One request that never answers used to leave isPushing true forever."""
    _wait_pushes(page)
    stub["mode"] = "hang"
    page.evaluate("arStats.timeouts = 0; arStats.pushes = 0")
    page.wait_for_function("() => typeof arStats !== 'undefined' && arStats.timeouts > 0",
                           timeout=15000)
    stub["mode"] = "ok"
    # the loop recovers on its own: successful pushes resume and the error clears
    page.evaluate("arStats.ok = 0")
    page.wait_for_function("() => typeof arStats !== 'undefined' && arStats.ok > 1",
                           timeout=20000)
    assert page.evaluate("arStats.error") == ""
    assert page.evaluate("targetQuad") is not None


def test_stale_overlay_is_dropped(page, stub):
    """A box from an old frame must not sit on a live preview."""
    _wait_pushes(page)
    assert page.evaluate("targetQuad") is not None
    stub["mode"] = "hang"
    page.wait_for_function("() => targetQuad === null", timeout=15000)
    chip = page.evaluate("document.querySelector('#hud-status').textContent")
    assert chip in ("RECONNECTING", "SEARCHING", "PUSH TIMED OUT"), chip
    assert page.evaluate("arStats.stale") > 0


def test_still_upload_is_cropped_to_the_tracked_card(page, stub):
    """At the shop the still went up whole, twice. Now it is the tracked card
    plus margin, with its place in the photo so the server keeps the lens."""
    _wait_pushes(page)
    page.evaluate("window.ImageCapture = undefined")      # still = the video frame
    vw, vh = (int(x) for x in page.evaluate("arStats.video").split("x"))
    page.evaluate("triggerARFreeze()")
    page.wait_for_function("() => !stillBusy && !document.querySelector('#btn-action').disabled",
                           timeout=20000)
    m, i = stub["/measure"][0], stub["/identify"][0]
    f = m["fields"]
    assert f["full_w"] == vw and f["full_h"] == vh
    assert 64 <= m["w"] < vw / 2 and 64 <= m["h"] < vh
    assert f["crop_x"] + m["w"] <= vw and f["crop_y"] + m["h"] <= vh
    # the crop contains the tracked card
    quad = QUAD
    crop = page.evaluate("pushCrop")
    off = page.evaluate("({w: offscreenCanvas.width, h: offscreenCanvas.height})")
    cx = crop["x"] + (quad[0][0] + quad[1][0]) / 2 * crop["w"] / off["w"]
    cy = crop["y"] + (quad[0][1] + quad[2][1]) / 2 * crop["h"] / off["h"]
    assert f["crop_x"] < cx < f["crop_x"] + m["w"]
    assert f["crop_y"] < cy < f["crop_y"] + m["h"]
    # /identify gets the same small image, and the AR loop gave up the link
    assert (i["w"], i["h"]) == (m["w"], m["h"])
    assert m["pushes_during"] <= 1
    assert "card crop" in page.evaluate("arStats.still")
    # and the loop resumes afterwards
    page.evaluate("arStats.ok = 0")
    page.wait_for_function("() => arStats.ok > 1", timeout=10000)


def test_a_hung_still_upload_says_so(page, stub):
    _wait_pushes(page)
    page.evaluate("window.ImageCapture = undefined; STILL_TIMEOUT_MS = 1500")
    stub["mode"] = "still_hang"
    page.evaluate("triggerARFreeze()")
    page.wait_for_function(
        "() => document.querySelector('#out').textContent.includes('no reply')", timeout=15000)
    assert not page.evaluate("document.querySelector('#btn-action').disabled")


def test_an_older_server_writing_infinity_still_tracks(page, stub):
    """Field report: 'HTTP 200 not JSON' on every frame from a server that
    wrote float('inf') as Infinity. The page now reads it as null."""
    stub["mode"] = "infinity"
    page.evaluate("arStats.ok = 0; arStats.fail = 0; arStats.error = ''")
    page.wait_for_function("() => arStats.ok > 2", timeout=20000)
    assert page.evaluate("arStats.error") == ""
    assert page.evaluate("targetQuad") is not None
    assert page.evaluate("parseReply('{\"a\": [NaN, -Infinity], \"b\": Infinity}')") == \
        {"a": [None, None], "b": None}
    assert page.evaluate("parseReply('<html>502</html>') === undefined")


def test_tapping_the_preview_does_not_break_the_loop(page, stub):
    """The fake camera has no focus controls: a tap must be a quiet no-op."""
    _wait_pushes(page)
    box = page.locator("#ar-video").bounding_box()
    page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 3)
    assert page.evaluate("focusAt(10, 10)") is False
    page.evaluate("arStats.ok = 0")
    page.wait_for_function("() => arStats.ok > 1", timeout=10000)


def test_tap_on_the_preview_picks_the_card(page, stub):
    """A shop counter has several cards: a tap sends the point, in the pushed
    frame's own 0..1 coordinates, with the next frame, once."""
    _wait_pushes(page)
    box = page.locator("#ar-video").bounding_box()
    n0 = len(stub["pushes"])
    page.mouse.click(box["x"] + box["width"] * 0.25, box["y"] + box["height"] * 0.5)
    page.wait_for_function("() => arStats.aim !== ''", timeout=5000)
    page.wait_for_timeout(1500)
    aimed = [p for p in stub["pushes"][n0:] if "aim_x" in p]
    assert len(aimed) == 1, stub["pushes"][n0:]
    assert 0.15 < aimed[0]["aim_x"] < 0.35 and 0.4 < aimed[0]["aim_y"] < 0.6
    # every push says how many camera pixels each pushed pixel stands for
    crop = page.evaluate("pushCrop")
    assert stub["pushes"][-1]["source_scale"] == pytest.approx(crop["w"] / 540, rel=0.02)


def test_the_still_is_the_video_frame_when_the_video_is_big_enough(page):
    """2.19.1 field report: Measure Card and Identify found no card in the
    camera's separate photo while every live frame of those seconds did. A 4K
    video frame is the still (same pixels the outline was found in); a small
    video still uses the photo."""
    assert page.evaluate("stillSource(2160, 3840, true)") == "frame"
    assert page.evaluate("stillSource(3840, 2160, true)") == "frame"
    assert page.evaluate("stillSource(1920, 1080, true)") == "photo"
    assert page.evaluate("stillSource(640, 480, false)") == "frame"


def test_a_settled_live_number_the_photo_refused_is_not_shown_as_settled(page, stub):
    """The chip said '66.8% · ~PSA 8 · range 6-10' and 'Target settled' over
    'Measurement Refused'. A refused photo now leaves the live number marked
    as not confirmed."""
    _wait_pushes(page)
    page.evaluate("window.ImageCapture = undefined")
    stub["mode"] = "settled"
    page.wait_for_function(
        "() => /PSA 8/.test(document.querySelector('#hud-status').textContent)", timeout=15000)
    page.evaluate("triggerARFreeze()")
    page.wait_for_function(
        "() => /photo refused/.test(document.querySelector('#hud-status').textContent)",
        timeout=20000)
    banner = page.evaluate("document.querySelector('#hud-guidance').textContent")
    assert "refused" in banner and "settled" not in banner.lower()


def test_guidance_wraps_instead_of_being_cut_off(page):
    """2.21.0 screenshots: 'frame is soft -- if it stays sof' under the tilt
    chip; the part that said what to do was never seen."""
    style = page.evaluate("getComputedStyle(document.querySelector('#hud-guidance')).whiteSpace")
    assert style == "normal"


def test_frame_sharpness_is_measured(page):
    v = page.evaluate("frameSharpness(null)")
    assert v >= 0 and v == v


def test_zooms_in_when_the_live_view_is_too_coarse(page):
    """2.21.1: at a distance the camera focuses the live frame had 3.3-4.2
    px/mm and live needs 4.5; it never measured. The phone zooms in itself
    (to ~6 px/mm, at most 2x), but not over a zoom the user just set."""
    page.evaluate("setZoom(1); userZoomAt = 0; autoZoomAt = 0; coarseRuns = 0")
    coarse = "({ok: true, tracking: true, live_px_per_mm: 3.5, card_frac: 0.3})"
    page.evaluate(f"maybeAutoZoom({coarse})")
    assert page.evaluate("zoomFactor()") == 1          # one frame is not enough
    page.evaluate(f"maybeAutoZoom({coarse})")
    z = page.evaluate("zoomFactor()")
    assert 1.5 <= z <= 2.0
    # a hand zoom wins for 10 s
    page.evaluate("setZoom(1); userZoomAt = Date.now(); autoZoomAt = 0; coarseRuns = 0")
    page.evaluate(f"maybeAutoZoom({coarse}); maybeAutoZoom({coarse})")
    assert page.evaluate("zoomFactor()") == 1
    page.evaluate("setZoom(1); userZoomAt = 0")


def test_bigger_live_frame_when_the_card_is_coarse(page, stub):
    """2.22.0: the card read 3.4-4.4 px/mm in the 540 px live frame at a
    distance the camera focuses; live needs 4.5. The phone sends 720 px
    frames while it tracks a coarse card, and 540 again once tracking has
    been lost for 1.5 s."""
    _wait_pushes(page)
    page.evaluate("liveW = LIVE_W; sizeARCanvases()")
    assert page.evaluate("offscreenCanvas.width") == 540
    page.evaluate("maybeLiveHiRes({ok: true, tracking: true, live_px_per_mm: 4.0, card_frac: 0.5})")
    assert page.evaluate("offscreenCanvas.width") == 720
    page.evaluate("maybeLiveHiRes({ok: true, tracking: false}); liveHiLostAt = Date.now() - 2000;"
                  "maybeLiveHiRes({ok: true, tracking: false})")
    assert page.evaluate("offscreenCanvas.width") == 540

