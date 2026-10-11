"""Accessibility and WCAG compliance tests for Bakugo web UI."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading

import pytest
from playwright.sync_api import sync_playwright

from cardcenter.serve import PAGE


@pytest.fixture()
def app_server():
    state = {"url": ""}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path == "/":
                body = PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/config":
                body = json.dumps({
                    "ok": True,
                    "device_id": "test_dev",
                    "ip": "127.0.0.1",
                    "version": "2.25.0",
                    "holders": [{"id": "raw", "label": "Raw"}],
                }).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok": true}')

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    state["url"] = f"http://127.0.0.1:{srv.server_address[1]}"
    yield state
    srv.shutdown()
    srv.server_close()


@pytest.fixture()
def browser_page(app_server):
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(args=["--use-fake-ui-for-media-stream",
                                              "--use-fake-device-for-media-stream"])
        except Exception as exc:
            pytest.skip(f"chromium unavailable: {exc}")
        ctx = browser.new_context(viewport={"width": 412, "height": 915},
                                  permissions=["camera"], base_url=app_server["url"])
        pg = ctx.new_page()
        pg.goto("/")
        yield pg
        browser.close()


def test_viewport_meta_allows_zoom(browser_page):
    """WCAG 1.4.4: Pinch-to-zoom must not be disabled by maximum-scale or user-scalable=no."""
    meta = browser_page.evaluate("""() => {
        const el = document.querySelector('meta[name="viewport"]');
        return el ? el.getAttribute('content') : null;
    }""")
    assert meta is not None, "Missing meta viewport tag"
    content = meta.lower()
    assert "user-scalable=no" not in content, "user-scalable=no blocks zoom"
    assert "user-scalable=0" not in content, "user-scalable=0 blocks zoom"
    assert "maximum-scale=1" not in content, "maximum-scale=1 blocks zoom"
    assert "width=device-width" in content


def test_axe_core_scan_zero_serious_or_critical_violations(browser_page):
    """Run an inlined axe-core scan verifying zero serious or critical violations."""
    import axe_playwright_python

    axe_path = Path(axe_playwright_python.__file__).parent / "axe.min.js"
    assert axe_path.exists(), f"Local axe.min.js not found at {axe_path}"

    axe_source = axe_path.read_text(encoding="utf-8")
    # Inline the script directly without any CDN request
    browser_page.evaluate(axe_source)

    results = browser_page.evaluate("""() => {
        return new Promise((resolve, reject) => {
            axe.run(document, {
                runOnly: {
                    type: "tag",
                    values: ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "best-practice"]
                }
            }, (err, results) => {
                if (err) reject(err);
                else resolve(results);
            });
        });
    }""")

    violations = results.get("violations", [])
    serious_or_critical = [v for v in violations if v.get("impact") in ("serious", "critical")]
    if serious_or_critical:
        details = []
        for v in serious_or_critical:
            nodes_info = [n.get("html", "") for n in v.get("nodes", [])]
            details.append(f"[{v.get('impact')}] {v.get('id')}: {v.get('help')} -> {nodes_info}")
        pytest.fail("Axe-core reported serious/critical violations:\n" + "\n".join(details))


def test_text_size_switcher_persists(browser_page):
    """Verify that A / A+ / A++ switcher changes html font-size and saves to localStorage."""
    # Click A+
    browser_page.click('.btn-text-size[data-size="125%"]')
    font_size = browser_page.evaluate("document.documentElement.style.fontSize")
    assert font_size == "125%"
    stored = browser_page.evaluate("localStorage.getItem('bakugo_text_size')")
    assert stored == "125%"

    # Click A++
    browser_page.click('.btn-text-size[data-size="150%"]')
    font_size = browser_page.evaluate("document.documentElement.style.fontSize")
    assert font_size == "150%"
    stored = browser_page.evaluate("localStorage.getItem('bakugo_text_size')")
    assert stored == "150%"
