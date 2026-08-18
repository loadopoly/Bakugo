"""Smoke-test the web app's embedded Python glue in CPython before the browser.

Extracts the <script type="text/x-python"> block from the deployed index.html,
rewrites its Emscripten /data path to a temp dir, execs it, and drives every
function the JS calls: holders_json, measure_json, import_labels_json,
model_observations, reset_learned.
"""

import json
import os
import re
import sys
import tempfile
import time

import cv2

HTML = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "Supply-Chain-Brain", "Loadopoly-Portal", "bakugo", "index.html",
)

html = open(HTML, encoding="utf-8").read()
m = re.search(r'<script type="text/x-python" id="boot-py">(.*?)</script>', html, re.S)
assert m, "boot-py block not found in index.html"
code = m.group(1)

tmp = tempfile.mkdtemp()
data_dir = os.path.join(tmp, "data").replace("\\", "/")
code = code.replace('"/data"', repr(data_dir)).replace('"/data/cardcenter.db"', repr(data_dir + "/cardcenter.db"))

ns: dict = {}
exec(compile(code, "boot-py", "exec"), ns)

# 1. holders
h = json.loads(ns["holders_json"]())
assert h["version"] and any(x["id"] == "raw" for x in h["holders"]), h
print(f"holders_json ok: {len(h['holders'])} holders, v{h['version']}")

# 2. measure a synthetic capture (off-centre so the band is interesting)
from cardcenter.synth import render_capture

img, truth, _ = render_capture(left_mm=2.2, right_mm=3.8, top_mm=3.0, bottom_mm=3.0)
ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
assert ok
d = json.loads(ns["measure_json"](buf.tobytes(), "raw", "main"))
assert d["ok"], d
assert d.get("saved_local") is True, d
assert d.get("scan_id"), d
assert d.get("scan_count", 0) >= 1, d
assert d["predicted_grades"]["PSA"]["used_learned"] is False
print(f"measure_json ok: ratio {d['ratio']}, scan {d['scan_id']}, PSA {d['predicted_grades']['PSA']['grade']}, overlay {len(d['overlay'])}b64")

# 3. junk input handled
bad = json.loads(ns["measure_json"](b"not an image", "raw", "main"))
assert not bad["ok"] and bad["error"]
print("measure_json rejects junk:", bad["error"])

# 4. import a sync payload (certified + a vote that must not train)
from cardcenter import __version__

payload = {
    "schema_version": "cardcenter/2",
    "engine_version": __version__,
    "exported_at": time.time(),
    "client_id": "webapp-test",
    "scans": [{
        "id": 1, "card_key": "sv01-mew", "holder": "raw",
        "worst_ratio_pct": d["ratio"], "worst_ratio_sigma": 0.5,
        "worst_axis": "horizontal", "h_ratio_pct": d["ratio"], "v_ratio_pct": 50.0,
        "px_per_mm": 10.0, "inner_confidence": 0.95, "refraction_applied": 0,
        "warnings": "", "phash": 42, "created_at": time.time(),
    }],
    "labels": [
        {"scan_id": 1, "grader": "PSA", "grade": "9", "kind": "certified", "cert_number": "99887766"},
        {"scan_id": 1, "grader": "BGS", "grade": "9.5", "kind": "marketplace_vote", "cert_number": None},
    ],
}
r = json.loads(ns["import_labels_json"](json.dumps(payload)))
assert r["ok"] and r["observations"] == 1, r
print("import_labels_json (payload) ok:", r["note"])

# 5. the learned model now changes the measurement in that band
d2 = json.loads(ns["measure_json"](buf.tobytes(), "raw", "main"))
psa = d2["predicted_grades"]["PSA"]
assert psa["used_learned"] is True and psa["n_observations"] == 1, psa
print(f"learned prediction ok: PSA {psa['grade']} [learned x{psa['n_observations']}]")

# 6. training-set export path + contamination refusal
exp = {"manifest": {"ground_truth_only": True},
       "examples": [{"kind": "certified", "cert_number": "1", "grader": "PSA",
                     "grade": "8", "worst_ratio_pct": d["ratio"], "inner_confidence": 0.9}]}
r2 = json.loads(ns["import_labels_json"](json.dumps(exp)))
assert r2["ok"] and r2["observations"] == 1, r2
dirty = {"manifest": {"ground_truth_only": False}, "examples": []}
r3 = json.loads(ns["import_labels_json"](json.dumps(dirty)))
assert not r3["ok"] and "certified-only" in r3["error"], r3
print("training-set import ok; contaminated export refused")

# 7. junk file + reset
r4 = json.loads(ns["import_labels_json"]("{{{"))
assert not r4["ok"]
r5 = json.loads(ns["reset_learned"]())
assert r5["observations"] == 0
assert ns["model_observations"]() == 0
print("reset ok")

# 8. live AR glue: start + push a synthetic RGB frame
start = json.loads(ns["ar_start"]("raw", "main"))
assert start["ok"], start
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w = rgb.shape[:2]
live = json.loads(ns["ar_push"](rgb.tobytes(), w, h))
assert live["ok"], live
assert "tracking" in live and "headline" in live
rgba = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
live2 = json.loads(ns["ar_push"](rgba.tobytes(), w, h))
assert live2["ok"], live2
print(f"ar_push ok: tracking={live['tracking']} headline={live['headline']!r}")
reset = json.loads(ns["ar_reset"]())
assert reset["ok"]
print("all glue functions pass")
