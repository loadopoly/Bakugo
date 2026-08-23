"""A local web UI and mobile-backend server, so any browser or native app can be the front end.

Rationale for this shape: the measurement code is Python, NumPy and OpenCV.
The server provides:
1. Complete REST API for mobile native apps (Capacitor / React Native / Flutter)
   with full CORS preflight support and streaming AR endpoints (/ar/push).
2. A high-framerate WebApp with WebGL/Canvas Live AR tracking, synthesized sci-fi
   audio cues (Web Audio API), and Snell ray-traced photo metrology.
3. Offline standalone capabilities and PWA manifest for home-screen installation.
"""

from __future__ import annotations

import base64
import json
import os
import socket
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

import cv2
import numpy as np

from . import __version__
from .ar import ARSession, ARStatus
from .capture import assess_frame
from .centering import measure_centering
from .grading import available_graders, grade_band, predict_overall_grade
from .learning import maybe_load_grade_model
from .render import annotate
from .types import SLAB_PRESETS, SLAB_STACKS, CaptureSpec, DetectionError, resolve_holder

# Typical horizontal fields of view for phone cameras.
LENS_FOV = {
    "main": 68.0,
    "ultrawide": 105.0,
    "tele2x": 38.0,
    "tele5x": 16.0,
}

# Multi-tenant in-memory AR sessions keyed by device_id
_AR_SESSIONS: dict[str, ARSession] = {}


def _get_or_create_ar_session(
    device_id: str,
    holder: str = "raw",
    lens: str = "main",
    boundary: float = 55.0,
) -> ARSession:
    fov = LENS_FOV.get(lens, LENS_FOV["main"])
    session = _AR_SESSIONS.get(device_id)
    if session is None or session.holder != holder or abs(session.fov_deg - fov) > 1e-3:
        session = ARSession(holder=holder, fov_deg=fov, boundary=boundary)
        _AR_SESSIONS[device_id] = session
    return session


def _parse_multipart(body: bytes, content_type: str) -> dict[str, bytes]:
    """Minimal multipart/form-data parser.

    Written without external deps to ensure it works across all standard Python
    runtimes (including Python 3.13+ where `cgi` was removed).
    """
    if "boundary=" not in content_type:
        return {}
    boundary = content_type.split("boundary=", 1)[1].strip().strip('"')
    sep = b"--" + boundary.encode()
    out: dict[str, bytes] = {}
    for part in body.split(sep):
        if not part or part in (b"--\r\n", b"--", b"\r\n"):
            continue
        if b"\r\n\r\n" not in part:
            continue
        raw_headers, content = part.split(b"\r\n\r\n", 1)
        headers = raw_headers.decode("utf-8", "replace")
        if 'name="' not in headers:
            continue
        name = headers.split('name="', 1)[1].split('"', 1)[0]
        out[name] = content.rstrip(b"\r\n")
    return out


def _measure_payload(image_bytes: bytes, holder: str, lens: str) -> dict:
    data = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise DetectionError("that file could not be read as an image")

    # Limit maximum dimension for speed without losing metrology accuracy
    max_side = 2400
    if max(image.shape[:2]) > max_side:
        scale = max_side / max(image.shape[:2])
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    fov = LENS_FOV.get(lens, LENS_FOV["main"])
    capture = CaptureSpec.from_fov(fov, image.shape)
    result = measure_centering(image, slab=resolve_holder(holder), capture=capture)

    bands = {g: grade_band(result.worst_ratio, g, "front") for g in available_graders()}
    quality = assess_frame(image, result.corners_px, px_per_mm=result.px_per_mm)

    # Card only without side panel for optimal mobile display
    overlay = annotate(result, bands)[:, : result.rectified.shape[1]]
    ok, buf = cv2.imencode(".jpg", overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    overlay_b64 = base64.b64encode(buf).decode() if ok else ""

    w = result.worst_ratio
    lo, hi = w.interval()
    return {
        "ok": True,
        "ratio": round(w.value, 1),
        "ratio_lo": round(lo, 1),
        "ratio_hi": round(hi, 1),
        "axis": result.worst_axis.axis,
        "wider": result.worst_axis.skew_toward,
        "borders": {
            "left": round(result.horizontal.low_mm.value, 2),
            "right": round(result.horizontal.high_mm.value, 2),
            "top": round(result.vertical.low_mm.value, 2),
            "bottom": round(result.vertical.high_mm.value, 2),
        },
        "px_per_mm": round(result.px_per_mm, 1),
        "inner_confidence": round(result.quality.inner_confidence, 3),
        "holder": result.slab.name,
        "refraction": result.quality.refraction_applied,
        "bands": {
            g: {
                "label": b.best if b.is_single else f"{b.worst}\u2013{b.best}",
                "limited_by": b.limited_by,
            }
            for g, b in bands.items()
        },
        "predicted_grades": {
            g: {
                "grade": p.grade_label,
                "score": p.grade_score,
                "condition": p.condition_name,
                "subgrades": {
                    "centering": p.centering_subgrade,
                    "corners": p.estimated_corners,
                    "edges": p.estimated_edges,
                    "surface": p.estimated_surface,
                },
                "used_learned": p.used_learned,
                "n_observations": p.n_observations,
            }
            for g, p in {
                name: predict_overall_grade(
                    w, quality=result.quality, grader=name, model=maybe_load_grade_model()
                )
                for name in bands.keys()
            }.items()
        },
        "warnings": list(result.quality.warnings) + list(quality.guidance),
        "overlay": overlay_b64,
    }


def persist_measure(payload: dict, source: str = "serve", *, cloud: bool = True) -> dict:
    """Write a successful measure locally, then best-effort cloud upsert."""
    extra: dict = {}
    db = os.environ.get("CARDCENTER_DB", "cardcenter.db")
    if not db or not payload.get("ok"):
        return extra
    if cloud and sys.platform == "emscripten":
        cloud = False
    try:
        from .store import ScanStore

        with ScanStore(db) as store:
            scan_id = store.add_scan_from_measure(payload, source=source)
            extra["scan_id"] = scan_id
            extra["scan_count"] = store.scan_count()
            extra["saved_local"] = True
            if not cloud:
                extra["cloud"] = {"ok": True, "skipped": True, "table": "bakugo_scans"}
                return extra
            try:
                from .cloud import sync_scan_id

                extra["cloud"] = sync_scan_id(store, scan_id).to_dict()
            except Exception as exc:  # pragma: no cover - never fail measure
                extra["cloud"] = {"ok": False, "error": str(exc)}
    except Exception as exc:  # pragma: no cover - never fail measure
        extra["persist_error"] = str(exc)
    return extra


MANIFEST_JSON = json.dumps(
    {
        "name": "Bakugo AR Metrology",
        "short_name": "Bakugo",
        "description": "Trading card centering metrology & live AR scanner",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#0B0F15",
        "theme_color": "#0B0F15",
        "icons": [
            {
                "src": "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><rect width='100' height='100' rx='20' fill='%23121822'/><circle cx='50' cy='50' r='30' stroke='%234ED2C6' stroke-width='6' fill='none'/><line x1='50' y1='10' x2='50' y2='90' stroke='%234ED2C6' stroke-width='2'/><line x1='10' y1='50' x2='90' y2='50' stroke='%234ED2C6' stroke-width='2'/></svg>",
                "sizes": "192x192 512x512",
                "type": "image/svg+xml",
            }
        ],
    },
    indent=2,
)


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover, maximum-scale=1, user-scalable=no">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="theme-color" content="#0B0F15">
<link rel="manifest" href="/manifest.json">
<title>Bakugo — AR Metrology Hub</title>
<style>
:root{
  --ink:#0B0F15; --surface:#121822; --surface-glass:rgba(18,24,34,0.85);
  --rise:#1D2635; --rule:#2B384A; --paper:#F3EFE6; --dim:#8E9EAF;
  --key:#4ED2C6; --key-glow:rgba(78,210,198,0.35);
  --pass:#4EBA82; --hold:#D9A83A; --stop:#D45440;
  --radius-lg:16px; --radius-md:12px; --radius-sm:8px;
}
*{box-sizing:border-box;-webkit-tap-highlight-color:transparent;user-select:none}
input,select,textarea{user-select:auto}
html,body{margin:0;padding:0;background:var(--ink);color:var(--paper);
  font-family:-apple-system,BlinkMacSystemFont,"SF Pro Text","Segoe UI",Roboto,sans-serif;
  height:100%;overflow-x:hidden;overscroll-behavior-y:contain}

header.app-bar{padding:calc(10px + env(safe-area-inset-top)) 16px 10px;
  display:flex;align-items:center;gap:10px;position:sticky;top:0;
  background:rgba(11,15,21,0.92);backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);
  border-bottom:1px solid var(--rule);z-index:40}
.brand{display:flex;align-items:baseline;gap:8px}
.brand h1{font-size:17px;margin:0;font-weight:700;color:#FFF;letter-spacing:-.02em}
.brand .badge{font-size:10px;letter-spacing:.15em;text-transform:uppercase;color:var(--key);font-weight:700;padding:2px 6px;background:rgba(78,210,198,.12);border-radius:4px}
.host-pill{margin-left:auto;font-size:10px;padding:3px 8px;border-radius:12px;background:var(--surface);border:1px solid var(--rule);color:var(--dim);cursor:pointer;display:flex;align-items:center;gap:5px}
.host-pill .dot{width:6px;height:6px;border-radius:50%;background:var(--pass)}

.tab-nav{display:flex;padding:8px 16px;gap:8px;background:rgba(11,15,21,0.6)}
.tab-btn{flex:1;padding:8px 12px;border:1px solid var(--rule);background:var(--surface);color:var(--dim);border-radius:var(--radius-md);font-size:12px;font-weight:600;display:flex;align-items:center;justify-content:center;gap:6px;cursor:pointer;transition:all .15s}
.tab-btn.active{background:var(--rise);border-color:var(--key);color:#FFF;box-shadow:0 0 10px var(--key-glow)}

.quick-pills{display:flex;gap:8px;padding:8px 16px;overflow-x:auto;scrollbar-width:none}
.quick-pills::-webkit-scrollbar{display:none}
.pill-select{background:var(--surface-glass);border:1px solid var(--rule);color:var(--paper);border-radius:20px;padding:5px 10px;font-size:12px;font-weight:500;display:flex;align-items:center;gap:6px;white-space:nowrap}
.pill-select select{background:transparent;border:none;color:inherit;font-size:inherit;font-weight:inherit;outline:none;cursor:pointer}

/* AR Viewport */
#ar-container{position:relative;width:calc(100% - 32px);margin:8px 16px;height:55vh;min-height:360px;border-radius:var(--radius-lg);overflow:hidden;background:#05070A;border:1px solid var(--rule);box-shadow:0 8px 30px rgba(0,0,0,0.5)}
#ar-video{width:100%;height:100%;object-fit:cover;display:block}
#ar-canvas{position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none}
.ar-hud-overlay{position:absolute;top:10px;left:10px;right:10px;display:flex;justify-content:space-between;align-items:center;pointer-events:none}
.hud-chip{background:rgba(11,15,21,0.85);backdrop-filter:blur(10px);border:1px solid var(--key);padding:4px 10px;border-radius:20px;font-size:11px;font-family:ui-monospace,"SF Mono",monospace;color:#FFF;display:flex;align-items:center;gap:6px}
.hud-chip .radar{width:7px;height:7px;border-radius:50%;background:var(--key);animation:p 1s infinite alternate}
@keyframes p{from{opacity:.3;transform:scale(.8)}to{opacity:1;transform:scale(1.2)}}
.hud-verdict{background:rgba(11,15,21,0.85);backdrop-filter:blur(10px);border:1px solid var(--rule);padding:4px 10px;border-radius:20px;font-size:11px;font-family:ui-monospace,"SF Mono",monospace;color:var(--dim)}

/* Still & Results Output */
#out{padding-bottom:90px}
.empty-card{margin:12px 16px;padding:20px;background:var(--surface);border:1px solid var(--rule);border-radius:var(--radius-lg);font-size:13.5px;line-height:1.6;color:var(--dim)}
.empty-card b{color:var(--paper)}
.empty-card ul{padding-left:18px;margin:10px 0 0}
.empty-card li{margin-bottom:6px}

.strip{margin:12px 16px 0;height:54px;position:relative;border-radius:var(--radius-md);background:var(--rise);border:1px solid var(--rule);overflow:hidden}
.strip .seg{position:absolute;top:0;bottom:0}
.strip .segL{left:0;background:linear-gradient(90deg,#243244,#2C3D52)}
.strip .segR{right:0;background:linear-gradient(270deg,#243244,#2C3D52)}
.strip .split{position:absolute;top:0;bottom:0;width:2px;background:var(--key);box-shadow:0 0 0 1px rgba(11,15,21,.7)}
.strip .ci{position:absolute;top:0;bottom:0;background:var(--key);opacity:.25}
.strip .mid{position:absolute;top:0;bottom:0;left:50%;width:1px;background:var(--paper);opacity:.35}
.strip .midcap{position:absolute;top:4px;left:50%;transform:translateX(-50%);font-size:8px;letter-spacing:.16em;color:var(--dim);text-transform:uppercase;background:var(--rise);padding:1px 5px;border-radius:3px}
.strip .tag{position:absolute;bottom:5px;font-size:9.5px;letter-spacing:.1em;color:#AFC0CF;text-transform:uppercase;font-family:ui-monospace,"SF Mono",monospace}

.ratio-row{display:flex;align-items:baseline;gap:10px;padding:14px 16px 2px}
.ratio-row b{font-size:44px;line-height:.92;font-weight:800;letter-spacing:-.03em;font-variant-numeric:tabular-nums;font-family:ui-monospace,"SF Mono",monospace}
.ratio-row span{font-size:13px;color:var(--dim);font-variant-numeric:tabular-nums;font-family:ui-monospace,"SF Mono",monospace}
.sub-meta{padding:0 16px 12px;font-size:12px;color:var(--dim)}

.sect-title{padding:12px 16px 6px;font-size:10px;letter-spacing:.2em;text-transform:uppercase;color:var(--dim);font-weight:700}
.chips-row{display:flex;gap:8px;padding:0 16px 6px;flex-wrap:wrap}
.chip{background:var(--surface);border:1px solid var(--rule);border-radius:var(--radius-sm);padding:7px 11px;font-size:12px;font-family:ui-monospace,"SF Mono",monospace;display:flex;align-items:center;gap:6px}
.chip i{font-style:normal;color:var(--dim);font-size:10px}
.chip.gold{border-color:var(--key);background:rgba(78,210,198,.12);color:#FFF}

table.mm-table{width:calc(100% - 32px);margin:8px 16px 0;border-collapse:collapse;font-family:ui-monospace,"SF Mono",monospace;font-size:12px}
table.mm-table td{padding:6px 0;border-top:1px solid var(--rule);color:var(--dim)}
table.mm-table td:last-child{text-align:right;color:var(--paper);font-variant-numeric:tabular-nums}

img.ov{width:calc(100% - 32px);max-height:44vh;object-fit:contain;margin:12px 16px 0;border-radius:var(--radius-md);border:1px solid var(--rule);display:block;background:var(--surface)}

/* Native Dock */
.native-dock{position:fixed;left:0;right:0;bottom:0;padding:10px 16px calc(12px + env(safe-area-inset-bottom));background:linear-gradient(to top,var(--ink) 80%,rgba(11,15,21,0));display:flex;align-items:center;gap:10px;z-index:50}
.btn-primary{flex:1;border:none;border-radius:var(--radius-md);padding:15px;font-size:15px;font-weight:700;background:var(--key);color:#061116;font-family:inherit;cursor:pointer;box-shadow:0 4px 16px var(--key-glow);display:flex;align-items:center;justify-content:center;gap:8px}
.btn-primary:active{transform:scale(.98)}
.btn-primary[disabled]{opacity:.45}
.btn-round{width:50px;height:50px;border-radius:var(--radius-md);border:1px solid var(--rule);background:var(--surface);color:var(--paper);display:flex;align-items:center;justify-content:center;font-size:18px;cursor:pointer}
.btn-round:active{transform:scale(.94)}
.btn-round.active{background:rgba(78,210,198,.2);border-color:var(--key);color:var(--key)}

/* Settings Modal */
.modal-backdrop{position:fixed;inset:0;background:rgba(0,0,0,0.7);backdrop-filter:blur(8px);z-index:100;display:flex;align-items:flex-end;justify-content:center}
.modal-sheet{background:var(--surface);border-top:1px solid var(--rule);border-radius:var(--radius-lg) var(--radius-lg) 0 0;width:100%;max-width:500px;padding:20px 20px calc(24px + env(safe-area-inset-bottom));box-shadow:0 -10px 40px rgba(0,0,0,0.6)}
.modal-title{font-size:16px;font-weight:700;color:#FFF;margin-bottom:14px;display:flex;justify-content:space-between;align-items:center}
.form-row{margin-bottom:14px}
.form-row label{display:block;font-size:11px;text-transform:uppercase;letter-spacing:.1em;color:var(--dim);margin-bottom:6px;font-weight:600}
.form-row input{width:100%;background:var(--rise);border:1px solid var(--rule);border-radius:var(--radius-sm);padding:10px 12px;color:#FFF;font-size:13px;font-family:ui-monospace,"SF Mono",monospace;outline:none}
.btn-save{width:100%;border:none;background:var(--key);color:#061116;font-weight:700;padding:12px;border-radius:var(--radius-md);cursor:pointer;font-size:14px}
input[type=file]{position:absolute;width:1px;height:1px;opacity:0;pointer-events:none}
.hidden{display:none !important}
</style>
</head>
<body>

<header class="app-bar">
  <div class="brand">
    <h1>Bakugo</h1>
    <span class="badge">AR Hub</span>
  </div>
  <div class="host-pill" id="btn-host-settings" title="Backend Connection Settings">
    <span class="dot"></span>
    <span id="host-label">Container</span>
  </div>
</header>

<div class="tab-nav">
  <button type="button" class="tab-btn active" id="tab-ar">⚡ Live AR Metrology</button>
  <button type="button" class="tab-btn" id="tab-still">📷 Single Photo</button>
</div>

<div class="quick-pills">
  <div class="pill-select">
    <span>Holder:</span>
    <select id="holder"></select>
  </div>
  <div class="pill-select">
    <span>Lens:</span>
    <select id="lens">
      <option value="main">Main 1&times;</option>
      <option value="tele2x">Tele 2&times;</option>
      <option value="tele5x">Tele 5&times;</option>
      <option value="ultrawide">Ultrawide</option>
    </select>
  </div>
  <div class="pill-select">
    <span>Audio:</span>
    <select id="audio-toggle">
      <option value="on">🔊 Synth ON</option>
      <option value="off">🔇 Mute</option>
    </select>
  </div>
</div>

<!-- Live AR Section -->
<div id="section-ar">
  <div id="ar-container">
    <video id="ar-video" playsinline muted autoplay></video>
    <canvas id="ar-canvas"></canvas>
    <div class="ar-hud-overlay">
      <div class="hud-chip"><span class="radar"></span><span id="hud-status">SEARCHING</span></div>
      <div class="hud-verdict" id="hud-verdict">SPRT IDLE</div>
    </div>
  </div>
</div>

<!-- Output Section -->
<div id="out">
  <div class="empty-card" id="empty-state">
    Point the rear camera at one collectible card against a contrasting surface.
    <ul>
      <li><b>Live AR:</b> Continuously tracks 4 edge contours, ray-traces holder refraction, and converges via Sequential Probability Ratio Test (SPRT).</li>
      <li><b>Audio Feedback:</b> Synthesizes sci-fi lock-on chimes when grade boundaries settle.</li>
      <li><b>Mobile Native & Container:</b> Connect any device on LAN to this central instance.</li>
    </ul>
  </div>
</div>

<div class="native-dock">
  <input type="file" id="file" accept="image/*" capture="environment">
  <button type="button" class="btn-round" id="btn-sound-test" title="Sound Synthesizer Test">🎵</button>
  <button type="button" class="btn-round" id="btn-ar-reset" title="Reset AR Tracking">↺</button>
  <button class="btn-primary" id="btn-action">📸 Measure Card</button>
</div>

<!-- Settings Modal -->
<div class="modal-backdrop hidden" id="modal-settings">
  <div class="modal-sheet">
    <div class="modal-title">
      <span>Container Backend Settings</span>
      <span style="cursor:pointer;color:var(--dim)" id="modal-close">&times;</span>
    </div>
    <div class="form-row">
      <label>Backend URL (Container API Host)</label>
      <input type="text" id="setting-api-url" placeholder="http://127.0.0.1:8765 or http://192.168.1.X:8765">
    </div>
    <div class="form-row">
      <label>Tenant Device ID</label>
      <input type="text" id="setting-device-id" readonly>
    </div>
    <button type="button" class="btn-save" id="setting-save">Save & Ping Container</button>
  </div>
</div>

<script>
const $=s=>document.querySelector(s);
let currentMode = 'ar';
let backendUrl = localStorage.getItem('bakugo_backend_url') || window.location.origin;
let deviceId = localStorage.getItem('bakugo_device_id');
if (!deviceId) {
  deviceId = 'dev_' + (crypto.randomUUID ? crypto.randomUUID() : Math.random().toString(36).substring(2, 12));
  localStorage.setItem('bakugo_device_id', deviceId);
}

// Web Audio API Synthesizer (Sci-Fi Audio Cues)
class AudioSynth {
  constructor() {
    this.ctx = null;
    this.enabled = true;
  }
  init() {
    if (!this.ctx) {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (AC) this.ctx = new AC();
    }
    if (this.ctx && this.ctx.state === 'suspended') this.ctx.resume();
  }
  playLock() {
    if (!this.enabled) return;
    this.init();
    if (!this.ctx) return;
    const now = this.ctx.currentTime;
    const osc = this.ctx.createOscillator(), gain = this.ctx.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(880, now);
    osc.frequency.exponentialRampToValueAtTime(1760, now + 0.08);
    gain.gain.setValueAtTime(0.15, now);
    gain.gain.linearRampToValueAtTime(0.01, now + 0.08);
    osc.connect(gain); gain.connect(this.ctx.destination);
    osc.start(now); osc.stop(now + 0.09);
  }
  playSettle() {
    if (!this.enabled) return;
    this.init();
    if (!this.ctx) return;
    const now = this.ctx.currentTime;
    [523.25, 659.25, 783.99, 1046.50].forEach((f, i) => {
      const osc = this.ctx.createOscillator(), gain = this.ctx.createGain();
      osc.type = 'triangle';
      osc.frequency.setValueAtTime(f, now + i * 0.03);
      gain.gain.setValueAtTime(0.12, now + i * 0.03);
      gain.gain.exponentialRampToValueAtTime(0.001, now + i * 0.03 + 0.3);
      osc.connect(gain); gain.connect(this.ctx.destination);
      osc.start(now + i * 0.03); osc.stop(now + i * 0.03 + 0.32);
    });
  }
  playWarn() {
    if (!this.enabled) return;
    this.init();
    if (!this.ctx) return;
    const now = this.ctx.currentTime;
    const osc = this.ctx.createOscillator(), gain = this.ctx.createGain();
    osc.type = 'sawtooth';
    osc.frequency.setValueAtTime(220, now);
    gain.gain.setValueAtTime(0.08, now);
    gain.gain.linearRampToValueAtTime(0.01, now + 0.12);
    osc.connect(gain); gain.connect(this.ctx.destination);
    osc.start(now); osc.stop(now + 0.13);
  }
}
const synth = new AudioSynth();

function haptic(t) {
  if (!navigator.vibrate) return;
  try {
    if (t === 'tap') navigator.vibrate(10);
    else if (t === 'lock') navigator.vibrate(20);
    else if (t === 'settle') navigator.vibrate([30, 40, 50]);
  } catch(e){}
}

// Fetch holders & server config
function refreshConfig() {
  fetch(backendUrl + '/config', { headers: { 'X-Device-ID': deviceId } })
    .then(r => r.json())
    .then(d => {
      if (d.holders) {
        $('#holder').innerHTML = d.holders.map(h =>
          `<option value="${h.id}"${h.id==='raw'?' selected':''}>${h.label}</option>`).join('');
      }
      $('#host-label').textContent = (d.ip ? d.ip : 'Connected') + ' v' + d.version;
    })
    .catch(() => {
      $('#host-label').textContent = 'Offline / Standalone';
    });
}
refreshConfig();

// Tab switching
$('#tab-ar').onclick = () => setMode('ar');
$('#tab-still').onclick = () => setMode('still');
function setMode(m) {
  currentMode = m;
  $('#tab-ar').classList.toggle('active', m === 'ar');
  $('#tab-still').classList.toggle('active', m === 'still');
  $('#section-ar').classList.toggle('hidden', m !== 'ar');
  if (m === 'ar') {
    $('#btn-action').textContent = '⚡ Freeze AR Frame';
    startARStream();
  } else {
    $('#btn-action').textContent = '📸 Measure Photo';
    stopARStream();
  }
}

// Live AR Loop & WebRTC Camera Stream
let videoStream = null, arInterval = null, isPushing = false, lastSettled = false, lastLocked = false;
const arVideo = $('#ar-video'), arCanvas = $('#ar-canvas'), ctx = arCanvas.getContext('2d');
const offscreenCanvas = document.createElement('canvas'), offCtx = offscreenCanvas.getContext('2d');

async function startARStream() {
  if (videoStream) return;
  try {
    synth.init();
    videoStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false
    });
    arVideo.srcObject = videoStream;
    arVideo.onloadedmetadata = () => {
      arCanvas.width = arVideo.videoWidth;
      arCanvas.height = arVideo.videoHeight;
      offscreenCanvas.width = 540;
      offscreenCanvas.height = Math.round(540 * (arVideo.videoHeight / arVideo.videoWidth));
      arInterval = setInterval(arTick, 200);
    };
  } catch(err) {
    $('#hud-status').textContent = 'CAMERA BLOCKED';
    console.warn("Camera streaming unavailable:", err);
  }
}

function stopARStream() {
  if (arInterval) { clearInterval(arInterval); arInterval = null; }
  if (videoStream) {
    videoStream.getTracks().forEach(t => t.stop());
    videoStream = null;
    arVideo.srcObject = null;
  }
  ctx.clearRect(0, 0, arCanvas.width, arCanvas.height);
}

async function arTick() {
  if (isPushing || !arVideo.videoWidth || currentMode !== 'ar') return;
  isPushing = true;
  try {
    offCtx.drawImage(arVideo, 0, 0, offscreenCanvas.width, offscreenCanvas.height);
    const blob = await new Promise(res => offscreenCanvas.toBlob(res, 'image/jpeg', 0.75));
    if (!blob) return;

    const fd = new FormData();
    fd.append('holder', $('#holder').value);
    fd.append('lens', $('#lens').value);
    fd.append('image', blob, 'frame.jpg');

    const res = await fetch(backendUrl + '/ar/push', {
      method: 'POST',
      headers: { 'X-Device-ID': deviceId },
      body: fd
    });
    const d = await res.json();
    drawARHUD(d);
  } catch(e) {
    // Gracefully handle frame drops
  } finally {
    isPushing = false;
  }
}

function drawARHUD(d) {
  ctx.clearRect(0, 0, arCanvas.width, arCanvas.height);
  if (!d || !d.ok) return;

  const scaleX = arCanvas.width / offscreenCanvas.width;
  const scaleY = arCanvas.height / offscreenCanvas.height;

  if (d.tracking && d.quad && d.quad.length === 4) {
    if (!lastLocked) { synth.playLock(); haptic('lock'); lastLocked = true; }
    
    // Draw sci-fi quad
    ctx.strokeStyle = '#4ED2C6';
    ctx.lineWidth = 4;
    ctx.shadowColor = '#4ED2C6';
    ctx.shadowBlur = 12;
    ctx.beginPath();
    ctx.moveTo(d.quad[0][0] * scaleX, d.quad[0][1] * scaleY);
    for (let i = 1; i < 4; i++) {
      ctx.lineTo(d.quad[i][0] * scaleX, d.quad[i][1] * scaleY);
    }
    ctx.closePath();
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Laser crosshairs on corners
    d.quad.forEach(pt => {
      const px = pt[0] * scaleX, py = pt[1] * scaleY;
      ctx.fillStyle = '#FFFFFF';
      ctx.fillRect(px - 4, py - 4, 8, 8);
    });

    $('#hud-status').textContent = (d.ratio ? `${d.ratio.toFixed(1)}%` : 'TRACKING') + (d.grade_ceiling ? ` · ${d.grade_ceiling}` : '');
  } else {
    lastLocked = false;
    $('#hud-status').textContent = 'SEARCHING';
  }

  if (d.settled && !lastSettled) {
    synth.playSettle();
    haptic('settle');
    lastSettled = true;
  } else if (!d.settled) {
    lastSettled = false;
  }

  $('#hud-verdict').textContent = d.verdict ? `SPRT: ${d.verdict}` : (d.settled ? 'SPRT SETTLED' : 'ACCUMULATING');
}

// Action button
$('#btn-action').onclick = () => {
  synth.init();
  haptic('tap');
  if (currentMode === 'still') {
    $('#file').click();
  } else {
    // Freeze current AR frame and run high-res metrology
    if (!arVideo.videoWidth) return;
    const freezeCanvas = document.createElement('canvas');
    freezeCanvas.width = arVideo.videoWidth;
    freezeCanvas.height = arVideo.videoHeight;
    freezeCanvas.getContext('2d').drawImage(arVideo, 0, 0);
    freezeCanvas.toBlob(sendStillPhoto, 'image/jpeg', 0.92);
  }
};

$('#file').onchange = () => { if ($('#file').files[0]) sendStillPhoto($('#file').files[0]); };

function sendStillPhoto(fileOrBlob) {
  $('#btn-action').disabled = true;
  $('#btn-action').textContent = 'Calculating Metrology…';
  $('#out').innerHTML = '<div class="strip"><div class="seg segL" style="width:50%"></div><div class="seg segR" style="width:50%"></div></div><div class="sub-meta" style="padding:14px 16px;color:var(--key)">🔬 Ray-tracing Snell refraction & calculating confidence intervals…</div>';
  
  const fd = new FormData();
  fd.append('holder', $('#holder').value);
  fd.append('lens', $('#lens').value);
  fd.append('image', fileOrBlob, 'shot.jpg');

  fetch(backendUrl + '/measure', {
    method: 'POST',
    headers: { 'X-Device-ID': deviceId },
    body: fd
  })
  .then(r => r.json())
  .then(renderResults)
  .catch(e => {
    $('#out').innerHTML = `<div class="empty-card" style="border-color:var(--stop)"><b style="color:var(--stop)">Connection Error</b><div>${esc(String(e))}</div></div>`;
  })
  .finally(() => {
    $('#btn-action').disabled = false;
    $('#btn-action').textContent = currentMode === 'ar' ? '⚡ Freeze AR Frame' : '📸 Measure Photo';
    $('#file').value = '';
  });
}

function renderResults(d) {
  if (!d.ok) {
    synth.playWarn();
    $('#out').innerHTML = `<div class="empty-card" style="border-color:var(--stop)"><b style="color:var(--stop)">Measurement Refused</b><div>${esc(d.error)}</div></div>`;
    return;
  }
  synth.playSettle();
  haptic('settle');
  const wide = d.ratio, narrow = +(100 - d.ratio).toFixed(1);
  const horiz = d.axis === 'horizontal';
  const a = horiz ? d.borders.left : d.borders.top;
  const b = horiz ? d.borders.right : d.borders.bottom;
  const split = 100 * a / (a + b);
  const ci = Math.max(0.8, d.ratio_hi - d.ratio_lo);

  $('#out').innerHTML = `
  <div class="strip">
    <div class="seg segL" style="width:${split.toFixed(2)}%"></div>
    <div class="seg segR" style="width:${(100-split).toFixed(2)}%"></div>
    <div class="ci" style="left:${(split-ci/2).toFixed(2)}%;width:${ci.toFixed(2)}%"></div>
    <div class="mid"></div><div class="midcap">50/50</div>
    <div class="split" style="left:${split.toFixed(2)}%"></div>
    <div class="tag" style="left:10px">${esc(horiz?'left':'top')} ${a.toFixed(2)}mm</div>
    <div class="tag" style="right:10px">${b.toFixed(2)}mm ${esc(horiz?'right':'bottom')}</div>
  </div>
  <div class="ratio-row"><b>${wide.toFixed(1)}/${narrow.toFixed(1)}</b>
    <span>&plusmn; ${((d.ratio_hi-d.ratio_lo)/2).toFixed(1)}%</span></div>
  <div class="sub-meta">${esc(d.axis)} axis binding &middot; 95% CI ${d.ratio_lo.toFixed(1)}&ndash;${d.ratio_hi.toFixed(1)}%</div>
  <div class="sect-title">Centering Grade Ceiling</div>
  <div class="chips-row">${Object.entries(d.bands).map(([g,b])=>
    `<div class="chip ${String(b.label).includes('10')?'gold':''}"><i>${esc(g)}</i><b>${esc(b.label)}</b></div>`).join('')}</div>
  <table class="mm-table">
    <tr><td>Horizontal Borders (L/R)</td><td>${d.borders.left.toFixed(2)} / ${d.borders.right.toFixed(2)} mm</td></tr>
    <tr><td>Vertical Borders (T/B)</td><td>${d.borders.top.toFixed(2)} / ${d.borders.bottom.toFixed(2)} mm</td></tr>
    <tr><td>Sensor Metric Scale</td><td>${d.px_per_mm} px/mm</td></tr>
    <tr><td>Optical Refraction</td><td>${esc(d.holder)}${d.refraction?' · Snell Corrected':''}</td></tr>
  </table>
  ${d.overlay ? `<img class="ov" alt="Card Metrology" src="data:image/jpeg;base64,${d.overlay}">` : ''}`;
  window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
}

function esc(s) { return String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }

// Settings & Audio buttons
$('#btn-sound-test').onclick = () => { synth.playSettle(); haptic('settle'); };
$('#btn-ar-reset').onclick = () => {
  fetch(backendUrl + '/ar/reset', { method: 'POST', headers: { 'X-Device-ID': deviceId } });
  haptic('tap');
};
$('#audio-toggle').onchange = (e) => { synth.enabled = (e.target.value === 'on'); };

$('#btn-host-settings').onclick = () => {
  $('#setting-api-url').value = backendUrl;
  $('#setting-device-id').value = deviceId;
  $('#modal-settings').classList.remove('hidden');
};
$('#modal-close').onclick = () => { $('#modal-settings').classList.add('hidden'); };
$('#setting-save').onclick = () => {
  backendUrl = $('#setting-api-url').value.trim().replace(/\\/+$/, '');
  localStorage.setItem('bakugo_backend_url', backendUrl);
  $('#modal-settings').classList.add('hidden');
  refreshConfig();
};

// Start AR immediately if supported
if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
  startARStream();
}
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    server_version = "cardcenter"

    def log_message(self, fmt, *args):
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        # Full CORS support for Mobile Native and WebApp clients
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, HEAD")
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type, X-Device-ID, X-Client-ID, Authorization, Range",
        )
        self.send_header(
            "Access-Control-Expose-Headers", "Content-Length, Content-Type, X-Device-ID"
        )
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        """Handle CORS preflight requests for mobile and cross-origin clients."""
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, HEAD")
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type, X-Device-ID, X-Client-ID, Authorization, Range",
        )
        self.send_header("Access-Control-Max-Age", "86400")
        self.end_headers()

    def _extract_device_id(self, fields: Optional[dict] = None) -> str:
        """Extract persistent tenant device_id from headers, query, or body."""
        from urllib.parse import parse_qs, urlparse

        dev = self.headers.get("X-Device-ID") or self.headers.get("X-Client-ID")
        if dev:
            return dev.strip()
        if fields and fields.get("device_id"):
            val = fields["device_id"]
            return (
                val.decode("utf-8", "replace").strip()
                if isinstance(val, bytes)
                else str(val).strip()
            )
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        dev_list = qs.get("device_id")
        if dev_list and dev_list[0]:
            return dev_list[0].strip()
        return "anonymous"

    def do_GET(self) -> None:
        from urllib.parse import urlparse

        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif path == "/manifest.json":
            self._send(200, MANIFEST_JSON.encode(), "application/json")
        elif path == "/health":
            db = os.environ.get("CARDCENTER_DB", "cardcenter.db")
            self._send(
                200,
                json.dumps(
                    {
                        "status": "healthy",
                        "version": __version__,
                        "server": "cardcenter",
                        "db": db,
                        "time": time.time(),
                    }
                ).encode(),
                "application/json",
            )
        elif path == "/slabs":
            # Publish the measured slab metrology, not just the holder labels.
            # The mesh grounds warehouse dimension solving on these physical
            # constants, and a consumer cannot use a thickness it cannot read.
            slabs = [
                {
                    "id": name,
                    "label": name.replace("_", " ").title(),
                    "acrylic_thickness_mm": round(slab.acrylic_thickness_mm, 4),
                    "acrylic_thickness_sigma_mm": round(slab.acrylic_thickness_sigma_mm, 4),
                    "refractive_index": round(slab.refractive_index, 4),
                }
                for name, slab in sorted(SLAB_PRESETS.items())
            ]
            self._send(
                200,
                json.dumps(
                    {"slabs": slabs, "unit": "mm", "version": __version__}
                ).encode(),
                "application/json",
            )
        elif path == "/holders":
            holders = [{"id": "raw", "label": "Raw card"}]
            holders += [
                {"id": k, "label": k.replace("_", " ").title()}
                for k in sorted(SLAB_PRESETS)
                if k not in ("raw", "case_glass")
            ]
            holders += [
                {"id": k, "label": "In case: " + k.replace("case_", "").upper()}
                for k in sorted(SLAB_STACKS)
            ]
            self._send(
                200,
                json.dumps({"holders": holders, "version": __version__}).encode(),
                "application/json",
            )
        elif path == "/config":
            holders = [{"id": "raw", "label": "Raw card"}]
            holders += [
                {"id": k, "label": k.replace("_", " ").title()}
                for k in sorted(SLAB_PRESETS)
                if k not in ("raw", "case_glass")
            ]
            holders += [
                {"id": k, "label": "In case: " + k.replace("case_", "").upper()}
                for k in sorted(SLAB_STACKS)
            ]
            self._send(
                200,
                json.dumps(
                    {
                        "ok": True,
                        "version": __version__,
                        "graders": available_graders(),
                        "holders": holders,
                        "ip": local_ip(),
                        "quipu_enabled": bool(os.environ.get("CARDCENTER_QUIPU_URL")),
                    }
                ).encode(),
                "application/json",
            )
        elif path == "/my-scans":
            device_id = self._extract_device_id()
            db = os.environ.get("CARDCENTER_DB", "cardcenter.db")
            from .store import ScanStore

            with ScanStore(db) as store:
                scans = store.scans_for_device(device_id)
            self._send(
                200,
                json.dumps({"ok": True, "device_id": device_id, "scans": scans}).encode(),
                "application/json",
            )
        elif path == "/my-analytics":
            device_id = self._extract_device_id()
            db = os.environ.get("CARDCENTER_DB", "cardcenter.db")
            try:
                from .analytics import AnalyticsEngine, available as analytics_available

                if analytics_available():
                    with AnalyticsEngine(db) as engine:
                        rows = engine._con.execute(
                            """
                            SELECT
                                COUNT(*)                       AS total_scans,
                                COUNT(DISTINCT phash)          AS distinct_cards,
                                ROUND(AVG(worst_ratio_pct), 2) AS avg_centering,
                                ROUND(MIN(worst_ratio_pct), 2) AS min_centering,
                                ROUND(MAX(worst_ratio_pct), 2) AS max_centering
                            FROM cc.scans
                            WHERE device_id = ?
                        """,
                            [device_id],
                        ).fetchone()
                        res = {
                            "device_id": device_id,
                            "total_scans": rows[0] if rows else 0,
                            "distinct_cards": rows[1] if rows else 0,
                            "avg_centering": rows[2] if rows else None,
                            "min_centering": rows[3] if rows else None,
                            "max_centering": rows[4] if rows else None,
                        }
                else:
                    res = {
                        "device_id": device_id,
                        "total_scans": 0,
                        "note": "analytics unavailable",
                    }
            except Exception as exc:
                res = {"device_id": device_id, "error": str(exc)}
            self._send(200, json.dumps(res).encode(), "application/json")
        elif path == "/quipu":
            try:
                from .quipu_client import enabled, guidance

                payload = {"enabled": enabled()}
                if enabled():
                    g = guidance()
                    payload["guidance"] = {
                        "axis": g.get("axis"),
                        "calibration": g.get("calibration"),
                        "sources": g.get("sources"),
                        "mesh": g.get("mesh"),
                        "numeric_lexicon": (g.get("numeric_lexicon") or [])[:10],
                    }
            except Exception as exc:  # pragma: no cover - observer is optional
                payload = {"enabled": False, "error": str(exc)}
            self._send(200, json.dumps(payload).encode(), "application/json")
        elif path == "/marketplace/assets":
            try:
                from .marketplace_client import get_marketplace_assets
                assets = get_marketplace_assets(axis="touch")
                self._send(200, json.dumps({"ok": True, "assets": assets}).encode(), "application/json")
            except Exception as exc:
                self._send(200, json.dumps({"ok": False, "error": str(exc)}).encode(), "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        from urllib.parse import urlparse

        parsed = urlparse(self.path)
        path = parsed.path

        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length > 0 else b""
            ctype = self.headers.get("Content-Type", "")
            fields = _parse_multipart(body, ctype) if "multipart/form-data" in ctype else {}
            device_id = self._extract_device_id(fields)

            if path == "/measure":
                image = fields.get("image")
                if not image and body and not fields:
                    image = body
                if not image:
                    raise DetectionError("no photo was attached")
                notes = (
                    fields.get("notes", b"").decode("utf-8", "replace").strip()
                    if fields.get("notes")
                    else ""
                )
                payload = _measure_payload(
                    image,
                    fields.get("holder", b"raw").decode("utf-8", "replace"),
                    fields.get("lens", b"main").decode("utf-8", "replace"),
                )
                payload["device_id"] = device_id
                if notes:
                    payload["notes"] = notes
                if payload.get("ok"):
                    payload.update(persist_measure(payload, source="serve"))
                    try:
                        from .quipu_client import observe_measure_async

                        observe_measure_async(payload)
                    except Exception:  # pragma: no cover - observer is optional
                        pass

            elif path == "/ar/push":
                # Real-time streaming AR frame push
                holder = fields.get("holder", b"raw").decode("utf-8", "replace")
                lens = fields.get("lens", b"main").decode("utf-8", "replace")
                image_bytes = fields.get("image") or (body if not fields else None)
                if not image_bytes:
                    raise DetectionError("no frame image provided")

                data = np.frombuffer(image_bytes, dtype=np.uint8)
                frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if frame is None:
                    raise DetectionError("frame could not be decoded")

                session = _get_or_create_ar_session(device_id, holder=holder, lens=lens)
                status: ARStatus = session.push(frame)

                payload = {
                    "ok": True,
                    "tracking": status.tracking,
                    "headline": status.headline(),
                    "quad": status.quad.tolist() if status.quad is not None else None,
                    "guidance": list(status.guidance),
                    "measured_frames": status.measured_frames,
                    "seen_frames": status.seen_frames,
                    "ratio": round(status.ratio.value, 2) if status.ratio else None,
                    "ratio_ci": (
                        [round(x, 2) for x in status.ratio.interval()]
                        if status.ratio
                        else None
                    ),
                    "settled": status.settled,
                    "grade_ceiling": status.grade_ceiling,
                    "bands": status.bands,
                    "verdict": (
                        session.verdict.name
                        if hasattr(session.verdict, "name")
                        else str(session.verdict)
                    ),
                    "scale": round(status.scale.value, 2) if status.scale else None,
                }

            elif path == "/ar/session" or path == "/ar/reset":
                holder = fields.get("holder", b"raw").decode("utf-8", "replace")
                lens = fields.get("lens", b"main").decode("utf-8", "replace")
                session = _get_or_create_ar_session(device_id, holder=holder, lens=lens)
                session.reset()
                payload = {"ok": True, "device_id": device_id, "status": "reset"}

            elif path == "/marketplace/tokenize":
                try:
                    data = json.loads(body.decode("utf-8")) if body else {}
                    from .marketplace_client import tokenize_metrology_scan
                    payload = tokenize_metrology_scan(
                        scan_id=str(data.get("scan_id", f"scan-{int(time.time())}")),
                        title=str(data.get("title", "Centering Metrology Scan")),
                        contributor_wallet=str(data.get("wallet", "0x89A21B7401B5f6d871C7656EC7ab88b098defB75")),
                        user_id=device_id,
                        centering_ratio=float(data.get("ratio", 50.0)),
                        ratio_ci=data.get("ratio_ci", [49.5, 50.5]),
                        cramer_rao_floor_px=float(data.get("cramer_rao_floor_px", 0.045)),
                        grade_ceiling=str(data.get("grade_ceiling", "PSA 10")),
                        holder=str(data.get("holder", "raw")),
                        refraction=bool(data.get("refraction", False)),
                        shard_count=int(data.get("shard_count", 218)),
                        shard_price_base=float(data.get("shard_price_base", 10.0)),
                        metadata=data.get("metadata", {}),
                    )
                except Exception as exc:
                    payload = {"ok": False, "error": str(exc)}

            else:
                self._send(404, b"not found", "text/plain")
                return

        except DetectionError as exc:
            payload = {"ok": False, "error": str(exc)}
        except Exception as exc:  # pragma: no cover
            traceback.print_exc()
            payload = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

        self._send(200, json.dumps(payload).encode(), "application/json")


def local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


def serve(host: str = "0.0.0.0", port: int = 8765) -> None:
    if not os.environ.get("CARDCENTER_DB"):
        os.environ["CARDCENTER_DB"] = "cardcenter.db"
    httpd = ThreadingHTTPServer((host, port), Handler)
    print()
    print("  ========================================================")
    print("  ⚡ Bakugo AR Metrology Server running")
    print(f"     Version     : v{__version__}")
    print(f"     Local UI    : http://127.0.0.1:{port}")
    print(f"     LAN Mobile  : http://{local_ip()}:{port}")
    print(f"     Storage DB  : {os.environ['CARDCENTER_DB']}")
    print("  ========================================================")
    print()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped.\n")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    p = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    serve(port=p)
