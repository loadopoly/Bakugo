"""A local web UI, so a phone can be the front end and Python stays the engine.

Rationale for this shape rather than an APK: the measurement code is Python,
NumPy and OpenCV. Packaging that into an Android app means python-for-android or
Chaquopy, a toolchain, a build, and signing -- and it would still be the same
code underneath. Serving a page from localhost gets the same result in about
thirty seconds, works entirely offline, and updates when you `git pull`.

Everything here is standard library. No Flask, no npm, nothing to install beyond
what the measurement already needs. That matters because every dependency is
another thing that can fail to build on a phone.

The page uses `<input type="file" accept="image/*" capture="environment">`, which
on Android Chrome opens the rear camera directly rather than a file picker. That
one attribute is the difference between a web page and something usable at a
counter.
"""

from __future__ import annotations

import base64
import json
import socket
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

import cv2
import numpy as np

import os

from .capture import assess_frame
from .centering import measure_centering
from .grading import available_graders, grade_band, predict_overall_grade
from .learning import maybe_load_grade_model
from .render import annotate
from .types import SLAB_PRESETS, SLAB_STACKS, CaptureSpec, DetectionError, resolve_holder

# Typical horizontal fields of view. Phone cameras vary, but within a few
# degrees these are close enough that the tilt recovery works; the alternative
# is asking the user for a number they do not have.
LENS_FOV = {
    "main": 68.0,
    "ultrawide": 105.0,
    "tele2x": 38.0,
    "tele5x": 16.0,
}


def _parse_multipart(body: bytes, content_type: str) -> dict[str, bytes]:
    """Minimal multipart/form-data parser.

    Written by hand because `cgi` was removed in Python 3.13 and a phone may
    well have it. Handles exactly what this form sends: a few text fields and
    one file.
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

    # Large phone photos cost time and buy nothing past a point; the card only
    # needs enough pixels per millimetre, not the whole 50MP sensor.
    max_side = 2400
    if max(image.shape[:2]) > max_side:
        scale = max_side / max(image.shape[:2])
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    fov = LENS_FOV.get(lens, LENS_FOV["main"])
    capture = CaptureSpec.from_fov(fov, image.shape)
    result = measure_centering(image, slab=resolve_holder(holder), capture=capture)

    bands = {g: grade_band(result.worst_ratio, g, "front") for g in available_graders()}
    quality = assess_frame(image, result.corners_px, px_per_mm=result.px_per_mm)

    # Card only, without annotate()'s side panel: at phone width that panel
    # renders at about 6px type, and everything in it is already in the table
    # above. The picture's job here is to let you check the detected border by
    # eye, so give the whole width to the card.
    overlay = annotate(result, bands)[:, : result.rectified.shape[1]]
    ok, buf = cv2.imencode(".jpg", overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
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
    """Write a successful measure locally, then best-effort cloud upsert.

    Failures here never fail the measurement. Local ScanStore is source of
    truth; Supabase is a mirror of metadata only (no photo). Pyodide skips
    the urllib hop — the Pages app posts metadata from JavaScript instead.
    """
    extra: dict = {}
    db = os.environ.get("CARDCENTER_DB")
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


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover, maximum-scale=1, user-scalable=no">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="theme-color" content="#0B0F15">
<title>Bakugo — CardCenter Metrology</title>
<style>
:root{
  --ink:#0B0F15; --surface:#121822; --surface-glass:rgba(18,24,34,0.85);
  --rise:#1D2635; --rule:#2B384A; --paper:#F3EFE6; --dim:#8E9EAF;
  --key:#4ED2C6; --key-glow:rgba(78,210,198,0.25);
  --pass:#4EBA82; --hold:#D9A83A; --stop:#D45440;
  --radius-lg:16px; --radius-md:12px; --radius-sm:8px;
}
*{box-sizing:border-box;-webkit-tap-highlight-color:transparent;user-select:none}
input,select,textarea{user-select:auto}
html,body{margin:0;padding:0;background:var(--ink);color:var(--paper);
  font-family:-apple-system,BlinkMacSystemFont,"SF Pro Text","Segoe UI",Roboto,sans-serif;
  height:100%;overflow-x:hidden;overscroll-behavior-y:contain}
header.app-bar{padding:calc(10px + env(safe-area-inset-top)) 16px 10px;
  display:flex;align-items:center;gap:12px;position:sticky;top:0;
  background:rgba(11,15,21,0.85);backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);
  border-bottom:1px solid var(--rule);z-index:20}
.brand{display:flex;align-items:baseline;gap:8px}
.brand h1{font-size:17px;margin:0;font-weight:700;color:#FFF}
.brand .badge{font-size:10px;letter-spacing:.15em;text-transform:uppercase;color:var(--key);font-weight:700;padding:2px 6px;background:rgba(78,210,198,.12);border-radius:4px}
header .v{margin-left:auto;font-size:10px;color:var(--dim);font-family:ui-monospace,"SF Mono",monospace}

.quick-pills{display:flex;gap:8px;padding:10px 16px 0;overflow-x:auto;scrollbar-width:none}
.quick-pills::-webkit-scrollbar{display:none}
.pill-select{background:var(--surface-glass);border:1px solid var(--rule);color:var(--paper);border-radius:20px;padding:6px 12px;font-size:12px;font-weight:500;display:flex;align-items:center;gap:6px;white-space:nowrap}
.pill-select select{background:transparent;border:none;color:inherit;font-size:inherit;font-weight:inherit;outline:none;cursor:pointer}

.voice-bubble{margin:10px 16px 0;background:rgba(22,30,42,.92);border:1px solid var(--key);border-radius:var(--radius-md);padding:8px 12px;font-size:12px;color:#FFF;display:flex;align-items:center;gap:8px}
.voice-bubble.hidden{display:none}
.voice-bubble .mic-pulse{width:8px;height:8px;border-radius:50%;background:var(--key);box-shadow:0 0 8px var(--key);animation:p 1s infinite alternate}
@keyframes p{from{opacity:.4;transform:scale(.9)}to{opacity:1;transform:scale(1.15)}}

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
.chip.gold{border-color:var(--key);background:rgba(78,210,198,.08)}

table.mm-table{width:calc(100% - 32px);margin:8px 16px 0;border-collapse:collapse;font-family:ui-monospace,"SF Mono",monospace;font-size:12px}
table.mm-table td{padding:6px 0;border-top:1px solid var(--rule);color:var(--dim)}
table.mm-table td:last-child{text-align:right;color:var(--paper);font-variant-numeric:tabular-nums}

img.ov{width:calc(100% - 32px);max-height:44vh;object-fit:contain;margin:12px 16px 0;border-radius:var(--radius-md);border:1px solid var(--rule);display:block;background:var(--surface)}
.voice-tag-chip{background:rgba(78,210,198,.12);border:1px solid var(--key);color:var(--paper);border-radius:var(--radius-sm);padding:6px 10px;font-size:12px;margin:0 16px 8px}

.native-dock{position:fixed;left:0;right:0;bottom:0;padding:10px 16px calc(12px + env(safe-area-inset-bottom));background:linear-gradient(to top,var(--ink) 75%,rgba(11,15,21,0));display:flex;align-items:center;gap:10px;z-index:25}
.btn-primary{flex:1;border:none;border-radius:var(--radius-md);padding:16px;font-size:15px;font-weight:700;background:var(--key);color:#061116;font-family:inherit;cursor:pointer;box-shadow:0 4px 14px var(--key-glow)}
.btn-primary:active{transform:scale(.98)}
.btn-primary[disabled]{opacity:.45}
.btn-round{width:50px;height:50px;border-radius:var(--radius-md);border:1px solid var(--rule);background:var(--surface);color:var(--paper);display:flex;align-items:center;justify-content:center;font-size:18px;cursor:pointer}
.btn-round:active{transform:scale(.94)}
.btn-round.mic-active{background:rgba(78,210,198,.2);border-color:var(--key);color:var(--key)}
.spin{padding:24px 16px;color:var(--key);font-size:13.5px}
input[type=file]{position:absolute;width:1px;height:1px;opacity:0;pointer-events:none}
</style>
</head>
<body>
<header class="app-bar">
  <div class="brand">
    <h1>Bakugo</h1>
    <span class="badge">Metrology</span>
  </div>
  <span class="v" id="ver"></span>
</header>

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
</div>

<div class="voice-bubble hidden" id="voice-bubble">
  <span class="mic-pulse"></span>
  <span id="voice-text">Listening for condition notes…</span>
</div>

<div id="out">
  <div class="empty-card">
    Fill the frame with one card, all four edges visible against a plain background. Square up to it.
    <ul>
      <li>Shooting into a case? Pick a <b>case</b> holder — the dielectric glass bends light.</li>
      <li>Zoom to 2&times; if you can. It measures tighter.</li>
      <li>Tap 🎙️ to narrate condition notes (Scanlily-style) before taking the shot.</li>
    </ul>
  </div>
</div>

<div class="native-dock">
  <input type="file" id="file" accept="image/*" capture="environment">
  <button type="button" class="btn-round" id="btn-mic" title="Voice Condition Tagging">🎙️</button>
  <button class="btn-primary" id="shoot">Measure a card</button>
</div>

<script>
const $=s=>document.querySelector(s);
const out=$('#out'), file=$('#file'), shoot=$('#shoot');

function haptic(t){
  if(!navigator.vibrate) return;
  try{
    if(t==='tap') navigator.vibrate(10);
    else if(t==='settle') navigator.vibrate([25,35,25]);
    else if(t==='voice') navigator.vibrate([15,15]);
  }catch(e){}
}

let deviceId = localStorage.getItem('bakugo_device_id');
if (!deviceId) {
  deviceId = 'dev_' + (crypto.randomUUID ? crypto.randomUUID() : Math.random().toString(36).substring(2, 12));
  localStorage.setItem('bakugo_device_id', deviceId);
}

fetch('/holders', {headers: {'X-Device-ID': deviceId}}).then(r=>r.json()).then(d=>{
  $('#holder').innerHTML=d.holders.map(h=>
    `<option value="${h.id}"${h.id==='raw'?' selected':''}>${h.label}</option>`).join('');
  $('#ver').textContent='v'+d.version;
});

// Voice dictation
let voiceActive = false, speechRecognizer = null, activeVoiceNotes = '';
const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
if(SpeechRec){
  speechRecognizer = new SpeechRec();
  speechRecognizer.continuous = true;
  speechRecognizer.interimResults = true;
  speechRecognizer.lang = 'en-US';
  speechRecognizer.onresult = (e)=>{
    let t = '';
    for(let i=e.resultIndex; i<e.results.length; ++i) t += e.results[i][0].transcript;
    activeVoiceNotes = t.trim();
    $('#voice-text').textContent = '🎙️ "' + activeVoiceNotes + '"';
    haptic('voice');
  };
  speechRecognizer.onend = ()=>{ if(voiceActive) speechRecognizer.start(); };
  $('#btn-mic').onclick = ()=>{
    haptic('tap');
    voiceActive = !voiceActive;
    $('#btn-mic').classList.toggle('mic-active', voiceActive);
    $('#voice-bubble').classList.toggle('hidden', !voiceActive);
    if(voiceActive) try{ speechRecognizer.start(); }catch(e){}
    else if(speechRecognizer) try{ speechRecognizer.stop(); }catch(e){}
  };
} else {
  $('#btn-mic').style.display = 'none';
}

shoot.onclick=()=>{ haptic('tap'); file.click(); };
file.onchange=()=>{ if(file.files[0]) send(file.files[0]); };

function send(f){
  shoot.disabled=true; shoot.textContent='Measuring Borders…';
  out.innerHTML='<div class="spin">🔬 Ray-tracing dielectric refraction & calculating border ratios…</div>';
  const fd=new FormData();
  fd.append('holder',$('#holder').value);
  fd.append('lens',$('#lens').value);
  fd.append('device_id', deviceId);
  if(activeVoiceNotes) fd.append('notes', activeVoiceNotes);
  fd.append('image',f,'card.jpg');
  fetch('/measure',{method:'POST',headers:{'X-Device-ID': deviceId},body:fd})
    .then(r=>r.json()).then(render)
    .catch(e=>fail('Engine unreachable',String(e)))
    .finally(()=>{shoot.disabled=false;shoot.textContent='Measure a card';file.value='';});
}

function fail(t,m){
  out.innerHTML=`<div class="empty-card" style="border-color:var(--stop)"><b style="color:var(--stop)">${esc(t)}</b><div>${esc(m)}</div></div>`;
}
function esc(s){return String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}

function render(d){
  if(!d.ok){ fail('Not measured', d.error); return; }
  haptic('settle');
  const wide=d.ratio, narrow=+(100-d.ratio).toFixed(1);
  const horiz = d.axis==='horizontal';
  const a = horiz? d.borders.left : d.borders.top;
  const b = horiz? d.borders.right : d.borders.bottom;
  const split = 100*a/(a+b);
  const ci = Math.max(0.8, d.ratio_hi-d.ratio_lo);

  out.innerHTML=`
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
  ${activeVoiceNotes?`<div class="voice-tag-chip">🎙️ <b>Spoken Note:</b> ${esc(activeVoiceNotes)}</div>`:''}
  <div class="sect-title">Centering Grade Ceiling</div>
  <div class="chips-row">${Object.entries(d.bands).map(([g,b])=>
    `<div class="chip ${String(b.label).includes('10')?'gold':''}"><i>${esc(g)}</i><b>${esc(b.label)}</b></div>`).join('')}</div>
  <table class="mm-table">
    <tr><td>Horizontal Borders (L/R)</td><td>${d.borders.left.toFixed(2)} / ${d.borders.right.toFixed(2)} mm</td></tr>
    <tr><td>Vertical Borders (T/B)</td><td>${d.borders.top.toFixed(2)} / ${d.borders.bottom.toFixed(2)} mm</td></tr>
    <tr><td>Sensor Metric Scale</td><td>${d.px_per_mm} px/mm</td></tr>
    <tr><td>Optical Refraction</td><td>${esc(d.holder)}${d.refraction?' · Snell Corrected':''}</td></tr>
  </table>
  ${d.overlay?`<img class="ov" alt="Card Metrology" src="data:image/jpeg;base64,${d.overlay}">`:''}
  <div class="sub-meta" style="padding-top:14px">Centering only. Corners, edges and surface decide final grade.</div>`;
  window.scrollTo({top:0,behavior:'smooth'});
}
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    server_version = "cardcenter"

    def log_message(self, fmt, *args):  # keep the terminal readable on a phone
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _extract_device_id(self, fields: Optional[dict] = None) -> str:
        """Extract persistent tenant device_id from headers, query, or body."""
        from urllib.parse import parse_qs, urlparse
        dev = self.headers.get("X-Device-ID") or self.headers.get("X-Client-ID")
        if dev:
            return dev.strip()
        if fields and fields.get("device_id"):
            val = fields["device_id"]
            return val.decode("utf-8", "replace").strip() if isinstance(val, bytes) else str(val).strip()
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        dev_list = qs.get("device_id")
        if dev_list and dev_list[0]:
            return dev_list[0].strip()
        return "anonymous"

    def do_GET(self) -> None:
        from urllib.parse import parse_qs, urlparse
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif path == "/holders":
            from . import __version__

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
        elif path == "/my-scans":
            # Multi-tenant isolated scans: external user can ONLY see their own scans
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
            # Multi-tenant isolated analytics: aggregated over caller's device_id
            device_id = self._extract_device_id()
            db = os.environ.get("CARDCENTER_DB", "cardcenter.db")
            try:
                from .analytics import AnalyticsEngine, available as analytics_available
                if analytics_available():
                    with AnalyticsEngine(db) as engine:
                        # Scoped to caller's device_id
                        rows = engine._con.execute("""
                            SELECT
                                COUNT(*)                       AS total_scans,
                                COUNT(DISTINCT phash)          AS distinct_cards,
                                ROUND(AVG(worst_ratio_pct), 2) AS avg_centering,
                                ROUND(MIN(worst_ratio_pct), 2) AS min_centering,
                                ROUND(MAX(worst_ratio_pct), 2) AS max_centering
                            FROM cc.scans
                            WHERE device_id = ?
                        """, [device_id]).fetchone()
                        res = {
                            "device_id": device_id,
                            "total_scans": rows[0] if rows else 0,
                            "distinct_cards": rows[1] if rows else 0,
                            "avg_centering": rows[2] if rows else None,
                            "min_centering": rows[3] if rows else None,
                            "max_centering": rows[4] if rows else None,
                        }
                else:
                    res = {"device_id": device_id, "total_scans": 0, "note": "analytics unavailable"}
            except Exception as exc:
                res = {"device_id": device_id, "error": str(exc)}
            self._send(200, json.dumps(res).encode(), "application/json")
        elif path == "/quipu":
            # The Observer link: what Bakugo feeds up and receives back.
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
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        from urllib.parse import urlparse
        parsed = urlparse(self.path)
        if parsed.path != "/measure":
            self._send(404, b"not found", "text/plain")
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            fields = _parse_multipart(body, self.headers.get("Content-Type", ""))
            image = fields.get("image")
            if not image:
                raise DetectionError("no photo was attached")
            notes = fields.get("notes", b"").decode("utf-8", "replace").strip() if fields.get("notes") else ""
            payload = _measure_payload(
                image,
                fields.get("holder", b"raw").decode(),
                fields.get("lens", b"main").decode(),
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
        except DetectionError as exc:
            payload = {"ok": False, "error": str(exc)}
        except Exception as exc:  # pragma: no cover - surfaced to the phone
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


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    if not os.environ.get("CARDCENTER_DB"):
        os.environ["CARDCENTER_DB"] = "cardcenter.db"
    httpd = ThreadingHTTPServer((host, port), Handler)
    print()
    print("  cardcenter is running.")
    print()
    print(f"    on this phone : http://127.0.0.1:{port}")
    if host != "127.0.0.1":
        print(f"    on your wifi  : http://{local_ip()}:{port}")
    print()
    print(f"  scans saved to  : {os.environ['CARDCENTER_DB']}")
    print("  Open that in Chrome. Ctrl+C here to stop.")
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
