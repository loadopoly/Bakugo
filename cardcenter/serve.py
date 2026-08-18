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
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#10151C">
<title>Centering</title>
<style>
:root{
  --ink:#10151C; --glass:#1A2330; --rise:#222E3D; --rule:#31404F;
  --paper:#E9E5DB; --dim:#8797A8; --key:#5BC0C8;
  --pass:#57A97A; --hold:#C9A03A; --stop:#B4553F;
}
*{box-sizing:border-box;-webkit-tap-highlight-color:transparent}
html,body{margin:0;padding:0;background:var(--ink);color:var(--paper);
  font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  overscroll-behavior-y:contain}
body{padding-bottom:calc(96px + env(safe-area-inset-bottom))}
.eyebrow{font-size:10px;letter-spacing:.22em;text-transform:uppercase;
  color:var(--dim);font-weight:600}
header{padding:14px 16px 10px;border-bottom:1px solid var(--rule);
  display:flex;align-items:baseline;gap:10px;position:sticky;top:0;
  background:var(--ink);z-index:5}
header h1{font-size:15px;margin:0;letter-spacing:.02em;font-weight:650}
header .v{margin-left:auto;font-size:10px;color:var(--dim);
  font-family:ui-monospace,"Roboto Mono",monospace}
.controls{display:grid;grid-template-columns:1fr 1fr;gap:8px;padding:12px 16px}
label.field{display:block}
label.field .eyebrow{display:block;margin-bottom:5px}
select{width:100%;appearance:none;background:var(--glass);color:var(--paper);
  border:1px solid var(--rule);border-radius:9px;padding:11px 12px;font-size:15px;
  font-family:inherit}
select:focus-visible{outline:2px solid var(--key);outline-offset:1px}

/* Signature: the two opposing borders drawn AS the ratio, meeting at a split.
   Drawing the card at true scale looks correct and reads as nothing -- borders
   are ~4% of a card's width, so the asymmetry that decides the grade is four
   pixels wide. Scaling the two borders against each other instead puts the
   measured quantity on screen at full size, and the distance from the split to
   the centre line IS the miscentering. */
.strip{margin:8px 16px 0;height:56px;position:relative;border-radius:10px;
  background:var(--rise);border:1px solid var(--rule);overflow:hidden}
.strip .seg{position:absolute;top:0;bottom:0}
.strip .segL{left:0;background:linear-gradient(90deg,#243244,#2C3D52)}
.strip .segR{right:0;background:linear-gradient(270deg,#243244,#2C3D52)}
.strip .split{position:absolute;top:0;bottom:0;width:2px;background:var(--key);
  box-shadow:0 0 0 1px rgba(16,21,28,.7)}
.strip .ci{position:absolute;top:0;bottom:0;background:var(--key);opacity:.22}
.strip .mid{position:absolute;top:0;bottom:0;left:50%;width:1px;
  background:var(--paper);opacity:.35}
.strip .midcap{position:absolute;top:4px;left:50%;transform:translateX(-50%);
  font-size:8px;letter-spacing:.16em;color:var(--dim);text-transform:uppercase;
  background:var(--rise);padding:0 4px;border-radius:3px}
.strip .tag{position:absolute;bottom:6px;font-size:9.5px;letter-spacing:.12em;
  color:#AFC0CF;text-transform:uppercase;
  font-family:ui-monospace,"Roboto Mono",monospace}

.ratio{display:flex;align-items:baseline;gap:10px;padding:14px 16px 2px}
.ratio b{font-size:46px;line-height:.9;font-weight:700;letter-spacing:-.03em;
  font-variant-numeric:tabular-nums;font-family:ui-monospace,"Roboto Mono",monospace}
.ratio span{font-size:13px;color:var(--dim);font-variant-numeric:tabular-nums;
  font-family:ui-monospace,"Roboto Mono",monospace}
.sub{padding:0 16px 12px;font-size:12px;color:var(--dim)}

.grades{display:flex;gap:6px;padding:0 16px 4px;flex-wrap:wrap}
.chip{background:var(--glass);border:1px solid var(--rule);border-radius:7px;
  padding:7px 10px;font-size:12px;font-family:ui-monospace,"Roboto Mono",monospace}
.chip i{font-style:normal;color:var(--dim);margin-right:6px;font-size:10px;
  letter-spacing:.1em}

table.mm{width:calc(100% - 32px);margin:10px 16px 0;border-collapse:collapse;
  font-family:ui-monospace,"Roboto Mono",monospace;font-size:12px}
table.mm td{padding:6px 0;border-top:1px solid var(--rule);color:var(--dim)}
table.mm td:last-child{text-align:right;color:var(--paper);
  font-variant-numeric:tabular-nums}

.notes{margin:12px 16px 0;padding:0;list-style:none}
.notes li{border-left:2px solid var(--hold);padding:7px 0 7px 10px;
  margin-bottom:7px;font-size:12.5px;line-height:1.45;color:#CBD5DF}
.err{margin:16px;padding:14px;border:1px solid var(--stop);border-radius:10px;
  background:rgba(180,85,63,.1);font-size:13.5px;line-height:1.5}
.err b{display:block;margin-bottom:5px;font-size:11px;letter-spacing:.16em;
  text-transform:uppercase;color:var(--stop)}
/* A portrait card at full phone width is ~950px tall and buries everything
   under it. Cap the height: the picture's job is a glance-check that the
   detector found the right edge, not a viewing experience. */
img.ov{width:calc(100% - 32px);max-height:46vh;object-fit:contain;
  object-position:top;margin:12px 16px 0;border-radius:10px;
  border:1px solid var(--rule);display:block;background:var(--glass)}

.dock{position:fixed;left:0;right:0;bottom:0;padding:12px 16px
  calc(12px + env(safe-area-inset-bottom));background:linear-gradient(
  to top,var(--ink) 62%,rgba(16,21,28,0));z-index:6}
.shoot{width:100%;border:0;border-radius:12px;padding:17px;font-size:16px;
  font-weight:650;background:var(--key);color:#08131A;font-family:inherit;
  letter-spacing:.01em}
.shoot:active{transform:translateY(1px)}
.shoot[disabled]{opacity:.5}
.shoot:focus-visible{outline:2px solid var(--paper);outline-offset:2px}
input[type=file]{position:absolute;width:1px;height:1px;opacity:0;pointer-events:none}
.empty{padding:26px 16px;color:var(--dim);font-size:13.5px;line-height:1.6}
.empty ul{padding-left:16px;margin:9px 0 0}
.empty li{margin-bottom:5px}
.spin{padding:26px 16px;color:var(--key);font-size:13.5px}
@media (prefers-reduced-motion:no-preference){
  .spin::after{content:"";display:inline-block;width:6px;height:6px;
    margin-left:6px;border-radius:50%;background:var(--key);
    animation:p 1s ease-in-out infinite}
  @keyframes p{0%,100%{opacity:.25}50%{opacity:1}}
}
</style>
</head>
<body>
<header>
  <h1>Centering</h1>
  <span class="v" id="ver"></span>
</header>

<div class="controls">
  <label class="field"><span class="eyebrow">Holder</span>
    <select id="holder"></select></label>
  <label class="field"><span class="eyebrow">Lens</span>
    <select id="lens">
      <option value="main">Main 1&times;</option>
      <option value="tele2x">Tele 2&times;</option>
      <option value="tele5x">Tele 5&times;</option>
      <option value="ultrawide">Ultrawide</option>
    </select></label>
</div>

<div id="out">
  <div class="empty">
    Fill the frame with one card, all four edges visible, against a plain
    background. Square up to it.
    <ul>
      <li>Shooting into a case? Pick a <b>case</b> holder — the glass bends the image.</li>
      <li>Glare is the usual reason a shot gets refused. Shade the case with your body.</li>
      <li>Zoom to 2&times; if you can. It measures tighter.</li>
    </ul>
  </div>
</div>

<div class="dock">
  <input type="file" id="file" accept="image/*" capture="environment">
  <button class="shoot" id="shoot">Measure a card</button>
</div>

<script>
const $=s=>document.querySelector(s);
const out=$('#out'), file=$('#file'), shoot=$('#shoot');

fetch('/holders').then(r=>r.json()).then(d=>{
  $('#holder').innerHTML=d.holders.map(h=>
    `<option value="${h.id}"${h.id==='raw'?' selected':''}>${h.label}</option>`).join('');
  $('#ver').textContent='v'+d.version;
});

shoot.onclick=()=>file.click();
file.onchange=()=>{ if(file.files[0]) send(file.files[0]); };

function send(f){
  shoot.disabled=true; shoot.textContent='Measuring…';
  out.innerHTML='<div class="spin">Reading the borders</div>';
  const fd=new FormData();
  fd.append('holder',$('#holder').value);
  fd.append('lens',$('#lens').value);
  fd.append('image',f,'card.jpg');
  fetch('/measure',{method:'POST',body:fd})
    .then(r=>r.json()).then(render)
    .catch(e=>fail('Could not reach the engine',String(e)))
    .finally(()=>{shoot.disabled=false;shoot.textContent='Measure a card';file.value='';});
}

function fail(t,m){
  out.innerHTML=`<div class="err"><b>${esc(t)}</b>${esc(m)}</div>`;
}
function esc(s){return String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}

function render(d){
  if(!d.ok){ fail('Not measured', d.error); return; }
  const wide=d.ratio, narrow=+(100-d.ratio).toFixed(1);
  const horiz = d.axis==='horizontal';
  const a = horiz? d.borders.left : d.borders.top;
  const b = horiz? d.borders.right : d.borders.bottom;
  // Split point = this border's share of the two. That is the ratio itself.
  const split = 100*a/(a+b);
  const ci = Math.max(0.8, d.ratio_hi-d.ratio_lo);

  out.innerHTML=`
  <div class="strip">
    <div class="seg segL" style="width:${split.toFixed(2)}%"></div>
    <div class="seg segR" style="width:${(100-split).toFixed(2)}%"></div>
    <div class="ci" style="left:${(split-ci/2).toFixed(2)}%;width:${ci.toFixed(2)}%"></div>
    <div class="mid"></div><div class="midcap">even</div>
    <div class="split" style="left:${split.toFixed(2)}%"></div>
    <div class="tag" style="left:10px">${esc(horiz?'left':'top')} ${a.toFixed(2)}mm</div>
    <div class="tag" style="right:10px">${b.toFixed(2)}mm ${esc(horiz?'right':'bottom')}</div>
  </div>
  <div class="ratio"><b>${wide.toFixed(1)}/${narrow.toFixed(1)}</b>
    <span>&plusmn; ${((d.ratio_hi-d.ratio_lo)/2).toFixed(1)}</span></div>
  <div class="sub">${esc(d.axis)} axis is worse &middot; wider on ${esc(d.wider)}
    &middot; 95% CI ${d.ratio_lo.toFixed(1)}&ndash;${d.ratio_hi.toFixed(1)}</div>
  <div class="grades">${Object.entries(d.bands).map(([g,b])=>
    `<div class="chip"><i>${esc(g)}</i>${esc(b.label)}</div>`).join('')}</div>
  <table class="mm">
    <tr><td>left / right</td><td>${d.borders.left.toFixed(2)} / ${d.borders.right.toFixed(2)} mm</td></tr>
    <tr><td>top / bottom</td><td>${d.borders.top.toFixed(2)} / ${d.borders.bottom.toFixed(2)} mm</td></tr>
    <tr><td>scale</td><td>${d.px_per_mm} px/mm</td></tr>
    <tr><td>holder</td><td>${esc(d.holder)}${d.refraction?' · corrected':''}</td></tr>
  </table>
  ${d.warnings.length?`<ul class="notes">${d.warnings.map(w=>`<li>${esc(w)}</li>`).join('')}</ul>`:''}
  ${d.overlay?`<img class="ov" alt="Detected border drawn over the card"
     src="data:image/jpeg;base64,${d.overlay}">`:''}
  <div class="sub" style="padding-top:14px">Centering only. This caps a grade;
    it never confirms one. Corners, edges and surface are not measured.</div>`;
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

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif self.path == "/holders":
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
        elif self.path == "/quipu":
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
        if self.path != "/measure":
            self._send(404, b"not found", "text/plain")
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            fields = _parse_multipart(body, self.headers.get("Content-Type", ""))
            image = fields.get("image")
            if not image:
                raise DetectionError("no photo was attached")
            payload = _measure_payload(
                image,
                fields.get("holder", b"raw").decode(),
                fields.get("lens", b"main").decode(),
            )
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
