"""Perceptopoly: the AR capture front end.

WHAT THIS IS
------------
A live page you point at a card. It runs the triage router every frame, tells
you what is wrong with the shot while you can still fix it, and measures only
when the frame is worth measuring. It is built on what the real session and the
certified PSA card actually showed, not on what the synthetic tests suggested.

THREE FINDINGS FROM REAL DATA THAT SHAPE THE UI
------------------------------------------------
1. STANDOFF IS THE DOMINANT PROBLEM. Across 201 real photographs the median card
   filled 3.6% of the frame -- about 3 px/mm, below the usable floor -- and 23%
   had the card cropped by a frame edge. Neither is a software problem. Both are
   fixable in the moment if the tool says so while the user is still holding the
   phone, which is the entire argument for an AR view over a gallery importer.

2. MOST CARDS ARE NOT MEASURABLE, AND THAT IS CORRECT. A certified PSA GEM MT 10
   (cert 143341329, 2024 TWM #225 Rescue Board, Hyper Rare) was refused on all
   three photographs. Inspecting the rectified crop shows why: the detector
   found the card correctly -- 1.446 aspect, 1.44 px line-fit residual -- and the
   card is full-bleed gold foil with no printed border anywhere. There is no
   border-centering to measure. Energy cards, full-arts, and Hyper Rares are all
   in this class. The UI has to present that as a category, not a failure, or
   users will assume the tool is broken on their most valuable cards.

3. A SLABBED CARD NEEDS A DIFFERENT MEASUREMENT ENTIRELY. Graders assess a
   slabbed card's centering against the card edges, but a full-bleed card inside
   a slab offers only the slab window as a reference, and the slab window is not
   registered to the card. This tool does not do that measurement and says so
   rather than substituting the card outline for it.

WHAT THIS DELIBERATELY DOES NOT DO
-----------------------------------
No grade is emitted. No price. Nothing here connects to a wallet, a contract, or
a token multiplier. Centering caps a grade; it never confirms one, and corners,
edges and surface -- the other three attributes -- are not measurable at phone
resolution through a slab.
"""

from __future__ import annotations

import base64
import json
import socket
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

import cv2
import numpy as np

from .centering import measure_centering
from .grading import available_graders, grade_band, predict_overall_grade
from .learning import maybe_load_grade_model
from .serve import LENS_FOV, _parse_multipart
from .triage import Route, triage
from .types import CaptureSpec, DetectionError, resolve_holder

# Below this the card is too small in frame for border location to be reliable.
GOOD_PX_PER_MM = 9.0


def analyse(image_bytes: bytes, holder: str, lens: str) -> dict:
    """Triage first, measure only if the frame earns it."""
    data = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise DetectionError("that file could not be read as an image")

    # 1600 px is where the detector actually works. Measured: at 2400 px the
    # candidate stage returns zero quads on real photographs that succeed
    # cleanly at 1600. Not yet diagnosed; capped here rather than silently
    # failing on high-resolution input.
    if max(image.shape[:2]) > 1600:
        s = 1600 / max(image.shape[:2])
        image = cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

    t = triage("frame", image=image)
    out: dict = {
        "ok": True,
        "route": t.route.value,
        "reason": t.reason,
        "advice": t.advice,
        "px_per_mm": round(t.px_per_mm, 1) if t.px_per_mm else None,
        "caliper": t.caliper_present,
        "measured": False,
    }

    # Capability is assessed even for frames triage would not route to
    # measurement. A slab photographed close-up genuinely touches the frame
    # edge -- triage is right to call that RESHOOT_CROPPED -- but the CARD
    # inside the slab may still be complete and measurable. Returning early
    # discarded a reading (54.3/45.7 on a certified PSA 10) that the pipeline
    # was perfectly capable of producing.
    from .capability import Capability, assess as assess_capability

    cap = assess_capability(image)
    out["capability"] = cap.capability.value
    out["available"] = list(cap.available)
    out["unavailable"] = list(cap.unavailable)
    if cap.geometry is not None:
        out["geometry"] = {
            "squareness_assessable": cap.geometry.squareness_assessable,
            "max_angle_error_deg": round(cap.geometry.max_angle_error_deg, 2),
            "notes": list(cap.geometry.notes),
        }
    if cap.capability.has_ratio and cap.worst_ratio is not None:
        from .evidence import SequentialBoundaryTest, information_value

        w = cap.worst_ratio
        lo, hi = w.interval()
        tst = SequentialBoundaryTest(threshold=55.0)
        tst.update(w)
        out.update({
            "measured": True,
            "ratio": round(w.value, 1),
            "ratio_lo": round(lo, 1),
            "ratio_hi": round(hi, 1),
            "axis": cap.worst_axis_name,
            "partial_axis": cap.capability is Capability.SINGLE_AXIS,
            "verdict": tst.verdict.value,
            "info_value": round(information_value(w, 55.0), 3),
            "more_views_useful": bool(information_value(w, 55.0) > 0.02),
        })

    if not t.route.is_measurable:
        return out

    if t.px_per_mm and t.px_per_mm < GOOD_PX_PER_MM:
        out["advice"] = (
            f"card is {t.px_per_mm:.1f} px/mm -- move closer. Below "
            f"{GOOD_PX_PER_MM:.0f} px/mm border location is unreliable."
        )

    try:
        res = measure_centering(
            image,
            slab=resolve_holder(holder),
            capture=CaptureSpec.from_fov(LENS_FOV.get(lens, 68.0), image.shape),
        )
    except DetectionError as exc:
        if out.get("measured"):
            # Capability already produced a single-axis result; the full
            # measurement refusing is expected, not an error.
            out["reason"] = cap.reason
            out["advice"] = (
                "One axis only. The other axis has no detectable border pair, "
                "so this is a ceiling on one axis and the other could be worse."
            )
            return out
        full = str(exc)
        msg = full.split("\n")[0]
        out["route"] = "no_border"
        out["reason"] = msg
        # The border-detection failure message lists per-side reasons on
        # CONTINUATION lines, so matching only the first line missed every
        # full-bleed card and told the user to fix the lighting instead. Search
        # the whole message.
        if "full-bleed" in full or "no measurable border" in full or "confidence too low" in full:
            out["reason"] = cap.reason
            out["advice"] = (
                "Full-art, Hyper Rare and Energy cards have no printed border, "
                "so there is no border centering to measure. Verified against a "
                "certified PSA GEM MT 10 (cert 143341329): the card was detected "
                "correctly and refused correctly. Cut geometry and trim checking "
                "still apply, and on a full-bleed card those are the only "
                "objective measurements there are."
            )
        else:
            out["advice"] = "re-shoot with even light and the whole card in frame"
        return out

    w = res.worst_ratio
    lo, hi = w.interval()
    bands = {g: grade_band(w, g, "front") for g in available_graders()}
    overlay = ""
    if res.rectified is not None:
        ok, buf = cv2.imencode(".jpg", res.rectified, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok:
            overlay = base64.b64encode(buf).decode()

    out.update(
        {
            "measured": True,
            "ratio": round(w.value, 1),
            "ratio_lo": round(lo, 1),
            "ratio_hi": round(hi, 1),
            "axis": res.worst_axis.axis,
            "wider": res.worst_axis.skew_toward,
            "borders": {
                "left": round(res.horizontal.low_mm.value, 2),
                "right": round(res.horizontal.high_mm.value, 2),
                "top": round(res.vertical.low_mm.value, 2),
                "bottom": round(res.vertical.high_mm.value, 2),
            },
            "bands": {
                g: (b.best if b.is_single else f"{b.worst}\u2013{b.best}")
                for g, b in bands.items()
            },
            "predicted_grades": {
                g: {
                    "grade": p.grade_label,
                    "condition": p.condition_name,
                    "score": p.grade_score,
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
                        w, quality=res.quality, grader=name, model=maybe_load_grade_model()
                    )
                    for name in bands.keys()
                }.items()
            },
            "warnings": list(res.quality.warnings),
            "overlay": overlay,
        }
    )
    return out


PAGE = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="theme-color" content="#0E1116">
<title>Perceptopoly</title>
<style>
:root{--ink:#0E1116;--card:#181E27;--rise:#212A36;--rule:#2E3A48;
 --paper:#E8E6E1;--dim:#8494A6;--key:#6FD3C7;--warn:#D9A441;--stop:#C25A45;--ok:#5FA97C}
*{box-sizing:border-box;-webkit-tap-highlight-color:transparent}
body{margin:0;background:var(--ink);color:var(--paper);
 font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
 padding-bottom:calc(104px + env(safe-area-inset-bottom))}
.eyebrow{font-size:10px;letter-spacing:.22em;text-transform:uppercase;color:var(--dim);font-weight:600}
header{padding:14px 16px 10px;border-bottom:1px solid var(--rule);display:flex;
 align-items:baseline;gap:10px;position:sticky;top:0;background:var(--ink);z-index:5}
header h1{font-size:15px;margin:0;font-weight:650;letter-spacing:.01em}
header .v{margin-left:auto;font-size:10px;color:var(--dim);font-family:ui-monospace,monospace}
.ctl{display:grid;grid-template-columns:1fr 1fr;gap:8px;padding:12px 16px}
label.f{display:block}label.f .eyebrow{display:block;margin-bottom:5px}
select{width:100%;appearance:none;background:var(--card);color:var(--paper);
 border:1px solid var(--rule);border-radius:9px;padding:11px 12px;font-size:15px;font-family:inherit}

/* Verdict banner: the route decides the colour, because route is the answer */
.verdict{margin:8px 16px 0;border-radius:12px;padding:14px;border:1px solid var(--rule);
 background:var(--card);border-left-width:3px}
.verdict.ok{border-left-color:var(--ok)} .verdict.warn{border-left-color:var(--warn)}
.verdict.stop{border-left-color:var(--stop)} .verdict.info{border-left-color:var(--key)}
.verdict b{display:block;font-size:11px;letter-spacing:.16em;text-transform:uppercase;
 margin-bottom:6px;color:var(--dim)}
.verdict p{margin:0;font-size:14px;line-height:1.5}
.verdict .fix{margin-top:9px;font-size:13px;color:var(--key);line-height:1.45}

.strip{margin:10px 16px 0;height:52px;position:relative;border-radius:10px;
 background:var(--rise);border:1px solid var(--rule);overflow:hidden}
.strip .seg{position:absolute;top:0;bottom:0}
.strip .l{left:0;background:linear-gradient(90deg,#243244,#2E4155)}
.strip .r{right:0;background:linear-gradient(270deg,#243244,#2E4155)}
.strip .sp{position:absolute;top:0;bottom:0;width:2px;background:var(--key)}
.strip .ci{position:absolute;top:0;bottom:0;background:var(--key);opacity:.22}
.strip .mid{position:absolute;top:0;bottom:0;left:50%;width:1px;background:var(--paper);opacity:.35}
.strip .tag{position:absolute;bottom:5px;font-size:9.5px;letter-spacing:.1em;color:#AFC0CF;
 text-transform:uppercase;font-family:ui-monospace,monospace}
.ratio{display:flex;align-items:baseline;gap:10px;padding:14px 16px 2px}
.ratio b{font-size:42px;line-height:.9;font-weight:700;letter-spacing:-.03em;
 font-variant-numeric:tabular-nums;font-family:ui-monospace,monospace}
.ratio span{font-size:12px;color:var(--dim);font-family:ui-monospace,monospace}
.sub{padding:0 16px 10px;font-size:12px;color:var(--dim);line-height:1.5}
.chips{display:flex;gap:6px;padding:0 16px;flex-wrap:wrap}
.chip{background:var(--card);border:1px solid var(--rule);border-radius:7px;padding:7px 10px;
 font-size:12px;font-family:ui-monospace,monospace}
.chip i{font-style:normal;color:var(--dim);margin-right:6px;font-size:10px;letter-spacing:.1em}
table{width:calc(100% - 32px);margin:12px 16px 0;border-collapse:collapse;
 font-family:ui-monospace,monospace;font-size:12px}
td{padding:6px 0;border-top:1px solid var(--rule);color:var(--dim)}
td:last-child{text-align:right;color:var(--paper);font-variant-numeric:tabular-nums}
ul.notes{margin:12px 16px 0;padding:0;list-style:none}
ul.notes li{border-left:2px solid var(--warn);padding:7px 0 7px 10px;margin-bottom:7px;
 font-size:12.5px;line-height:1.45;color:#C9D4DF}
img.ov{width:calc(100% - 32px);max-height:42vh;object-fit:contain;object-position:top;
 margin:12px 16px 0;border-radius:10px;border:1px solid var(--rule);display:block}
.dock{position:fixed;left:0;right:0;bottom:0;padding:12px 16px calc(12px + env(safe-area-inset-bottom));
 background:linear-gradient(to top,var(--ink) 64%,rgba(14,17,22,0));z-index:6}
.shoot{width:100%;border:0;border-radius:12px;padding:17px;font-size:16px;font-weight:650;
 background:var(--key);color:#06201C;font-family:inherit}
.shoot[disabled]{opacity:.5}
input[type=file]{position:absolute;width:1px;height:1px;opacity:0;pointer-events:none}
.empty{padding:22px 16px;color:var(--dim);font-size:13.5px;line-height:1.6}
.empty ul{padding-left:16px;margin:9px 0 0}.empty li{margin-bottom:6px}
.spin{padding:24px 16px;color:var(--key);font-size:13.5px}
</style></head><body>
<header><h1>Perceptopoly</h1><span class="v" id="v"></span></header>
<div class="ctl">
 <label class="f"><span class="eyebrow">Holder</span><select id="holder"></select></label>
 <label class="f"><span class="eyebrow">Lens</span><select id="lens">
  <option value="main">Main 1&times;</option><option value="tele2x">Tele 2&times;</option>
  <option value="tele5x">Tele 5&times;</option><option value="ultrawide">Ultrawide</option>
 </select></label>
</div>
<div id="out"><div class="empty">
 Point at one card, whole card in frame, against a dark matte background.
 <ul>
  <li><b>Fill the frame.</b> The commonest problem in real sessions is standing too far back &mdash; median card filled 3.6% of frame and read ~3&nbsp;px/mm, below the usable floor.</li>
  <li><b>All four edges inside.</b> 23% of real shots had the card cut off, which leaves no closed outline to find.</li>
  <li><b>Full-art, Hyper Rare and Energy cards have no printed border.</b> There is no border centering to measure, and the tool will say so rather than invent one.</li>
 </ul>
</div></div>
<div class="dock">
 <input type="file" id="file" accept="image/*" capture="environment">
 <button class="shoot" id="shoot">Capture</button>
</div>
<script>
const $=s=>document.querySelector(s), out=$('#out'), file=$('#file'), shoot=$('#shoot');
fetch('/holders').then(r=>r.json()).then(d=>{
 $('#holder').innerHTML=d.holders.map(h=>`<option value="${h.id}"${h.id==='raw'?' selected':''}>${h.label}</option>`).join('');
 $('#v').textContent='v'+d.version;});
shoot.onclick=()=>file.click();
file.onchange=()=>{if(file.files[0])send(file.files[0]);};
function esc(s){return String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function send(f){shoot.disabled=true;shoot.textContent='Analysing…';
 out.innerHTML='<div class="spin">Routing the frame</div>';
 const fd=new FormData();fd.append('holder',$('#holder').value);fd.append('lens',$('#lens').value);
 fd.append('image',f,'c.jpg');
 fetch('/analyse',{method:'POST',body:fd}).then(r=>r.json()).then(render)
  .catch(e=>out.innerHTML=`<div class="verdict stop"><b>Error</b><p>${esc(e)}</p></div>`)
  .finally(()=>{shoot.disabled=false;shoot.textContent='Capture';file.value='';});}
const TONE={measure:'ok',measure_low_res:'warn',reshoot_cropped:'stop',no_border:'info',
 calibration:'info',equipment:'info',environment:'info',screenshot:'info',unusable:'stop'};
const TITLE={measure:'Measurable',measure_low_res:'Too small',reshoot_cropped:'Card cut off',
 no_border:'No printed border',calibration:'Calibration frame',equipment:'Equipment',
 environment:'Background',screenshot:'Screenshot',unusable:'Unusable'};
function render(d){
 if(!d.ok){out.innerHTML=`<div class="verdict stop"><b>Not analysed</b><p>${esc(d.error)}</p></div>`;return;}
 let h=`<div class="verdict ${TONE[d.route]||'info'}"><b>${esc(TITLE[d.route]||d.route)}</b>
  <p>${esc(d.reason)}</p>${d.advice?`<div class="fix">${esc(d.advice)}</div>`:''}</div>`;
 if(d.measured){
  const wide=d.ratio,nar=+(100-d.ratio).toFixed(1),horiz=d.axis==='horizontal';
  const a=horiz?d.borders.left:d.borders.top, b=horiz?d.borders.right:d.borders.bottom;
  const split=100*a/(a+b), ci=Math.max(.8,d.ratio_hi-d.ratio_lo);
  h+=`<div class="strip">
   <div class="seg l" style="width:${split.toFixed(2)}%"></div>
   <div class="seg r" style="width:${(100-split).toFixed(2)}%"></div>
   <div class="ci" style="left:${(split-ci/2).toFixed(2)}%;width:${ci.toFixed(2)}%"></div>
   <div class="mid"></div><div class="sp" style="left:${split.toFixed(2)}%"></div>
   <div class="tag" style="left:10px">${esc(horiz?'left':'top')} ${a.toFixed(2)}mm</div>
   <div class="tag" style="right:10px">${b.toFixed(2)}mm ${esc(horiz?'right':'bottom')}</div></div>
  <div class="ratio"><b>${wide.toFixed(1)}/${nar.toFixed(1)}</b>
   <span>&plusmn;${((d.ratio_hi-d.ratio_lo)/2).toFixed(1)}</span></div>
  <div class="sub">${esc(d.axis)} axis is worse &middot; wider on ${esc(d.wider)}
   &middot; 95% CI ${d.ratio_lo.toFixed(1)}&ndash;${d.ratio_hi.toFixed(1)}</div>
  <div class="chips">${Object.entries(d.bands).map(([g,v])=>
   `<div class="chip"><i>${esc(g)}</i>${esc(v)}</div>`).join('')}</div>
  <table>
   <tr><td>left / right</td><td>${d.borders.left.toFixed(2)} / ${d.borders.right.toFixed(2)} mm</td></tr>
   <tr><td>top / bottom</td><td>${d.borders.top.toFixed(2)} / ${d.borders.bottom.toFixed(2)} mm</td></tr>
   <tr><td>scale</td><td>${d.px_per_mm} px/mm</td></tr>
   <tr><td>metric reference</td><td>${d.caliper?'caliper in frame':'none'}</td></tr>
  </table>`;
  if(d.warnings&&d.warnings.length)
   h+=`<ul class="notes">${d.warnings.map(w=>`<li>${esc(w)}</li>`).join('')}</ul>`;
  if(d.overlay)h+=`<img class="ov" alt="rectified card" src="data:image/jpeg;base64,${d.overlay}">`;
  h+=`<div class="sub" style="padding-top:14px">Centering only. This caps a grade; it never
   confirms one. Corners, edges and surface are not measured.</div>`;
 } else if(d.px_per_mm){
  h+=`<table><tr><td>scale</td><td>${d.px_per_mm} px/mm</td></tr>
   <tr><td>metric reference</td><td>${d.caliper?'caliper in frame':'none'}</td></tr></table>`;
 }
 out.innerHTML=h;window.scrollTo({top:0,behavior:'smooth'});
}
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    server_version = "perceptopoly"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif self.path == "/holders":
            from . import __version__
            from .types import SLAB_PRESETS, SLAB_STACKS

            hs = [{"id": "raw", "label": "Raw card"}]
            hs += [
                {"id": k, "label": k.replace("_", " ").title()}
                for k in sorted(SLAB_PRESETS)
                if k not in ("raw", "case_glass")
            ]
            hs += [
                {"id": k, "label": "In case: " + k.replace("case_", "").upper()}
                for k in sorted(SLAB_STACKS)
            ]
            self._send(
                200,
                json.dumps({"holders": hs, "version": __version__}).encode(),
                "application/json",
            )
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/analyse":
            self._send(404, b"not found", "text/plain")
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            fields = _parse_multipart(self.rfile.read(n), self.headers.get("Content-Type", ""))
            img = fields.get("image")
            if not img:
                raise DetectionError("no photo was attached")
            payload = analyse(
                img,
                fields.get("holder", b"raw").decode(),
                fields.get("lens", b"main").decode(),
            )
        except DetectionError as exc:
            payload = {"ok": False, "error": str(exc)}
        except Exception as exc:  # pragma: no cover
            traceback.print_exc()
            payload = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        self._send(200, json.dumps(payload).encode(), "application/json")


def serve(host: str = "127.0.0.1", port: int = 8766) -> None:
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"\n  Perceptopoly running.\n\n    http://127.0.0.1:{port}\n\n  Ctrl+C to stop.\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped.\n")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    serve()
