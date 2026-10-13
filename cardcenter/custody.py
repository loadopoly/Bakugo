"""Physical custody evidence: a photo proving someone holds a card right now.

A scan of a card is data about the card; it says nothing about who holds it. To become the
custodian of record for a card's GARD Shards (which is what lets them move on the
marketplace), a person asks the GARD Exchange for a short challenge code, writes it on paper,
and photographs the card next to it from their own Bakugo device. This module keeps that
photo and what Bakugo could check about it:

* ``card``: what the card reader makes of the photo (name, dex, matched token);
* ``challenge_seen``: whether the code could be read in the photo (handwriting defeats OCR
  often, so a miss here is a hint for the human verifier, not a refusal);
* the device that took it (from its server-issued token), the time, and the image's sha256.

Evidence is written as files, ``<CARDCENTER_CUSTODY_DIR>/<evidence_id>.json`` and ``.jpg``
(default: ``custody/`` beside the database). They are never rotated, so a custody decision
can always be reviewed against the photo it was made on. The exchange reads them read-only;
the decision (a human verifier's attestation, then a ledger row) is the exchange's.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

CHALLENGE_RE = re.compile(r"^[A-HJ-NP-Z2-9]{3}-[A-HJ-NP-Z2-9]{3}$")     # no 0/O/1/I: easy to write and read
ASSET_RE = re.compile(r"^[A-Za-z0-9:._-]{3,200}$")
MAX_IMAGE_BYTES = 12 * 1024 * 1024


class CustodyError(ValueError):
    """The evidence cannot be accepted; the message is safe to show."""


def custody_dir() -> Path:
    d = os.environ.get("CARDCENTER_CUSTODY_DIR")
    if d:
        return Path(d)
    db = os.environ.get("CARDCENTER_DB", "cardcenter.db")
    return Path(db).resolve().parent / "custody"


def _normal(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", s.upper().replace("O", "0").replace("I", "1"))


def challenge_in(text: str, challenge: str) -> bool:
    """Whether the challenge appears in OCR text, forgiving dashes, spaces and O/0, I/1 confusions."""
    return bool(challenge) and _normal(challenge) in _normal(text)


def read_text(image_bytes: bytes) -> str:
    """Sparse-text OCR over the whole photo (Tesseract psm 11), letters and digits only. Empty when unavailable."""
    if shutil.which("tesseract") is None:
        return ""
    try:
        import cv2
        import numpy as np

        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return ""
        long_side = max(img.shape[:2])
        if long_side > 2400:
            s = 2400 / long_side
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        from .ocr import background_run_kwargs

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "photo.png"
            cv2.imwrite(str(path), img)
            out = subprocess.run(["tesseract", str(path), "stdout", "--psm", "11", "-c",
                                  "tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-", "-c", "debug_file=/dev/null"],
                                 capture_output=True, text=True, timeout=40, check=False, **background_run_kwargs())
        return out.stdout[:2000]
    except Exception:
        return ""


def record_evidence(device_id: str, challenge: str, asset_id: str, image_bytes: bytes,
                    identify: Optional[Callable[[bytes], dict]] = None,
                    ocr: Callable[[bytes], str] = read_text, version: str = "") -> dict:
    """Keeps one custody photo and what could be checked about it. Returns the evidence document."""
    challenge = str(challenge or "").strip().upper()
    asset_id = str(asset_id or "").strip()
    if not CHALLENGE_RE.match(challenge):
        raise CustodyError("the challenge code looks like ABC-123 (from the GARD Exchange: holon market custody claim)")
    if not ASSET_RE.match(asset_id):
        raise CustodyError("asset_id must be the card's GARD asset id, e.g. bakugo:identifications:ab12cd34ef56ab78")
    if not image_bytes:
        raise CustodyError("attach a photo of the card with the code written next to it")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise CustodyError("the photo is over 12 MB")
    if not (image_bytes[:3] == b"\xff\xd8\xff" or image_bytes[:8] == b"\x89PNG\r\n\x1a\n" or image_bytes[8:12] == b"WEBP"):
        raise CustodyError("the photo must be a JPEG, PNG or WebP image")
    card: dict = {}
    if identify is not None:
        try:
            got = identify(image_bytes) or {}
            card = {k: got.get(k) for k in ("identified", "name", "dex", "matched_token", "corroborated", "identification_id")}
        except Exception as exc:          # a card the reader cannot place is still evidence for the human verifier
            card = {"identified": False, "error": str(exc)[:200]}
    text = ocr(image_bytes) if ocr else ""
    now = datetime.now(timezone.utc)
    evidence_id = now.strftime("%Y%m%dT%H%M%S_") + secrets.token_hex(4)
    ext = ".png" if image_bytes[:4] == b"\x89PNG" else (".webp" if image_bytes[8:12] == b"WEBP" else ".jpg")
    doc = {
        "evidence_id": evidence_id, "kind": "custody", "device_id": device_id, "challenge": challenge, "asset_id": asset_id,
        "sha256": hashlib.sha256(image_bytes).hexdigest(), "image": evidence_id + ext, "bytes": len(image_bytes),
        "utc": now.isoformat(), "unix": time.time(), "card": card, "challenge_seen": challenge_in(text, challenge),
        "ocr_text": text[:400], "version": version,
    }
    root = custody_dir()
    root.mkdir(parents=True, exist_ok=True)
    (root / doc["image"]).write_bytes(image_bytes)
    tmp = root / (evidence_id + ".json.tmp")
    tmp.write_text(json.dumps(doc, default=str, indent=1), encoding="utf-8")
    os.replace(tmp, root / (evidence_id + ".json"))             # readers never see a half-written document
    return doc


def evidence_for(device_id: str, challenge: str = "", limit: int = 20) -> list[dict]:
    """This device's evidence (for one challenge, or all of it), newest first."""
    root = custody_dir()
    if not root.exists():
        return []
    want = str(challenge or "").strip().upper()
    out = []
    for p in sorted(root.glob("*.json"), reverse=True):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if d.get("device_id") == device_id and (not want or d.get("challenge") == want):
            out.append({k: v for k, v in d.items() if k != "ocr_text"})
            if len(out) >= limit:
                break
    return out


def asset_id_for(table: str, row_id) -> str:
    """The GARD asset id the shard forge gives a record with an id (services/shardforge in the fleet repository)."""
    canon = json.dumps({"table": table, "id": row_id}, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
    return f"bakugo:{table}:{hashlib.sha256(canon.encode('ascii')).hexdigest()[:16]}"


def cards_for(db_path: str, device_id: str, limit: int = 100) -> list[dict]:
    """The cards this device identified, each with the GARD asset id a custody claim names. Scanning a card does not
    make anyone its custodian; this list only saves typing the asset id."""
    import sqlite3

    try:
        conn = sqlite3.connect(db_path, timeout=10)
        try:
            rows = conn.execute("SELECT id, name, dex, created_at FROM identifications WHERE device_id = ? "
                                "ORDER BY created_at DESC LIMIT ?", (device_id, int(limit))).fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        return []
    return [{"identification": rid, "name": name, "dex": dex, "created_at": created,
             "asset_id": asset_id_for("identifications", rid)} for rid, name, dex, created in rows]


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bakugo custody</title>
<style>
:root{--bg:#f3f1ec;--panel:#fff;--ink:#1f2328;--muted:#5d646d;--line:#d9d5cc;--ok:#1f7a62;--warn:#a8452a}
@media (prefers-color-scheme:dark){:root{--bg:#16181b;--panel:#1f2226;--ink:#e6e3dd;--muted:#a09a90;--line:#33373c;--ok:#4fb895;--warn:#e07a5f}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:560px;margin:0 auto;padding:20px 16px 40px}h1{font:600 22px/1.25 Georgia,serif;margin:0 0 6px}h2{font:600 17px/1.3 Georgia,serif;margin:26px 0 8px}
p{color:var(--muted);margin:6px 0 12px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
label{display:grid;gap:4px;font-size:14px;color:var(--muted);margin:10px 0}input{font:inherit;color:var(--ink);background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:9px 10px;width:100%}
button{font:inherit;border:1px solid var(--ink);background:var(--ink);color:var(--panel);border-radius:6px;padding:9px 16px;margin-top:8px}
code{font:14px ui-monospace,Consolas,monospace;overflow-wrap:anywhere}.msg{margin:12px 0;padding:10px 12px;border-radius:6px;border:1px solid var(--line)}
.ok{border-color:var(--ok)}.err{border-color:var(--warn);color:var(--warn)}ol{padding-left:20px;color:var(--muted)}
</style></head><body><main>
<h1>Prove you hold a card</h1>
<p>A scan shows what a card is. It does not show who holds it. A card's GARD Shards move only after its custodian is confirmed in the ledger.</p>
<ol><li>In Holon: <code>holon market custody claim &lt;asset id&gt;</code>. It gives you a code like <code>K7Q-4MP</code>.</li>
<li>Write the code on paper and lay it next to the card.</li><li>Take the photo here, from this phone.</li>
<li>Back in Holon: <code>holon market custody submit &lt;claim&gt;</code>. A verifier reviews the photo.</li></ol>
<div class="panel"><form id="f">
<label>Challenge code <input name="challenge" required placeholder="K7Q-4MP" autocapitalize="characters" autocomplete="off"></label>
<label>Card asset id <input name="asset_id" required placeholder="bakugo:identifications:..." autocomplete="off"></label>
<label>Photo: the card and the code on paper, both in focus <input name="image" type="file" accept="image/*" capture="environment" required></label>
<button>Send the photo</button></form><div id="out"></div></div>
<h2>Link this device to Holon</h2>
<p>Holon needs this device's key once: <code>holon market link bakugo --device-token &lt;key&gt;</code>. Anyone with the key can act as this device, so keep it private.</p>
<div class="panel"><button id="show" type="button">Show this device's key</button><p id="key"></p></div>
</main><script>
const out=document.getElementById("out");
const esc=(s)=>String(s??"").replace(/[&<>"']/g,(c)=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
document.getElementById("f").addEventListener("submit",async(e)=>{e.preventDefault();out.innerHTML='<div class="msg">Sending and reading the photo…</div>';
 try{const r=await fetch("/custody/evidence",{method:"POST",body:new FormData(e.target),credentials:"same-origin"});const j=await r.json();
  if(!j.ok){out.innerHTML=`<div class="msg err">${esc(j.error||"refused")}</div>`;return}
  const c=j.evidence.card||{};out.innerHTML=`<div class="msg ok">Kept as evidence ${esc(j.evidence.evidence_id)}.<br>Card read: ${esc(c.name||"not read")}${c.dex?" #"+esc(c.dex):""}.<br>Code ${j.evidence.challenge_seen?"read in the photo":"not read (a verifier will look)"}.<br>Now submit the claim in Holon.</div>`}
 catch(err){out.innerHTML=`<div class="msg err">${esc(err.message)}</div>`}});
document.getElementById("show").addEventListener("click",async()=>{const j=await (await fetch("/custody/device",{credentials:"same-origin"})).json();document.getElementById("key").innerHTML=`<code>${esc(j.device_token)}</code>`});
</script></body></html>"""
