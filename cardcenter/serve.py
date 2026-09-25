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
import math
import os
import socket
import sys
import hashlib
import secrets
import threading
import time
import traceback
from collections import OrderedDict
from http.cookies import CookieError, SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

import cv2
import numpy as np

from . import __version__
from .ar import ARSession, ARStatus
from .capture import MIN_PX_PER_MM, assess_frame
from .centering import measure_centering
from .framing import frame_card_for_measure
from .grading import available_graders, grade_band, predict_overall_grade
from .learning import maybe_load_grade_model
from .render import annotate
from .types import SLAB_PRESETS, SLAB_STACKS, CaptureSpec, DetectionError, resolve_holder


def _finite(obj):
    """The same structure with every non-finite float replaced by None.

    json.dumps writes float('inf') and NaN as the bare tokens Infinity and
    NaN. That is not JSON, and JSON.parse in the browser throws on it: one
    unmeasurable side in a soft frame made every /ar/push answer "HTTP 200 not
    JSON" and the live view showed PUSH FAILED while the server was working.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    if isinstance(obj, np.generic):
        return _finite(obj.item())
    return obj


def _dumps(obj, **kw) -> str:
    """Strict JSON for anything sent to a browser."""
    return json.dumps(_finite(obj), allow_nan=False, **kw)

# Typical horizontal fields of view for phone cameras.
LENS_FOV = {
    "main": 68.0,
    "ultrawide": 105.0,
    "tele2x": 38.0,
    "tele5x": 16.0,
}



# ---------------------------------------------------------------------------
# Request limits and security settings. Every value can be overridden from the
# environment; the defaults are what the public deployment runs with.
# ---------------------------------------------------------------------------


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


# Largest request body accepted, checked against Content-Length before reading.
MAX_BODY_BYTES = _env_int("CARDCENTER_MAX_BODY_BYTES", 12 * 1024 * 1024)
# Largest decoded image, checked from the file header before decoding.
MAX_IMAGE_PIXELS = _env_int("CARDCENTER_MAX_IMAGE_PIXELS", 40_000_000)
# Per-connection socket timeout, so a slow client cannot hold a thread forever.
SOCKET_TIMEOUT_S = _env_int("CARDCENTER_SOCKET_TIMEOUT", 30)
# Concurrent connections served; further connections wait, then are dropped.
MAX_CONNECTIONS = _env_int("CARDCENTER_MAX_CONNECTIONS", 32)
# Live AR sessions kept in memory (least recently used are evicted).
MAX_AR_SESSIONS = _env_int("CARDCENTER_MAX_AR_SESSIONS", 256)

# Browser origins allowed to call the API. Override with a comma-separated
# CARDCENTER_CORS_ORIGINS. The two localhost entries are the Capacitor shell in
# mobile/ (androidScheme https -> https://localhost; iOS -> capacitor://localhost).
DEFAULT_CORS_ORIGINS = (
    "https://bakugo.loadopoly.com",
    "https://loadopoly.com",
    "https://localhost",
    "capacitor://localhost",
)


def cors_origins() -> frozenset:
    raw = os.environ.get("CARDCENTER_CORS_ORIGINS")
    if raw is None:
        return frozenset(DEFAULT_CORS_ORIGINS)
    return frozenset(o.strip().rstrip("/") for o in raw.split(",") if o.strip())


# ---------------------------------------------------------------------------
# Tenant identity. The server issues each device a random token (HttpOnly
# cookie for the web page, Bearer token for cross-origin clients). The tenant
# key stored with scans is a hash of that token, so a device_id seen in a
# response or in the database cannot be replayed to read someone's data.
# ---------------------------------------------------------------------------

DEVICE_COOKIE = "bakugo_device"
_TOKEN_PREFIX = "bk1_"
_TOKEN_CHARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_")


def new_device_token() -> str:
    return _TOKEN_PREFIX + secrets.token_urlsafe(32)


def _valid_token(tok: Optional[str]) -> bool:
    if not tok or not tok.startswith(_TOKEN_PREFIX):
        return False
    rest = tok[len(_TOKEN_PREFIX):]
    return 40 <= len(rest) <= 64 and all(c in _TOKEN_CHARS for c in rest)


def device_id_for_token(token: str) -> str:
    return "dev_" + hashlib.sha256(token.encode("ascii")).hexdigest()[:32]


# Multi-tenant in-memory AR sessions keyed by device_id, bounded LRU.
_AR_SESSIONS: "OrderedDict[str, ARSession]" = OrderedDict()
_AR_LOCK = threading.Lock()


# how long a push waits for the previous one on the same session
AR_BUSY_WAIT_S = 1.5


class _Busy(Exception):
    """The session is still working on the previous frame."""


def _get_or_create_ar_session(
    device_id: str,
    holder: str = "raw",
    lens: str = "main",
    boundary: float = 55.0,
) -> ARSession:
    fov = LENS_FOV.get(lens, LENS_FOV["main"])
    with _AR_LOCK:
        session = _AR_SESSIONS.get(device_id)
        if session is None or session.holder != holder or abs(session.fov_deg - fov) > 1e-3:
            session = ARSession(holder=holder, fov_deg=fov, boundary=boundary)
            # one push at a time per session (see /ar/push)
            session._serve_lock = threading.Lock()
            _AR_SESSIONS[device_id] = session
        _AR_SESSIONS.move_to_end(device_id)
        while len(_AR_SESSIONS) > max(1, MAX_AR_SESSIONS):
            _AR_SESSIONS.popitem(last=False)
    return session


def _image_dims(buf: bytes) -> Optional[tuple]:
    """Width and height from a JPEG, PNG or WebP header, without decoding pixels."""
    if buf[:8] == b"\x89PNG\r\n\x1a\n" and len(buf) >= 24:
        return int.from_bytes(buf[16:20], "big"), int.from_bytes(buf[20:24], "big")
    if buf[:4] == b"RIFF" and buf[8:12] == b"WEBP" and len(buf) >= 30:
        chunk = buf[12:16]
        if chunk == b"VP8X":
            return 1 + int.from_bytes(buf[24:27], "little"), 1 + int.from_bytes(buf[27:30], "little")
        if chunk == b"VP8 ":
            return (int.from_bytes(buf[26:28], "little") & 0x3FFF,
                    int.from_bytes(buf[28:30], "little") & 0x3FFF)
        if chunk == b"VP8L":
            bits = int.from_bytes(buf[21:25], "little")
            return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
        return None
    if buf[:2] == b"\xff\xd8":
        sof = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
        i, n = 2, len(buf)
        while i + 9 <= n:
            if buf[i] != 0xFF:
                i += 1
                continue
            marker = buf[i + 1]
            if marker == 0xFF:
                i += 1
                continue
            if marker in (0x01, 0xD8) or 0xD0 <= marker <= 0xD7:
                i += 2
                continue
            seg = int.from_bytes(buf[i + 2:i + 4], "big")
            if marker in sof:
                return int.from_bytes(buf[i + 7:i + 9], "big"), int.from_bytes(buf[i + 5:i + 7], "big")
            if seg < 2:
                return None
            i += 2 + seg
    return None


def decode_image(buf: bytes, what: str = "image") -> np.ndarray:
    """Decode an uploaded image, refusing unknown formats and oversized images before decoding."""
    dims = _image_dims(buf)
    if dims is None:
        raise DetectionError(f"that {what} is not a JPEG, PNG or WebP file")
    w, h = dims
    limit_mp = MAX_IMAGE_PIXELS // 1_000_000
    if w <= 0 or h <= 0 or w * h > MAX_IMAGE_PIXELS:
        raise DetectionError(f"that {what} is too large ({w}x{h}); the limit is {limit_mp} MP")
    image = cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise DetectionError(f"that {what} could not be decoded")
    if image.shape[0] * image.shape[1] > MAX_IMAGE_PIXELS:
        raise DetectionError(f"that {what} is too large; the limit is {limit_mp} MP")
    return image


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


def _field_float(fields: dict, name: str) -> Optional[float]:
    """A finite number from a form field, or None."""
    raw = fields.get(name)
    if not raw:
        return None
    try:
        v = float(raw.decode("ascii", "replace").strip())
    except (ValueError, UnicodeDecodeError):
        return None
    return v if math.isfinite(v) else None


def _parent_crop(fields: dict, image_bytes: bytes):
    """(x, y, full_w, full_h) when the phone uploaded a crop of its photo.

    The app crops a full-resolution still to the card it was tracking before
    it uploads (a 12 MP photo over shop Wi-Fi, twice, was the slow part). The
    lens still belongs to the whole photo, so the crop's place in it comes
    along. Anything missing or inconsistent means "not a crop": the image is
    then measured as the whole photo, exactly as before.
    """
    keys = ("crop_x", "crop_y", "full_w", "full_h")
    if not all(fields.get(k) for k in keys):
        return None
    try:
        x, y, fw, fh = (int(float(fields[k].decode("ascii", "replace"))) for k in keys)
    except (ValueError, UnicodeDecodeError):
        return None
    dims = _image_dims(image_bytes)
    if dims is None:
        return None
    w, h = dims
    if (x < 0 or y < 0 or fw <= 0 or fh <= 0 or x + w > fw + 2 or y + h > fh + 2
            or fw * fh > 4 * MAX_IMAGE_PIXELS):
        return None
    return (x, y, fw, fh)


# Focus of a still, as the Laplacian variance of the card region resampled to
# 8 px/mm (so it does not depend on how big the card is in the photo). On the
# field stills: 155-267 for the ones that measured, 6-39 for the soft ones
# that were refused (one "edge shadow", three "border confidence too low").
SOFT_STILL = 60.0
SHARP_STILL = 150


def still_sharpness(image: np.ndarray, quad) -> Optional[float]:
    if quad is None:
        return None
    from .capture import assess_frame

    q = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    ppm = float(np.linalg.norm(q[1] - q[0])) / 63.0
    if ppm <= 0:
        return None
    s = min(1.0, 8.0 / ppm)
    small = cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1.0 else image
    return float(assess_frame(small, q * s).sharpness)


def _measure_payload(image_bytes: bytes, holder: str, lens: str, parent=None) -> dict:
    source = decode_image(image_bytes, "file")

    # Spend the 2400 px budget on the card, not on the room around it: capping
    # the whole frame first costs a card photographed across a counter half its
    # resolution for nothing. framing crops to the card and carries the
    # intrinsics and the located outline with it.
    fov = LENS_FOV.get(lens, LENS_FOV["main"])
    framed = frame_card_for_measure(source, fov_deg=fov, max_side=2400, parent=parent)
    if framed.px_per_mm is not None and framed.px_per_mm < MIN_PX_PER_MM:
        # Measured on synthetic captures: at 4.9 px/mm the ratio came back
        # 0.08 pp from truth, at 4.0 px/mm it came back 25 pp out with an
        # error bar that did not cover the miss. Refuse rather than report it.
        raise DetectionError(
            f"the card is only {framed.px_per_mm:.1f} px/mm in this photo "
            f"(needs {MIN_PX_PER_MM:.1f}). Move closer, zoom in, or use the "
            "telephoto lens -- from this distance the border measurement is "
            "not trustworthy."
        )
    image = framed.image
    capture = framed.capture
    try:
        result = measure_centering(image, slab=resolve_holder(holder), capture=capture,
                                   card_quad=framed.quad, quad_residual_px=framed.residual_px)
    except DetectionError as exc:
        sharp = still_sharpness(image, framed.quad)
        if sharp is not None and sharp < SOFT_STILL:
            # A soft photo fails in whatever check comes first -- border
            # confidence, an "edge shadow" -- and that reason sends the user
            # after the light. Say what it is.
            raise DetectionError(
                f"the photo is out of focus (sharpness {sharp:.0f}; a sharp one "
                f"is {SHARP_STILL}+), so the card's edges can't be placed. Hold the "
                "phone a little further back, keep it still, and tap the card to "
                f"focus before Freeze. (It failed on: {exc})") from exc
        raise

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


def _insitu_store():
    """The in-situ tables live in the same SQLite file as the scans."""
    db = os.environ.get("CARDCENTER_DB", "cardcenter.db")
    if not db:
        return None
    from .insitu import InSituStore

    return InSituStore(db)


def _json_body(body: bytes) -> dict:
    try:
        data = json.loads(body.decode("utf-8")) if body else {}
    except (UnicodeDecodeError, ValueError):
        raise DetectionError("request body must be JSON")
    if not isinstance(data, dict):
        raise DetectionError("request body must be a JSON object")
    return data


# The card is rectified for reading with this much of its height above and
# below it. The outline on a sleeved holo card is often one band off -- on a
# Meganium (2.20.0) its top sat on the artwork's top edge, under the name bar,
# and the name was cut off the card that OCR saw. A name is ~12 mm tall at
# most from the card's top; 15% of 88 mm keeps it in view either way up.
READ_MARGIN_FRAC = 0.15


def _rectify_for_reading(image: np.ndarray, quad: np.ndarray, px_per_mm: float):
    """(rectified card with READ_MARGIN_FRAC of margin top and bottom, the
    name band as fractions of its height)."""
    from .recognise import NAME_BAND
    from .types import STANDARD_CARD_H_MM, STANDARD_CARD_W_MM

    s = float(px_per_mm)
    w, h = STANDARD_CARD_W_MM * s, STANDARD_CARD_H_MM * s
    mx, my = 0.04 * w, READ_MARGIN_FRAC * h
    dst = np.float32([[mx, my], [mx + w, my], [mx + w, my + h], [mx, my + h]])
    M = cv2.getPerspectiveTransform(np.asarray(quad, dtype=np.float32).reshape(4, 2), dst)
    size = (int(round(w + 2 * mx)), int(round(h + 2 * my)))
    rect = cv2.warpPerspective(image, M, size, flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
    return rect, (0.0, (my + NAME_BAND * h) / (h + 2 * my))


def _identify_payload(image_bytes: bytes, device_id: str, owner: bool) -> dict:
    """OCR identification of one still, recorded for feedback and the second
    vote. The outline comes from locate_card (find_card_quad plus the
    information snap), and its information floor feeds the decision."""
    from .confidence import IdentificationVote, gate_identification
    from .edge_information import locate_card
    from .geometry import rectify
    from .recognise import recognise_card
    from .types import STANDARD_CARD_W_MM

    from .framing import card_region, locate_coarse_to_fine, min_area_frac_for

    image = decode_image(image_bytes, "image")
    # Across a counter the card is a small part of the photo, and a 1400 px
    # working copy of the whole frame leaves it too small to find or read.
    # Locate it coarse to fine first (the same search /measure uses), then do
    # the identification inside a generous crop of the original pixels.
    work, origin = image, (0, 0)
    found = locate_coarse_to_fine(image)
    if found is not None:
        x0, y0, x1, y1 = card_region(image.shape, found[0])
        if (x1 - x0) * (y1 - y0) < 0.5 * image.shape[0] * image.shape[1]:
            work, origin = image[y0:y1, x0:x1], (x0, y0)
    detect_long = 1400
    s = min(1.0, detect_long / max(work.shape[:2]))
    small = (
        cv2.resize(work, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        if s < 1.0
        else work
    )
    ih, iw = small.shape[:2]
    quad, _, _, info, snapped = locate_card(small, prefer_point=(iw / 2.0, ih / 2.0),
                                            min_area_frac=min_area_frac_for(work.shape))
    quad_full = quad / s + np.array(origin, dtype=np.float64)
    ppm = float(np.linalg.norm(quad_full[1] - quad_full[0])) / STANDARD_CARD_W_MM
    rect, band = _rectify_for_reading(image, quad_full, min(ppm, 24.0))
    rec = recognise_card(rect, name_band=band)
    payload = {
        "ok": True,
        "identified": rec.resolved,
        "name": rec.name,
        "dex": rec.dex,
        "matched_token": rec.matched_token,
        "edits": rec.edits,
        "alternatives": list(rec.alternatives),
        "corroborated": rec.corroborated,
        "orientation": rec.orientation,
        "tokens_considered": rec.tokens_considered,
        "engine": rec.engine,
        "warnings": list(rec.warnings),
        "quad": [[round(float(x), 1) for x in pt] for pt in quad_full],
        "image_size": [int(image.shape[1]), int(image.shape[0])],
        "px_per_mm": round(ppm, 3),
        "outline_snapped": bool(snapped),
        "resolved_by": "ocr" if rec.resolved else None,
    }
    from .embed import status as embed_status

    est = embed_status()
    if est.get("available"):
        eh, ew = est["input_size"]
        crop = cv2.resize(rect, (ew, eh), interpolation=cv2.INTER_AREA)
        ok_png, png = cv2.imencode(".png", crop)
        if ok_png:
            payload["embed_crop"] = base64.b64encode(png.tobytes()).decode("ascii")
    if info is not None:
        payload["information"] = info.scaled(1.0 / s).to_dict()
        payload["shot_ratio"] = round(float(info.shot_ratio), 3)
        payload["resolvable"] = bool(info.resolvable)
    store = _insitu_store()
    if store is None:
        payload["identification_id"] = None
        return payload
    with store:
        if not rec.resolved and rec.alternatives:
            tie = store.break_tie(device_id, rec.alternatives)
            if tie is not None:
                payload["name"], p = tie
                payload["resolved_by"] = "device_prior"
                payload["device_prior_probability"] = round(p, 3)
                payload["warnings"].append(
                    f"OCR could not choose between {', '.join(rec.alternatives)}; "
                    f"this device's confirmed history favours {tie[0]}"
                )
        gated = gate_identification(
            payload["name"], bool(rec.corroborated),
            IdentificationVote(False, reason="waiting for the second vote"),
            float(payload.get("shot_ratio", float("inf"))),
            bool(payload.get("resolvable", False)),
        )
        payload["decision"] = gated.decision.value
        payload["decision_reason"] = gated.reason
        ident = store.record_identification(device_id, payload)
    payload["identification_id"] = ident
    if owner:
        from .insitu import PENDING

        PENDING.put(ident, bytes(image_bytes))
    return payload


def _vote_payload(data: dict, device_id: str) -> dict:
    from .confidence import IdentificationVote, gate_identification

    ident_id = str(data.get("identification_id") or "")
    raw = data.get("vote") or {}
    if not isinstance(raw, dict):
        raise DetectionError("vote must be an object")
    store = _insitu_store()
    if store is None:
        raise DetectionError("identification records are disabled on this server")
    with store:
        ident = store.get_identification(ident_id, device_id)
        if ident is None:
            raise DetectionError("unknown identification for this device")

        def num(key, lo, hi, default=0.0):
            try:
                v = float(raw.get(key, default))
            except (TypeError, ValueError):
                return default
            return min(max(v, lo), hi)

        name = raw.get("name")
        vote = IdentificationVote(
            available=bool(raw.get("available")),
            name=str(name)[:120] if name else None,
            similarity=num("similarity", -1.0, 1.0),
            margin=num("margin", -2.0, 2.0),
            index_size=int(num("index_size", 0, 1e9)),
            model_id=str(raw.get("model_id") or "")[:80],
            reason=str(raw.get("reason") or "")[:200],
        )
        shot = ident.get("shot_ratio")
        gated = gate_identification(
            ident.get("name"), bool(ident.get("corroborated")), vote,
            float(shot) if shot is not None else float("inf"),
            bool(ident.get("resolvable")),
        )
        vote_record = {"available": vote.available, "name": vote.name,
                       "similarity": vote.similarity, "margin": vote.margin,
                       "index_size": vote.index_size, "model_id": vote.model_id}
        store.record_decision(ident_id, device_id, gated.decision.value, gated.reason, vote_record)
    return {"ok": True, "identification_id": ident_id, "decision": gated.decision.value,
            "decision_reason": gated.reason, "vote": vote_record}


def _feedback_payload(data: dict, device_id: str, owner: bool) -> dict:
    from .insitu import PENDING, FeedbackError, write_inbox
    from .pricing import PriceAttribution, PriceError, parse_price_text

    store = _insitu_store()
    if store is None:
        raise DetectionError("feedback is disabled on this server")
    ident_id = data.get("identification_id")
    price_spec = data.get("price")
    try:
        with store:
            fb = store.add_feedback(
                device_id, str(data.get("action") or ""),
                identification_id=str(ident_id) if ident_id else None,
                scan_id=data.get("scan_id") if isinstance(data.get("scan_id"), int) else None,
                name=data.get("name"), number=data.get("number"),
                franchise=data.get("franchise"), note=data.get("note"), owner=owner,
            )
            price_out = None
            if price_spec:
                if not isinstance(price_spec, dict):
                    raise FeedbackError("price must be an object")
                spec = {k: v for k, v in price_spec.items()
                        if k in ("amount", "currency", "source", "scope", "scope_key", "kind",
                                 "quantity", "venue", "text")}
                text = spec.pop("text", None)
                if text is not None:
                    parsed = parse_price_text(str(text)[:80])
                    if parsed is None:
                        raise FeedbackError(f"no price found in {text!r}")
                    spec.setdefault("raw_text", str(text)[:80])
                    for k, v in parsed.items():
                        spec.setdefault(k, v)
                spec.setdefault("method", "entered")
                att = PriceAttribution.from_dict(spec)
                store.add_price(device_id, att, identification_id=fb["identification"]["id"]
                                if fb["identification"] else None)
                price_out = att.to_dict()
    except (FeedbackError, PriceError, TypeError) as exc:
        raise DetectionError(str(exc))
    out = {"ok": True, "feedback_id": fb["feedback_id"], "learned": fb["learned"],
           "name": fb["name"], "number": fb["number"], "price": price_out, "inbox": False}
    ident = fb["identification"]
    if owner and ident is not None and data.get("action") in ("confirm", "correct"):
        photo = PENDING.pop(ident["id"])
        if photo is not None:
            label = {"name": fb["name"], "number": fb["number"],
                     "franchise": fb["franchise"], "action": data.get("action")}
            try:
                out["inbox"] = write_inbox(ident, label, price_out, photo, device_id) is not None
            except OSError as exc:
                out["inbox_error"] = str(exc)
    return out


MANIFEST_JSON = _dumps(
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
#ar-canvas{position:absolute;top:0;left:0;width:100%;height:100%;object-fit:cover;pointer-events:none}
#ar-debug{position:absolute;right:10px;bottom:56px;width:30%;max-width:180px;border:1px solid #FF5A5A;border-radius:6px;background:#000;z-index:5}
#ar-debug.hidden{display:none}
.ar-hud-overlay{position:absolute;top:10px;left:10px;right:10px;display:flex;justify-content:space-between;align-items:center;pointer-events:none}
.hud-chip{background:rgba(11,15,21,0.85);backdrop-filter:blur(10px);border:1px solid var(--key);padding:4px 10px;border-radius:20px;font-size:11px;font-family:ui-monospace,"SF Mono",monospace;color:#FFF;display:flex;align-items:center;gap:6px}
.hud-chip .radar{width:7px;height:7px;border-radius:50%;background:var(--key);animation:p 1s infinite alternate}
@keyframes p{from{opacity:.3;transform:scale(.8)}to{opacity:1;transform:scale(1.2)}}
.hud-verdict{background:rgba(11,15,21,0.85);backdrop-filter:blur(10px);border:1px solid var(--rule);padding:4px 10px;border-radius:20px;font-size:11px;font-family:ui-monospace,"SF Mono",monospace;color:var(--dim)}
.hud-chip.settled{border-color:var(--pass);color:var(--pass)}
.hud-chip.settled .radar{background:var(--pass);animation:none;opacity:1;transform:none}

.hud-banner{position:absolute;bottom:14px;left:50%;transform:translateX(-50%);background:rgba(11,15,21,0.85);backdrop-filter:blur(10px);border:1px solid rgba(78,210,198,0.4);border-radius:20px;padding:5px 14px;font-size:11px;font-weight:600;font-family:ui-monospace,"SF Mono",monospace;color:var(--paper);display:flex;align-items:center;gap:6px;pointer-events:none;transition:all .2s;white-space:normal;text-align:center;line-height:1.35;max-width:92%;width:max-content;z-index:10;box-shadow:0 4px 16px rgba(0,0,0,0.5)}
.hud-banner.warn{border-color:var(--hold);color:var(--hold)}
.hud-banner.good{border-color:var(--pass);color:var(--pass)}
.hud-spirit{position:absolute;top:46px;right:14px;background:rgba(11,15,21,0.8);backdrop-filter:blur(8px);border:1px solid var(--rule);border-radius:14px;padding:3px 8px;font-size:9.5px;font-family:ui-monospace,"SF Mono",monospace;color:var(--dim);display:flex;align-items:center;gap:5px;pointer-events:none;z-index:10}
.hud-spirit .spirit-bubble{width:6px;height:6px;border-radius:50%;background:var(--pass);transition:background .2s}
.hud-spirit.tilted{border-color:var(--hold);color:var(--hold)}
.hud-spirit.tilted .spirit-bubble{background:var(--hold)}

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
#zoom-bar{display:flex;align-items:center;gap:10px;padding:8px 16px 0}
#zoom-bar input[type=range]{flex:1;accent-color:var(--key)}
#zoom-bar .zoom-read{font-family:ui-monospace,"SF Mono",monospace;font-size:12px;color:var(--key);min-width:74px;text-align:right}
#zoom-bar button{border:1px solid var(--rule);background:var(--rise);color:var(--paper);border-radius:var(--radius-sm);padding:4px 8px;font-size:12px;cursor:pointer}
.fb-row{display:flex;flex-wrap:wrap;gap:6px;padding:8px 16px 12px}
.fb-row input,.fb-row select{flex:1 1 90px;min-width:0;background:var(--rise);border:1px solid var(--rule);border-radius:var(--radius-sm);padding:7px 8px;color:#FFF;font-size:12px}
.fb-row button{border:1px solid var(--rule);background:var(--rise);color:var(--paper);border-radius:var(--radius-sm);padding:7px 10px;font-size:12px;cursor:pointer}
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
    <canvas id="ar-debug" title="Detector view (540px frame sent to server). Tap to hide."></canvas>
    <div class="ar-hud-overlay">
      <div class="hud-chip"><span class="radar"></span><span id="hud-status">SEARCHING</span></div>
      <div class="hud-verdict" id="hud-verdict">SPRT IDLE</div>
    </div>
    <div id="hud-guidance" class="hud-banner">Align card inside viewfinder template</div>
    <div id="hud-spirit-level" class="hud-spirit"><span class="spirit-bubble"></span><span id="spirit-deg">0° LEVEL</span></div>
  </div>
  <!-- Reach: real zoom where the camera supports it, capture-crop where it
       does not, plus a camera picker so a telephoto module can be chosen. -->
  <div id="zoom-bar">
    <button type="button" id="zoom-out" title="Zoom out">&minus;</button>
    <input type="range" id="zoom" min="1" max="8" step="0.1" value="1" aria-label="Zoom">
    <button type="button" id="zoom-in" title="Zoom in">+</button>
    <span class="zoom-read" id="zoom-read">1.0&times;</span>
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
    <div class="form-row">
      <label>Camera (a telephoto module reaches further than any crop)</label>
      <select id="setting-camera"><option value="">Default rear camera</option></select>
    </div>
    <div class="form-row">
      <label>Owner Token (your devices only: saves confirmed photos to the private inbox)</label>
      <input type="password" id="setting-owner-token" autocomplete="off">
    </div>
    <button type="button" class="btn-save" id="setting-save">Save & Ping Container</button>
  </div>
</div>

<script>
const $=s=>document.querySelector(s);
let currentMode = 'ar';
let backendUrl = localStorage.getItem('bakugo_backend_url') || window.location.origin;
// Tenant identity is issued by the server as an HttpOnly cookie. A Bearer token
// is stored only when the backend is on another origin (the mobile shell),
// where that cookie is not sent.
try { localStorage.removeItem('bakugo_device_id'); } catch (e) {}
let deviceId = '';
function isCrossOrigin() {
  try { return new URL(backendUrl, window.location.href).origin !== window.location.origin; }
  catch (e) { return false; }
}
function tokenKey() { return 'bakugo_device_token:' + backendUrl; }
function api(path, opts) {
  opts = Object.assign({ credentials: 'include' }, opts || {});
  const headers = Object.assign({}, opts.headers || {});
  if (isCrossOrigin()) {
    try {
      const tok = localStorage.getItem(tokenKey());
      if (tok) headers['Authorization'] = 'Bearer ' + tok;
    } catch (e) {}
  }
  try {
    const owner = localStorage.getItem('bakugo_owner_token');
    if (owner) headers['X-Bakugo-Owner'] = owner;
  } catch (e) {}
  opts.headers = headers;
  return fetch(backendUrl + path, opts).then(r => {
    const issued = r.headers.get('X-Device-Token');
    if (issued && isCrossOrigin()) {
      try { localStorage.setItem(tokenKey(), issued); } catch (e) {}
    }
    return r;
  });
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
  api('/config')
    .then(r => r.json())
    .then(d => {
      if (d.device_id) deviceId = d.device_id;
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

// Gyroscope spirit level tracking (DeviceOrientation)
let deviceTiltDeg = 0;
function setupGyroscope() {
  if (window.DeviceOrientationEvent) {
    window.addEventListener('deviceorientation', (e) => {
      if (e.beta == null || e.gamma == null) return;
      const b = e.beta, g = e.gamma;
      const pitchDev = Math.abs(b) > 45 ? Math.abs(Math.abs(b) - 90) : Math.abs(b);
      const rollDev = Math.abs(g);
      const tilt = Math.round(Math.hypot(pitchDev, rollDev));
      deviceTiltDeg = Math.min(90, Math.max(0, tilt));
      const spirit = $('#hud-spirit-level'), spiritText = $('#spirit-deg');
      if (spirit && spiritText) {
        spiritText.textContent = `${deviceTiltDeg}° ${deviceTiltDeg <= 16 ? 'LEVEL' : 'TILT'}`;
        spirit.classList.toggle('tilted', deviceTiltDeg > 16);
      }
    }, true);
  }
}
setupGyroscope();

// Live AR Loop & WebRTC Camera Stream
let videoStream = null, arInterval = null, arAnimFrame = null, isPushing = false;
// The loop's own health. A silent catch here is why "SEARCHING" could mean
// anything; every failure now lands in arStats and on screen.
let arWatchdog = null, pushCrop = null, pushStartedAt = 0;
// Reach: optical zoom when the track exposes it, otherwise a capture crop
// (which is not extra detail, but does spend the 540 px budget on the card).
let videoTrack = null, zoomCaps = null, zoomLevel = 1, zoomIsOptical = false;
let cameraId = null;
try { cameraId = localStorage.getItem('bakugo_camera_id') || null; } catch (e) {}
const arStats = { pushes: 0, ok: 0, fail: 0, status: 0, ms: 0, bytes: 0, error: '', video: '',
                  lastOkAt: 0, timeouts: 0, stale: 0, still: '', msAvg: 0, focus: '', aim: '', shift: '' };
// One request that never settles used to end the session: isPushing stayed
// true, the interval kept returning early, and the HUD and inset froze on the
// last good frame while the camera carried on. Bound the wait, and reset the
// flag if a request outlives even that.
const PUSH_TIMEOUT_MS = 4000;
const PUSH_STUCK_MS = 8000;
const OVERLAY_STALE_MS = 800;
// On shop signal a push can take a second or more end to end; a fixed 800 ms
// limit then drops the box between every pair of answers and it flickers.
// "Stale" is measured against how long answers are actually taking here.
function overlayStaleMs() {
  const typical = arStats.msAvg || 0;
  return Math.min(PUSH_TIMEOUT_MS, Math.max(OVERLAY_STALE_MS, 2.5 * typical));
}
let lastSettled = false, lastLocked = false, autoCaptured = false, autoCaptureTimer = null;
// Declared here, before anything can start the camera: a still upload in
// flight, and the screen wake lock held while the camera is up.
let stillBusy = false, wakeLock = null;
let lastHUDData = null, currentQuad = null, targetQuad = null;

const arVideo = $('#ar-video'), arCanvas = $('#ar-canvas'), ctx = arCanvas.getContext('2d');
const offscreenCanvas = document.createElement('canvas'), offCtx = offscreenCanvas.getContext('2d');
// Debug inset: the exact frame the server analysed, with the raw quad it
// returned in that frame's own pixel coordinates. If the red box is right here
// but the teal overlay is wrong, the fault is display mapping; if the red box
// is also wrong, the fault is the detector.
const arDebug = $('#ar-debug'), dbgCtx = arDebug.getContext('2d');
arDebug.style.pointerEvents = 'auto';
arDebug.onclick = () => arDebug.classList.add('hidden');
// WHAT THE USER FRAMES IS WHAT THE DETECTOR GETS.
//
// #ar-video is object-fit: cover, so a landscape 1920x1080 stream in a
// portrait viewport shows less than half its width -- but the whole frame was
// being sent. The card the user lined up inside the guides arrived roughly
// half its apparent size, surrounded by scene they could not see. coverCrop()
// is the visible rectangle in video pixels; arTick sends exactly that, and
// drawARHUD maps the returned quad back through it.
// A 2.16.0 server wrote float('inf') as the bare token Infinity (one side of
// a soft frame had no finite information floor), JSON.parse threw on every
// answer, and the live view said PUSH FAILED while the server was tracking.
// The server now sends strict JSON; this keeps an older one usable.
// Returns undefined when the text is not JSON at all.
function parseReply(text) {
  try { return JSON.parse(text); } catch (e) { /* fall through */ }
  let t = text;
  for (const bad of ['-Infinity', 'Infinity', 'NaN']) {
    for (const lead of [': ', ', ', '[']) t = t.split(lead + bad).join(lead + 'null');
  }
  try { return JSON.parse(t); } catch (e) { return undefined; }
}

// Focus. Close-up, a phone left to itself hunts or settles behind the card;
// ask for continuous autofocus where the track offers it, and let a tap on
// the preview say where to focus.
function focusCaps() {
  try { return (videoTrack && videoTrack.getCapabilities && videoTrack.getCapabilities()) || {}; }
  catch (e) { return {}; }
}
async function startContinuousFocus() {
  const modes = focusCaps().focusMode || [];
  if (modes.indexOf('continuous') >= 0) {
    try { await videoTrack.applyConstraints({ advanced: [{ focusMode: 'continuous' }] }); } catch (e) {}
  }
}
async function focusAt(clientX, clientY) {
  if (!videoTrack) return false;
  const caps = focusCaps();
  const modes = caps.focusMode || [];
  if (!('pointsOfInterest' in caps) && modes.indexOf('single-shot') < 0) return false;
  const vw = arVideo.videoWidth, vh = arVideo.videoHeight;
  const box = arVideo.getBoundingClientRect();
  if (!vw || !vh || !box.width || !box.height) return false;
  // the preview is object-fit: cover -- map the tap into video coordinates
  const s = Math.max(box.width / vw, box.height / vh);
  const w = box.width / s, h = box.height / s;
  const x = ((vw - w) / 2 + (clientX - box.left) / s) / vw;
  const y = ((vh - h) / 2 + (clientY - box.top) / s) / vh;
  const c = { pointsOfInterest: [{ x: Math.min(1, Math.max(0, x)), y: Math.min(1, Math.max(0, y)) }] };
  if (modes.indexOf('single-shot') >= 0) c.focusMode = 'single-shot';
  else if (modes.indexOf('continuous') >= 0) c.focusMode = 'continuous';
  try {
    await videoTrack.applyConstraints({ advanced: [c] });
    arStats.focus = 'tap ' + x.toFixed(2) + ',' + y.toFixed(2);
    // a single-shot lock is right for this card; go back to continuous a
    // little later so the next card is not stuck at this distance
    if (c.focusMode === 'single-shot') setTimeout(startContinuousFocus, 4000);
    return true;
  } catch (e) {
    arStats.focus = 'tap refused: ' + ((e && e.name) || e);
    return false;
  }
}
// Tap a card to pick it. A shop counter has several cards and the one under
// the reticle is not always the one you want; the tap goes to the server with
// the next frame (as a point in that frame) and the session starts over on
// the card there. The same tap sets the focus point.
let pendingAim = null;
function videoPointAt(clientX, clientY) {
  const vw = arVideo.videoWidth, vh = arVideo.videoHeight;
  const box = arVideo.getBoundingClientRect();
  if (!vw || !vh || !box.width || !box.height) return null;
  const s = Math.max(box.width / vw, box.height / vh);
  const w = box.width / s, h = box.height / s;
  return { x: (vw - w) / 2 + (clientX - box.left) / s, y: (vh - h) / 2 + (clientY - box.top) / s };
}
function selectCardAt(clientX, clientY) {
  const p = videoPointAt(clientX, clientY);
  const c = pushCrop || coverCrop();
  if (!p || !c) return false;
  const ax = (p.x - c.x) / c.w, ay = (p.y - c.y) / c.h;
  if (ax < 0 || ax > 1 || ay < 0 || ay > 1) return false;
  pendingAim = { x: ax, y: ay };
  targetQuad = null;
  quadBase = null;
  currentQuad = null;
  lastLocked = false;
  arStats.aim = ax.toFixed(2) + ',' + ay.toFixed(2);
  const chip = $('#hud-status');
  if (chip) chip.textContent = 'PICKING CARD';
  return true;
}
arVideo.addEventListener('click', (ev) => {
  haptic('tap');
  selectCardAt(ev.clientX, ev.clientY);
  focusAt(ev.clientX, ev.clientY);
});

// WHERE THE VIEW HAS GONE SINCE THE FRAME AN ANSWER IS ABOUT.
//
// On shop LTE an answer comes back half a second to a second and a half after
// its frame was taken, and a hand-held phone has moved by then: in the owner's
// screenshots of 2.18.0 the outline sat 40-90 px beside the card it had found
// correctly, where the card had been. The page keeps a tiny grey copy of each
// frame it sends and, while the outline is up, measures how far the live
// preview has shifted from that copy (block matching on a 96 px thumbnail,
// coarse to fine, ~2 ms), and moves the outline with it. Translation only;
// a phone held over a counter mostly slides and tips, which both read as that.
let MOTION_COMP = true;
const MOTION_W = 96;
let quadBase = null, quadThumb = null, lastMotionAt = 0;
const motionCanvas = document.createElement('canvas');
const motionCtx = motionCanvas.getContext('2d', { willReadFrequently: true });
function grabThumb(crop) {
  if (!crop || !arVideo.videoWidth) return null;
  const w = MOTION_W, h = Math.max(8, Math.round(MOTION_W * crop.h / crop.w));
  if (motionCanvas.width !== w || motionCanvas.height !== h) { motionCanvas.width = w; motionCanvas.height = h; }
  try { motionCtx.drawImage(arVideo, crop.x, crop.y, crop.w, crop.h, 0, 0, w, h); }
  catch (e) { return null; }
  const d = motionCtx.getImageData(0, 0, w, h).data;
  const g = new Float32Array(w * h);
  let sum = 0;
  for (let i = 0, j = 0; j < g.length; i += 4, j++) { g[j] = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]; sum += g[j]; }
  const mean = sum / g.length;
  for (let j = 0; j < g.length; j++) g[j] -= mean;
  return { g: g, w: w, h: h, crop: { x: crop.x, y: crop.y, w: crop.w, h: crop.h } };
}
function halfThumb(t) {
  const w = t.w >> 1, h = t.h >> 1, g = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = 2 * y * t.w + 2 * x;
    g[y * w + x] = 0.25 * (t.g[i] + t.g[i + 1] + t.g[i + t.w] + t.g[i + t.w + 1]);
  }
  return { g: g, w: w, h: h };
}
// mean |a(x) - b(x + d)| over the overlap; Infinity when it is under half the frame
function sadAt(a, b, dx, dy) {
  const x0 = Math.max(0, -dx), x1 = Math.min(a.w, a.w - dx);
  const y0 = Math.max(0, -dy), y1 = Math.min(a.h, a.h - dy);
  if ((x1 - x0) * (y1 - y0) < 0.5 * a.w * a.h) return Infinity;
  let s = 0;
  for (let y = y0; y < y1; y++) {
    const ra = y * a.w, rb = (y + dy) * b.w + dx;
    for (let x = x0; x < x1; x++) s += Math.abs(a.g[ra + x] - b.g[rb + x]);
  }
  return s / ((x1 - x0) * (y1 - y0));
}
// Shift of b against a, in a's pixels, or null when the view has too little
// texture to tell (a blank counter, a frame blurred right through).
function estimateShift(a, b) {
  if (!a || !b || a.w !== b.w || a.h !== b.h) return null;
  const A = halfThumb(a), B = halfThumb(b);
  // up to 30% of the view either way (the screenshots showed 4-9% of it)
  const R = Math.round(0.3 * A.w);
  let best = Infinity, bx = 0, by = 0;
  const costs = [];
  for (let dy = -R; dy <= R; dy++) for (let dx = -R; dx <= R; dx++) {
    const c = sadAt(A, B, dx, dy);
    if (!isFinite(c)) continue;
    costs.push(c);
    if (c < best) { best = c; bx = dx; by = dy; }
  }
  if (!costs.length) return null;
  // a best match on the edge of the search is a view that moved further
  // than it reaches, or a wrong match: not a shift to trust
  if (Math.abs(bx) === R || Math.abs(by) === R) return null;
  costs.sort((p, q) => p - q);
  const typical = costs[costs.length >> 1];
  if (!(best < 0.7 * typical)) return null;
  let fb = Infinity, fx = 2 * bx, fy = 2 * by;
  for (let dy = 2 * by - 2; dy <= 2 * by + 2; dy++) for (let dx = 2 * bx - 2; dx <= 2 * bx + 2; dx++) {
    const c = sadAt(a, b, dx, dy);
    if (c < fb) { fb = c; fx = dx; fy = dy; }
  }
  return { dx: fx, dy: fy };
}
function updateMotion(force) {
  if (!MOTION_COMP || !quadBase || !quadThumb || !targetQuad) return;
  const now = Date.now();
  if (!force && now - lastMotionAt < 80) return;
  lastMotionAt = now;
  const crop = coverCrop();
  const c0 = quadThumb.crop;
  if (!crop || crop.x !== c0.x || crop.y !== c0.y || crop.w !== c0.w || crop.h !== c0.h) return;
  const s = estimateShift(quadThumb, grabThumb(crop));
  if (!s) return;
  const kx = crop.w / quadThumb.w, ky = crop.h / quadThumb.h;
  targetQuad = quadBase.map(p => [p[0] + s.dx * kx, p[1] + s.dy * ky]);
  arStats.shift = Math.round(s.dx * kx) + ',' + Math.round(s.dy * ky);
}

function coverCrop() {
  const vw = arVideo.videoWidth, vh = arVideo.videoHeight;
  if (!vw || !vh) return null;
  const bw = arVideo.clientWidth || vw, bh = arVideo.clientHeight || vh;
  const scale = Math.max(bw / vw, bh / vh);
  let w = Math.min(vw, Math.round(bw / scale));
  let h = Math.min(vh, Math.round(bh / scale));
  // Capture crop when the camera has no zoom of its own. It adds no detail
  // the sensor did not capture, but the 540 px the detector gets are spent on
  // the card instead of the table, which is most of the battle at distance.
  if (!zoomIsOptical && zoomLevel > 1) {
    w = Math.max(64, Math.round(w / zoomLevel));
    h = Math.max(64, Math.round(h / zoomLevel));
  }
  return { x: Math.round((vw - w) / 2), y: Math.round((vh - h) / 2), w: w, h: h };
}

async function setZoom(z) {
  zoomLevel = Math.max(1, Math.min(8, Number(z) || 1));
  zoomIsOptical = false;
  if (videoTrack && zoomCaps) {
    // Map 1..8 onto what this camera actually offers.
    const span = zoomCaps.max - zoomCaps.min;
    const want = zoomCaps.min + span * Math.min(1, (zoomLevel - 1) / 7);
    try {
      await videoTrack.applyConstraints({ advanced: [{ zoom: want }] });
      zoomIsOptical = true;
      const got = videoTrack.getSettings ? videoTrack.getSettings().zoom : want;
      $('#zoom-read').textContent = (got || want).toFixed(1) + 'x cam';
    } catch (e) {
      zoomIsOptical = false;
    }
  }
  if (!zoomIsOptical) $('#zoom-read').textContent = zoomLevel.toFixed(1) + 'x crop';
  sizeARCanvases();
}

// How much the camera (or the crop) magnifies, as a plain factor.
function zoomFactor() {
  if (zoomIsOptical && videoTrack && zoomCaps) {
    const got = (videoTrack.getSettings && videoTrack.getSettings().zoom) || zoomCaps.min;
    return Math.max(1, got / Math.max(zoomCaps.min, 1e-3));
  }
  return zoomLevel;
}

// The slider level that gives a zoom factor f.
function levelForFactor(f) {
  if (zoomIsOptical && zoomCaps) {
    const span = zoomCaps.max - zoomCaps.min;
    const want = Math.max(1, zoomCaps.min) * f;
    return Math.max(1, Math.min(8, 1 + 7 * (want - zoomCaps.min) / Math.max(span, 1e-3)));
  }
  return Math.max(1, Math.min(8, f));
}

// Zoom in by itself when the card is too coarse to measure live. At a
// distance the camera focuses (card about a third of the view wide) the
// 540 px live frame has 3.3-4.2 px/mm (2.21.1 field session) and live needs
// 4.5, so it never measured and never settled; only Freeze did. Closer,
// it cannot focus. Zoom from where it focuses: to ~6 px/mm, at most 2x (past
// that some phones switch to a telephoto that focuses further away still),
// and never when the user has zoomed by hand in the last 10 s.
const LIVE_MIN_PXMM = 4.5, LIVE_TARGET_PXMM = 6.0, AUTO_ZOOM_MAX = 2.0;
let userZoomAt = 0, autoZoomAt = 0, coarseRuns = 0;
function maybeAutoZoom(d) {
  if (!d || !d.ok || !d.tracking || !d.live_px_per_mm || !d.card_frac) { coarseRuns = 0; return; }
  if (d.live_px_per_mm >= LIVE_MIN_PXMM) { coarseRuns = 0; return; }
  if (++coarseRuns < 2) return;                      // two frames in a row
  const now = Date.now();
  if (now - userZoomAt < 10000 || now - autoZoomAt < 2500) return;
  const cur = zoomFactor();
  const want = Math.min(AUTO_ZOOM_MAX,
                        cur * LIVE_TARGET_PXMM / d.live_px_per_mm,
                        cur * 0.7 / d.card_frac);     // keep the card in view
  if (want < cur * 1.15) return;
  autoZoomAt = now;
  coarseRuns = 0;
  const lvl = levelForFactor(want);
  $('#zoom').value = lvl;
  setZoom(lvl);
  autoZoomNote = now;
}
let autoZoomNote = 0;

async function listCameras() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
  const sel = $('#setting-camera');
  if (!sel) return;
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const cams = devices.filter(d => d.kind === 'videoinput');
    sel.innerHTML = '<option value="">Default rear camera</option>' + cams.map((d, i) =>
      `<option value="${d.deviceId}">${esc(d.label || 'Camera ' + (i + 1))}</option>`).join('');
    if (cameraId) sel.value = cameraId;
  } catch (e) {}
}

function sizeARCanvases() {
  const crop = coverCrop();
  if (!crop) return false;
  arStats.video = arVideo.videoWidth + 'x' + arVideo.videoHeight;
  if (arCanvas.width !== arVideo.videoWidth || arCanvas.height !== arVideo.videoHeight) {
    arCanvas.width = arVideo.videoWidth;
    arCanvas.height = arVideo.videoHeight;
  }
  // 540 px ACROSS, not 540 on the long side: the tracking budget has always
  // been the frame's width, and a portrait camera (1080x1920 here) has a tall
  // crop, so scaling its long side to 540 would shrink the card below what the
  // uncropped frame used to carry. Height is capped so an extreme crop cannot
  // cost more than the old full frame did.
  const s = Math.min(1, 540 / crop.w, 960 / crop.h);
  const ow = Math.max(64, Math.round(crop.w * s)), oh = Math.max(64, Math.round(crop.h * s));
  if (offscreenCanvas.width !== ow || offscreenCanvas.height !== oh) {
    offscreenCanvas.width = ow;
    offscreenCanvas.height = oh;
  }
  return true;
}

// Idempotent, and called from several events plus a watchdog: a single
// loadedmetadata handler is one missed event away from a camera that previews
// forever without ever pushing a frame.
function startARLoop() {
  if (currentMode !== 'ar' || !sizeARCanvases()) return false;
  if (!arInterval) arInterval = setInterval(arTick, 180);
  if (!arAnimFrame) arAnimFrame = requestAnimationFrame(renderARHUDContinuous);
  return true;
}

function setPushStatus(text) {
  const chip = $('#hud-status');
  if (chip && !lastHUDData) chip.textContent = text;
}

function drawDebugInset(d) {
  if (arDebug.classList.contains('hidden')) return;
  arDebug.width = offscreenCanvas.width || 300;
  arDebug.height = offscreenCanvas.height || 150;
  dbgCtx.fillStyle = '#000';
  dbgCtx.fillRect(0, 0, arDebug.width, arDebug.height);
  if (offscreenCanvas.width) dbgCtx.drawImage(offscreenCanvas, 0, 0);
  if (d && d.quad && d.quad.length === 4) {
    dbgCtx.strokeStyle = '#FF3B3B';
    dbgCtx.lineWidth = 3;
    dbgCtx.beginPath();
    dbgCtx.moveTo(d.quad[0][0], d.quad[0][1]);
    for (let i = 1; i < 4; i++) dbgCtx.lineTo(d.quad[i][0], d.quad[i][1]);
    dbgCtx.closePath();
    dbgCtx.stroke();
  }
  dbgCtx.fillStyle = 'rgba(0,0,0,0.65)';
  dbgCtx.fillRect(0, 0, arDebug.width, 40);
  dbgCtx.font = '13px monospace';
  dbgCtx.fillStyle = '#fff';
  dbgCtx.fillText(`${offscreenCanvas.width}x${offscreenCanvas.height} of ${arStats.video} ${d && d.tracking ? 'TRACK' : 'none'}`, 5, 15);
  const age = arStats.lastOkAt ? (Date.now() - arStats.lastOkAt) / 1000 : 0;
  dbgCtx.fillStyle = arStats.error ? '#FF6B6B' : (age > 1 ? '#E0B341' : '#9FE8DF');
  dbgCtx.fillText(arStats.error
    ? arStats.error.slice(0, 40)
    : `${arStats.ok}/${arStats.pushes} ok  ${arStats.status}  ${arStats.ms}ms` +
      `${arStats.serverMs != null ? ' (srv ' + arStats.serverMs + ')' : ''}  ` +
      `${Math.round(arStats.bytes / 1024)}k  ${age > 1 ? age.toFixed(1) + 's old' : 'live'}`, 5, 32);
}

async function startARStream() {
  if (videoStream) return;
  try {
    synth.init();
    const video = cameraId
      ? { deviceId: { exact: cameraId }, width: { ideal: 3840 }, height: { ideal: 2160 } }
      : { facingMode: 'environment', width: { ideal: 3840 }, height: { ideal: 2160 } };
    videoStream = await navigator.mediaDevices.getUserMedia({ video: video, audio: false });
    videoTrack = videoStream.getVideoTracks()[0] || null;
    zoomCaps = null;
    if (videoTrack && videoTrack.getCapabilities) {
      const caps = videoTrack.getCapabilities();
      if (caps && caps.zoom && caps.zoom.max > caps.zoom.min) zoomCaps = caps.zoom;
    }
    $('#zoom-read').textContent = zoomCaps ? '1.0x cam' : '1.0x crop';
    startContinuousFocus();
    listCameras();
    arVideo.srcObject = videoStream;
    holdScreenOn();
    arVideo.onloadedmetadata = startARLoop;
    arVideo.onplaying = startARLoop;
    arVideo.onresize = () => { sizeARCanvases(); startARLoop(); };
    try { await arVideo.play(); } catch (e) { /* autoplay attribute covers this */ }
    startARLoop();
    // The camera can be previewing while the loop never started (a missed
    // event, a late layout, a zero-size video). Keep trying, and say so on
    // screen rather than sitting on "SEARCHING".
    if (arWatchdog) clearInterval(arWatchdog);
    const since = Date.now();
    arWatchdog = setInterval(() => {
      if (!videoStream || currentMode !== 'ar') return;
      startARLoop();
      if (isPushing && Date.now() - pushStartedAt > PUSH_STUCK_MS) {
        // belt and braces: the abort above should have fired already
        isPushing = false;
        arStats.fail++;
        arStats.error = 'request stuck; loop restarted';
        console.warn('[bakugo] /ar/push stuck, resetting');
      }
      if (Date.now() - since > 3000 && !arInterval) {
        setPushStatus(arVideo.videoWidth ? 'LOOP NOT RUNNING' : 'NO CAMERA FRAMES');
      }
    }, 500);
  } catch(err) {
    $('#hud-status').textContent = 'CAMERA BLOCKED';
    arStats.error = 'getUserMedia: ' + (err && err.name || err);
    console.warn("Camera streaming unavailable:", err);
  }
}

function stopARStream() {
  if (arInterval) { clearInterval(arInterval); arInterval = null; }
  if (arWatchdog) { clearInterval(arWatchdog); arWatchdog = null; }
  if (arAnimFrame) { cancelAnimationFrame(arAnimFrame); arAnimFrame = null; }
  releaseScreen();
  if (videoStream) {
    videoStream.getTracks().forEach(t => t.stop());
    videoStream = null;
    videoTrack = null;
    zoomCaps = null;
    arVideo.srcObject = null;
  }
  lastHUDData = null;
  currentQuad = null;
  targetQuad = null;
  quadBase = null;
  ctx.clearRect(0, 0, arCanvas.width, arCanvas.height);
}

async function arTick() {
  // The still upload gets the whole link while it runs.
  if (isPushing || stillBusy || currentMode !== 'ar') return;
  if (!sizeARCanvases()) { setPushStatus('NO CAMERA FRAMES'); return; }
  isPushing = true;
  const t0 = Date.now();
  pushStartedAt = t0;
  const abort = new AbortController();
  const timer = setTimeout(() => abort.abort(), PUSH_TIMEOUT_MS);
  try {
    const crop = coverCrop();
    pushCrop = crop;
    offCtx.drawImage(arVideo, crop.x, crop.y, crop.w, crop.h,
                     0, 0, offscreenCanvas.width, offscreenCanvas.height);
    const thumb = MOTION_COMP ? grabThumb(crop) : null;
    const blob = await new Promise(res => offscreenCanvas.toBlob(res, 'image/jpeg', 0.75));
    if (!blob) {
      arStats.fail++;
      arStats.error = 'canvas gave no frame';
      setPushStatus('FRAME CAPTURE FAILED');
      drawDebugInset(null);
      return;
    }
    arStats.bytes = blob.size;

    const fd = new FormData();
    fd.append('holder', $('#holder').value);
    fd.append('lens', $('#lens').value);
    fd.append('image', blob, 'frame.jpg');
    // camera pixels per pushed pixel: the guidance can then say what a
    // full-resolution Measure Card photo would have
    fd.append('source_scale', String(crop.w / (offscreenCanvas.width || crop.w)));
    fd.append('zoom', zoomFactor().toFixed(2));
    if (pendingAim) {
      fd.append('aim_x', pendingAim.x.toFixed(4));
      fd.append('aim_y', pendingAim.y.toFixed(4));
      pendingAim = null;
    }

    const res = await api('/ar/push', {
      method: 'POST',
      body: fd,
      signal: abort.signal
    });
    arStats.pushes++;
    arStats.status = res.status;
    arStats.ms = Date.now() - t0;
    arStats.msAvg = arStats.msAvg ? 0.8 * arStats.msAvg + 0.2 * arStats.ms : arStats.ms;
    const text = await res.text();
    const d = parseReply(text);
    if (d === undefined) throw new Error('HTTP ' + res.status + ' not JSON: ' + text.slice(0, 80));
    if (d && d.ok) {
      arStats.ok++;
      arStats.error = '';
      arStats.lastOkAt = Date.now();
      arStats.serverMs = d.server_ms;
    } else if (d && d.busy) {
      // the server is still on the previous frame: not an error, and the
      // next tick sends a newer frame
      arStats.busy = (arStats.busy || 0) + 1;
      setPushStatus('SERVER CATCHING UP');
      return;
    } else {
      arStats.fail++;
      arStats.error = (d && d.error ? d.error : 'HTTP ' + res.status).slice(0, 140);
      setPushStatus('SERVER REFUSED FRAME');
    }
    drawDebugInset(d);
    drawARHUD(d, thumb);
    maybeAutoZoom(d);
  } catch(e) {
    // A dropped frame and a broken loop look identical until one of them is
    // reported: say which, on screen and in the console.
    arStats.fail++;
    arStats.ms = Date.now() - t0;
    const aborted = e && e.name === 'AbortError';
    if (aborted) arStats.timeouts++;
    arStats.error = aborted
      ? `no reply in ${PUSH_TIMEOUT_MS / 1000}s (${arStats.timeouts} timeouts)`
      : String((e && e.message) || e).slice(0, 140);
    console.warn('[bakugo] /ar/push failed:', e);
    setPushStatus(aborted ? 'PUSH TIMED OUT' : 'PUSH FAILED');
    drawDebugInset(null);
  } finally {
    clearTimeout(timer);
    isPushing = false;
  }
}

function drawARHUD(d, thumb) {
  if (!d || !d.ok) {
    lastHUDData = null;
    targetQuad = null;
    quadBase = null;
    return;
  }
  lastHUDData = d;

  // The frame was a crop of the video, so the quad maps back through it.
  const crop = pushCrop || { x: 0, y: 0, w: arCanvas.width, h: arCanvas.height };
  const scaleX = crop.w / (offscreenCanvas.width || 1);
  const scaleY = crop.h / (offscreenCanvas.height || 1);

  if (d.tracking && d.quad && d.quad.length === 4) {
    targetQuad = d.quad.map(pt => [crop.x + pt[0] * scaleX, crop.y + pt[1] * scaleY]);
    // the answer is about the frame sent; move it to where that is now
    quadBase = targetQuad.map(p => [p[0], p[1]]);
    quadThumb = thumb || null;
    updateMotion(true);
    if (!currentQuad) {
      currentQuad = targetQuad.map(p => [...p]);
    }
    if (!lastLocked) { synth.playLock(); haptic('lock'); lastLocked = true; }

    const chip = $('#hud-status');
    const chipBox = document.querySelector('.hud-chip');
    const est = d.grade_estimate
      ? `~${d.grade_estimate}${d.grade_confidence != null ? ` (${Math.round(d.grade_confidence * 100)}%)` : ''}`
      : null;
    if (!d.ratio) {
      chip.textContent = 'TRACKING';
      if (chipBox) chipBox.classList.remove('settled');
    } else if (d.settled && stillRefusedAt) {
      chip.textContent = `live ${d.ratio.toFixed(1)}% · photo refused, not confirmed`;
      if (chipBox) chipBox.classList.remove('settled');
    } else if (d.settled) {
      chip.textContent = `${d.ratio.toFixed(1)}%` + (est ? ` · ${est}` : '') + (d.grade_ceiling ? ` · range ${d.grade_ceiling}` : '');
      if (chipBox) chipBox.classList.add('settled');
    } else {
      const n = d.measured_frames || 0;
      chip.textContent = `${d.ratio.toFixed(1)}%` + (est ? ` · ${est}` : '') + ` · narrowing (${n} view${n === 1 ? '' : 's'})`;
      if (chipBox) chipBox.classList.remove('settled');
    }
  } else {
    targetQuad = null;
    quadBase = null;
    lastLocked = false;
    $('#hud-status').textContent = 'SEARCHING';
    const chipBox = document.querySelector('.hud-chip');
    if (chipBox) chipBox.classList.remove('settled');
  }

  // Guidance banner coaching
  const banner = $('#hud-guidance');
  if (banner) {
    if (d.settled && stillRefusedAt) {
      banner.textContent = '⚠ the photo was refused -- see below; adjust and tap Freeze again';
      banner.className = 'hud-banner warn';
    } else if (d.settled) {
      banner.textContent = '✓ Target settled · Auto-capturing metrology';
      banner.className = 'hud-banner good';
    } else if (Date.now() - autoZoomNote < 3500) {
      banner.textContent = `zoomed in to ${zoomFactor().toFixed(1)}x so the live view can measure -- keep the phone where it is`;
      banner.className = 'hud-banner';
    } else if (d.guidance && d.guidance.length > 0) {
      banner.textContent = '⚠ ' + d.guidance[0];
      banner.className = 'hud-banner warn';
    } else if (d.tracking) {
      banner.textContent = '⚡ Tracking · Hold steady for multi-view convergence';
      banner.className = 'hud-banner';
    } else {
      banner.textContent = 'Align card inside viewfinder template';
      banner.className = 'hud-banner';
    }
  }

  // Peak-sharpness auto-capture on settlement
  if (d.settled && !lastSettled) {
    synth.playSettle();
    haptic('settle');
    lastSettled = true;
    if (!autoCaptured) {
      autoCaptured = true;
      clearTimeout(autoCaptureTimer);
      autoCaptureTimer = setTimeout(() => {
        if (currentMode === 'ar' && lastHUDData && lastHUDData.settled) {
          triggerARFreeze();
        }
      }, 400);
    }
  } else if (!d.settled) {
    lastSettled = false;
    autoCaptured = false;
    stillRefusedAt = 0;
    clearTimeout(autoCaptureTimer);
  }

  $('#hud-verdict').textContent = d.verdict ? `SPRT: ${d.verdict}` : (d.settled ? 'SPRT SETTLED' : 'ACCUMULATING');
}

// 60fps Smooth Canvas Render: Viewfinder reticle, lerped quad, laser caliper sweep, glowing lock
function renderARHUDContinuous() {
  if (currentMode === 'ar') {
    ctx.clearRect(0, 0, arCanvas.width, arCanvas.height);
    const W = arCanvas.width, H = arCanvas.height;

    // A box from an old frame sitting over a live preview is worse than no
    // box: it looks like a detector fault when the loop has simply stopped
    // answering. Drop it, and say the tracking went stale.
    const age = arStats.lastOkAt ? Date.now() - arStats.lastOkAt : 0;
    if (targetQuad && arStats.lastOkAt && age > overlayStaleMs()) {
      targetQuad = null;
      quadBase = null;
      currentQuad = null;
      lastLocked = false;
      arStats.stale++;
      lastHUDData = null;
      $('#hud-status').textContent = arStats.error ? 'RECONNECTING' : 'SEARCHING';
      const chipBox = document.querySelector('.hud-chip');
      if (chipBox) chipBox.classList.remove('settled');
    }

    updateMotion(false);
    if (targetQuad && currentQuad) {
      // Smoothly lerp towards target quad corners
      for (let i = 0; i < 4; i++) {
        currentQuad[i][0] += (targetQuad[i][0] - currentQuad[i][0]) * 0.45;
        currentQuad[i][1] += (targetQuad[i][1] - currentQuad[i][1]) * 0.45;
      }

      const isSettled = lastHUDData && lastHUDData.settled;
      const themeColor = isSettled ? '#4EBA82' : '#4ED2C6';

      // Draw glowing boundary polygon
      ctx.save();
      ctx.strokeStyle = themeColor;
      ctx.lineWidth = 3.5;
      ctx.shadowColor = themeColor;
      ctx.shadowBlur = isSettled ? 16 : 10;
      ctx.beginPath();
      ctx.moveTo(currentQuad[0][0], currentQuad[0][1]);
      for (let i = 1; i < 4; i++) {
        ctx.lineTo(currentQuad[i][0], currentQuad[i][1]);
      }
      ctx.closePath();
      ctx.stroke();

      // Corner crosshairs / brackets
      currentQuad.forEach(pt => {
        ctx.fillStyle = '#FFFFFF';
        ctx.fillRect(pt[0] - 4, pt[1] - 4, 8, 8);
        ctx.strokeStyle = themeColor;
        ctx.strokeRect(pt[0] - 7, pt[1] - 7, 14, 14);
      });

      // Laser Caliper Sweep animation
      const sweepTime = (Date.now() % 2200) / 2200;
      const sFrac = 0.5 - 0.5 * Math.cos(sweepTime * 2 * Math.PI);
      const pLeft = [
        currentQuad[0][0] + sFrac * (currentQuad[3][0] - currentQuad[0][0]),
        currentQuad[0][1] + sFrac * (currentQuad[3][1] - currentQuad[0][1])
      ];
      const pRight = [
        currentQuad[1][0] + sFrac * (currentQuad[2][0] - currentQuad[1][0]),
        currentQuad[1][1] + sFrac * (currentQuad[2][1] - currentQuad[1][1])
      ];

      ctx.strokeStyle = themeColor;
      ctx.lineWidth = 2.5;
      ctx.shadowColor = themeColor;
      ctx.shadowBlur = 10;
      ctx.beginPath();
      ctx.moveTo(pLeft[0], pLeft[1]);
      ctx.lineTo(pRight[0], pRight[1]);
      ctx.stroke();

      // Caliper beam trail
      const grad = ctx.createLinearGradient(pLeft[0], pLeft[1] - 8, pLeft[0], pLeft[1] + 8);
      grad.addColorStop(0, 'rgba(78,210,198,0)');
      grad.addColorStop(0.5, isSettled ? 'rgba(78,186,130,0.22)' : 'rgba(78,210,198,0.22)');
      grad.addColorStop(1, 'rgba(78,210,198,0)');
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.moveTo(pLeft[0], pLeft[1] - 8);
      ctx.lineTo(pRight[0], pRight[1] - 8);
      ctx.lineTo(pRight[0], pRight[1] + 8);
      ctx.lineTo(pLeft[0], pLeft[1] + 8);
      ctx.closePath();
      ctx.fill();

      ctx.restore();
    } else if (W && H) {
      currentQuad = null;
      // Draw Holographic Cybernetic Viewfinder Template
      const cx = W / 2, cy = H / 2;
      const cardH = Math.min(H * 0.70, W * 0.85 * 1.40);
      const cardW = cardH / 1.40;
      const x0 = cx - cardW / 2, y0 = cy - cardH / 2;
      const cornerLen = Math.min(cardW, cardH) * 0.15;

      ctx.save();
      const pulse = 0.5 + 0.5 * Math.sin(Date.now() / 400);

      // Holographic aura
      ctx.fillStyle = `rgba(78, 210, 198, ${0.015 + 0.015 * pulse})`;
      ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(x0, y0, cardW, cardH, 14);
      else ctx.rect(x0, y0, cardW, cardH);
      ctx.fill();

      // Outer dashed guide
      ctx.strokeStyle = `rgba(78, 210, 198, ${0.25 + 0.15 * pulse})`;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([8, 8]);
      ctx.stroke();
      ctx.setLineDash([]);

      // Corner brackets
      ctx.strokeStyle = '#4ED2C6';
      ctx.lineWidth = 3.5;
      ctx.shadowColor = '#4ED2C6';
      ctx.shadowBlur = 8;

      // TL
      ctx.beginPath();
      ctx.moveTo(x0, y0 + cornerLen); ctx.lineTo(x0, y0); ctx.lineTo(x0 + cornerLen, y0);
      ctx.stroke();
      // TR
      ctx.beginPath();
      ctx.moveTo(x0 + cardW - cornerLen, y0); ctx.lineTo(x0 + cardW, y0); ctx.lineTo(x0 + cardW, y0 + cornerLen);
      ctx.stroke();
      // BR
      ctx.beginPath();
      ctx.moveTo(x0 + cardW, y0 + cardH - cornerLen); ctx.lineTo(x0 + cardW, y0 + cardH); ctx.lineTo(x0 + cardW - cornerLen, y0 + cardH);
      ctx.stroke();
      // BL
      ctx.beginPath();
      ctx.moveTo(x0 + cornerLen, y0 + cardH); ctx.lineTo(x0, y0 + cardH); ctx.lineTo(x0, y0 + cardH - cornerLen);
      ctx.stroke();

      // Center crosshair
      ctx.strokeStyle = `rgba(78, 210, 198, ${0.4 + 0.3 * pulse})`;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(cx - 10, cy); ctx.lineTo(cx + 10, cy);
      ctx.moveTo(cx, cy - 10); ctx.lineTo(cx, cy + 10);
      ctx.stroke();

      // Viewfinder label
      ctx.fillStyle = `rgba(78, 210, 198, ${0.6 + 0.2 * pulse})`;
      ctx.font = '11px ui-monospace, "SF Mono", monospace';
      ctx.textAlign = 'center';
      ctx.fillText('VIEWFINDER · 2.5" × 3.5"', cx, y0 - 10);

      ctx.restore();
    }
  }
  arAnimFrame = requestAnimationFrame(renderARHUDContinuous);
}

// The preview stream is capped well below what the sensor can take, and the
// measurement wants every pixel the card can get -- the server crops to the
// card before it caps, so a full-resolution photo is where the reach comes
// from. takePhoto gives that where it is supported; the video frame is the
// fallback.
// WHICH STILL. The camera's own photo (ImageCapture.takePhoto) is a separate
// capture: on the owner's phone (2.19.1, 2160x3840 video) Measure Card and
// Identify both came back "no card here" on cards in plain view, where the
// live frames of the same seconds found them every time. The photo is taken
// with its own focus and field of view, and the card is cropped out of it by
// mapping the tracked outline from the preview -- a guess about how the two
// line up. The video frame is where the outline was found, at the same
// pixels. When it is big enough to measure from (4K: the card ~1000 px
// across, ~17 px/mm), it is the still; the photo is used when the video is
// small.
const STILL_VIDEO_MIN_PIXELS = 3500000;
function stillSource(vw, vh, hasPhoto) {
  return (!hasPhoto || vw * vh >= STILL_VIDEO_MIN_PIXELS) ? 'frame' : 'photo';
}
let lastStillSource = '';
async function grabStill(track) {
  const hasPhoto = !!videoTrack && typeof ImageCapture !== 'undefined';
  lastStillSource = stillSource(arVideo.videoWidth, arVideo.videoHeight, hasPhoto);
  if (lastStillSource === 'photo') {
    try {
      const shot = await new ImageCapture(videoTrack).takePhoto();
      if (shot && shot.size > 1024) {
        arStats.still = 'photo ' + Math.round(shot.size / 1024) + 'k';
        return shot;
      }
    } catch (e) {
      console.warn('[bakugo] takePhoto unavailable, using the video frame:', e);
    }
  }
  lastStillSource = 'frame';
  const c = await sharpestFrame(track);
  arStats.still = 'frame ' + c.width + 'x' + c.height + ' best of ' + STILL_BURST +
    ' (sharp ' + c.sharp.map(v => v.toFixed(0)).join('/') + ')';
  return await new Promise(res => c.toBlob(res, 'image/jpeg', 0.92));
}

// A Freeze takes a few video frames a moment apart and keeps the sharpest.
// One frame is whatever the hand and the focus were doing at that instant;
// the 2.21.0 field stills measured 2.5 to 26 on the same card (Laplacian
// variance at 2400 px), and the soft ones were the refusals.
const STILL_BURST = 4;
const STILL_BURST_GAP_MS = 110;

// Mean squared Laplacian of the card region (or the middle of the view),
// drawn at most 480 px wide: ranks frames of one scene by focus.
function frameSharpness(track) {
  const vw = arVideo.videoWidth, vh = arVideo.videoHeight;
  let x = vw * 0.2, y = vh * 0.2, w = vw * 0.6, h = vh * 0.6;
  if (track && track.quad) {
    const xs = track.quad.map(p => p[0]), ys = track.quad.map(p => p[1]);
    x = Math.max(0, Math.min(...xs)); y = Math.max(0, Math.min(...ys));
    w = Math.min(vw, Math.max(...xs)) - x; h = Math.min(vh, Math.max(...ys)) - y;
    if (w < 32 || h < 32) { x = vw * 0.2; y = vh * 0.2; w = vw * 0.6; h = vh * 0.6; }
  }
  const s = Math.min(1, 480 / w);
  const cw = Math.max(8, Math.round(w * s)), ch = Math.max(8, Math.round(h * s));
  const c = document.createElement('canvas');
  c.width = cw; c.height = ch;
  const g = c.getContext('2d', { willReadFrequently: true });
  g.drawImage(arVideo, x, y, w, h, 0, 0, cw, ch);
  const px = g.getImageData(0, 0, cw, ch).data;
  const lum = new Float32Array(cw * ch);
  for (let i = 0, j = 0; j < lum.length; i += 4, j++) lum[j] = 0.299 * px[i] + 0.587 * px[i + 1] + 0.114 * px[i + 2];
  let sum = 0, n = 0;
  for (let yy = 1; yy < ch - 1; yy++) {
    for (let xx = 1; xx < cw - 1; xx++) {
      const k = yy * cw + xx;
      const l = lum[k - 1] + lum[k + 1] + lum[k - cw] + lum[k + cw] - 4 * lum[k];
      sum += l * l; n++;
    }
  }
  return n ? sum / n : 0;
}

async function sharpestFrame(track) {
  const full = document.createElement('canvas');
  full.width = arVideo.videoWidth;
  full.height = arVideo.videoHeight;
  const ctx = full.getContext('2d');
  let best = -1;
  const seen = [];
  for (let i = 0; i < STILL_BURST; i++) {
    if (i) await new Promise(r => setTimeout(r, STILL_BURST_GAP_MS));
    // measured and drawn in the same turn: the same video frame
    const v = frameSharpness(track);
    seen.push(v);
    if (v > best) { best = v; ctx.drawImage(arVideo, 0, 0); }
  }
  full.sharp = seen;
  return full;
}

// ---- Shop-floor upload budget ----
// A full-resolution still is 3-12 MB, and it went up twice (/measure and
// /identify) over whatever signal the shop has, with no time limit: the button
// could sit on "Calculating" for minutes. Now the phone crops the photo to the
// card it was tracking (with a wide margin, so a small mapping error between
// preview and photo cannot cut the card off), sends where that crop sits in the
// photo so the server keeps the lens geometry, and gives up with a clear
// message instead of hanging.
let STILL_TIMEOUT_MS = 60000;           // let: the browser tests shorten it
const STILL_CROP_MARGIN = 0.6;          // of the card's long side, each way
const STILL_FRAME_MARGIN = 0.25;        // the same, when the still is the video frame
const STILL_MAX_PIXELS = 36000000;      // server refuses above 40 MP
const STILL_MAX_BYTES = 9 * 1024 * 1024; // server refuses above 12 MB

// The tracked outline in video pixels, if it is fresh enough to trust.
function freshTrackQuad() {
  if (currentMode !== 'ar' || !targetQuad || !lastHUDData || !lastHUDData.tracking) return null;
  if (!arStats.lastOkAt || Date.now() - arStats.lastOkAt > overlayStaleMs() + 700) return null;
  return { quad: targetQuad.map(p => [p[0], p[1]]), vw: arVideo.videoWidth, vh: arVideo.videoHeight };
}

function encodeCanvas(c, q) {
  return new Promise(res => c.toBlob(res, 'image/jpeg', q));
}

async function prepareStill(blob, track) {
  let bmp = null;
  try { bmp = await createImageBitmap(blob); } catch (e) { bmp = null; }
  if (!bmp) return { blob: blob, parent: null, note: 'sent as taken' };
  const pw = bmp.width, ph = bmp.height;
  let box = null;
  // The preview is (at most) a centred crop of the photo's field of view, so
  // the photo covers it at the smaller of the two scales. Skip the crop when
  // the orientations disagree: the mapping would be a guess.
  if (track && track.vw && track.vh && ((track.vw > track.vh) === (pw > ph))) {
    const s = Math.min(pw / track.vw, ph / track.vh);
    const xs = track.quad.map(p => (p[0] - track.vw / 2) * s + pw / 2);
    const ys = track.quad.map(p => (p[1] - track.vh / 2) * s + ph / 2);
    const side = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys));
    // the video frame is the frame the outline came from: no mapping to allow for
    const m = (lastStillSource === 'frame' ? STILL_FRAME_MARGIN : STILL_CROP_MARGIN) * side;
    const x0 = Math.max(0, Math.floor(Math.min(...xs) - m));
    const y0 = Math.max(0, Math.floor(Math.min(...ys) - m));
    const x1 = Math.min(pw, Math.ceil(Math.max(...xs) + m));
    const y1 = Math.min(ph, Math.ceil(Math.max(...ys) + m));
    if (x1 - x0 >= 64 && y1 - y0 >= 64 && (x1 - x0) * (y1 - y0) < 0.6 * pw * ph) {
      box = { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
    }
  }
  if (box) {
    const c = document.createElement('canvas');
    c.width = box.w; c.height = box.h;
    c.getContext('2d').drawImage(bmp, box.x, box.y, box.w, box.h, 0, 0, box.w, box.h);
    if (bmp.close) bmp.close();
    const out = await encodeCanvas(c, 0.92);
    if (out) return { blob: out, parent: { x: box.x, y: box.y, w: pw, h: ph },
                      note: 'card crop ' + box.w + 'x' + box.h + ' of ' + pw + 'x' + ph };
    return { blob: blob, parent: null, note: 'crop failed; sent as taken' };
  }
  if (pw * ph <= STILL_MAX_PIXELS && blob.size <= STILL_MAX_BYTES) {
    if (bmp.close) bmp.close();
    return { blob: blob, parent: null, note: 'full photo ' + pw + 'x' + ph };
  }
  // Too big for the server as it stands: shrink the whole photo (the server
  // still crops to the card before its own cap).
  const k = Math.min(1, Math.sqrt(STILL_MAX_PIXELS / (pw * ph)));
  const c = document.createElement('canvas');
  c.width = Math.round(pw * k); c.height = Math.round(ph * k);
  c.getContext('2d').drawImage(bmp, 0, 0, c.width, c.height);
  if (bmp.close) bmp.close();
  let out = await encodeCanvas(c, 0.9);
  if (out && out.size > STILL_MAX_BYTES) out = await encodeCanvas(c, 0.8);
  return { blob: out || blob, parent: null, note: 'resized ' + c.width + 'x' + c.height };
}

// Keep the screen on while the camera is up: a phone that dims mid-shop
// drops the stream and the tracking with it.
async function holdScreenOn() {
  try {
    if ('wakeLock' in navigator && document.visibilityState === 'visible' && !wakeLock) {
      wakeLock = await navigator.wakeLock.request('screen');
      wakeLock.addEventListener('release', () => { wakeLock = null; });
    }
  } catch (e) { wakeLock = null; }
}
function releaseScreen() {
  try { if (wakeLock) wakeLock.release(); } catch (e) {}
  wakeLock = null;
}
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible' && videoStream) holdScreenOn();
});

async function triggerARFreeze() {
  if (!arVideo.videoWidth || stillBusy) return;
  // Take the outline before anything else: the loop pauses for the upload and
  // the overlay goes stale while the photo is being taken.
  const track = freshTrackQuad();
  stillBusy = true;
  try {
    const blob = await grabStill(track);
    if (blob) await sendStillPhoto(blob, track);
  } finally {
    stillBusy = false;
  }
}

// Action button
$('#btn-action').onclick = () => {
  synth.init();
  haptic('tap');
  if (currentMode === 'still') {
    $('#file').click();
  } else {
    triggerARFreeze();
  }
};

$('#file').onchange = () => { if ($('#file').files[0]) sendStillPhoto($('#file').files[0], null); };

// fetch with a deadline; a hung upload on shop signal says so instead of
// spinning forever.
function apiTimed(path, opts, ms) {
  const abort = new AbortController();
  const timer = setTimeout(() => abort.abort(), ms);
  return api(path, Object.assign({}, opts, { signal: abort.signal }))
    .then(r => r.text().then(t => {
      const d = parseReply(t);
      return d === undefined ? { ok: false, error: 'HTTP ' + r.status + ': ' + t.slice(0, 80) } : d;
    }))
    .catch(e => ({ ok: false, error: e && e.name === 'AbortError'
      ? 'no reply in ' + Math.round(ms / 1000) + 's -- the signal here may be too weak; try again, or step nearer the shop Wi-Fi'
      : String((e && e.message) || e) }))
    .finally(() => clearTimeout(timer));
}

async function sendStillPhoto(fileOrBlob, track) {
  $('#btn-action').disabled = true;
  $('#btn-action').textContent = 'Calculating Metrology…';
  $('#out').innerHTML = '<div class="strip"><div class="seg segL" style="width:50%"></div><div class="seg segR" style="width:50%"></div></div><div class="sub-meta" style="padding:14px 16px;color:var(--key)">🔬 Ray-tracing Snell refraction & calculating confidence intervals…</div>';

  try {
    const prep = await prepareStill(fileOrBlob, track);
    arStats.still = (arStats.still ? arStats.still + ' -> ' : '') + prep.note + ' ' + Math.round(prep.blob.size / 1024) + 'k';
    const fd = new FormData();
    fd.append('holder', $('#holder').value);
    fd.append('lens', $('#lens').value);
    fd.append('image', prep.blob, 'shot.jpg');
    if (prep.parent) {
      fd.append('crop_x', String(prep.parent.x));
      fd.append('crop_y', String(prep.parent.y));
      fd.append('full_w', String(prep.parent.w));
      fd.append('full_h', String(prep.parent.h));
    }

    // Identify runs on the same still, in parallel. It has its own OCR
    // legibility gate, so a frame /measure refuses can still be named.
    const idFd = new FormData();
    idFd.append('image', prep.blob, 'shot.jpg');
    const idReq = apiTimed('/identify', { method: 'POST', body: idFd }, STILL_TIMEOUT_MS);

    const d = await apiTimed('/measure', { method: 'POST', body: fd }, STILL_TIMEOUT_MS);
    renderResults(d);
    renderIdentify(await secondVote(await idReq));
  } catch (e) {
    $('#out').innerHTML = `<div class="empty-card" style="border-color:var(--stop)"><b style="color:var(--stop)">Connection Error</b><div>${esc(String(e))}</div></div>`;
  } finally {
    $('#btn-action').disabled = false;
    $('#btn-action').textContent = currentMode === 'ar' ? '⚡ Freeze AR Frame' : '📸 Measure Photo';
    $('#file').value = '';
  }
}

// A live number the photo could not confirm is not a result: say so on the
// chip rather than leave "settled ~PSA 8" over a refusal (2.19.1 field report:
// the chip settled on 66.8% while every photo of the card was refused).
let stillRefusedAt = 0;
function renderResults(d) {
  stillRefusedAt = d.ok ? 0 : Date.now();
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

function renderIdentify(d) {
  const box = document.createElement('div');
  box.className = 'id-result';
  let html = '<div class="sect-title">Card Identity (OCR)</div>';
  if (!d || !d.ok) {
    html += `<div class="sub-meta" style="color:var(--stop)">Identify failed: ${esc(d && d.error || 'no response')}</div>`;
  } else if (d.identified) {
    html += `<table class="mm-table">
      <tr><td>Name</td><td><b>${esc(d.name)}</b>${d.dex != null ? ' &middot; #' + esc(d.dex) : ''}</td></tr>
      <tr><td>OCR token</td><td>${esc(d.matched_token || '')} (${esc(d.edits)} edits)</td></tr>
      <tr><td>Corroborated</td><td>${d.corroborated ? 'yes' : 'no'}</td></tr>
      <tr><td>Engine</td><td>${esc(d.engine)}</td></tr></table>`;
  } else {
    const alts = (d.alternatives || []).slice(0, 5).map(esc).join(', ');
    html += `<div class="sub-meta">Not identified (engine: ${esc(d.engine)}, ${esc(d.tokens_considered)} tokens)${alts ? ' &middot; candidates: ' + alts : ''}</div>`;
  }
  if (d && d.warnings && d.warnings.length) {
    html += d.warnings.map(w => `<div class="sub-meta" style="color:var(--warn, #e0b341)">&#9888; ${esc(w)}</div>`).join('');
  }
  if (d && d.ok && d.decision) {
    const v = d.vote || {};
    const voteTxt = v.available
      ? `embedding: ${esc(v.name || 'no match')} (similarity ${Number(v.similarity || 0).toFixed(2)}, ${esc(v.index_size)} cards on this device)`
      : `embedding: ${esc(v.reason || 'unavailable')}`;
    html += `<div class="sub-meta"><b>${esc(d.decision)}</b> &middot; ${esc(d.decision_reason || '')}<br>${voteTxt}</div>`;
  }
  box.innerHTML = html;
  if (d && d.ok && d.identification_id) box.appendChild(feedbackForm(d));
  $('#out').appendChild(box);
}

// ---- In-situ feedback: confirm / correct, and the price paid or asked ----
const PRICE_SOURCES = ['manual', 'sticker', 'tag', 'shelf_sign', 'page_label', 'receipt', 'listing', 'verbal'];
let fbSeq = 0;
function feedbackForm(d) {
  const n = ++fbSeq;
  const row = document.createElement('div');
  row.className = 'fb-row';
  row.innerHTML = `
    <input id="fb-name-${n}" placeholder="Right name">
    <input id="fb-num-${n}" placeholder="#" inputmode="numeric">
    <input id="fb-price-${n}" placeholder="Price: $2.50, 3 for $1, 50c">
    <select id="fb-src-${n}">${PRICE_SOURCES.map(s => `<option value="${s}">${s.replace('_', ' ')}</option>`).join('')}</select>
    <button type="button" id="fb-ok-${n}">&#10003; Correct</button>
    <button type="button" id="fb-fix-${n}">Fix name</button>
    <button type="button" id="fb-no-${n}">Wrong</button>
    <div class="sub-meta" id="fb-msg-${n}" style="flex-basis:100%;padding:0"></div>`;
  const go = (action) => submitFeedback(d, action, n);
  setTimeout(() => {
    const ok = document.getElementById('fb-ok-' + n);
    if (ok) {
      ok.disabled = !d.name;
      ok.onclick = () => go('confirm');
      document.getElementById('fb-fix-' + n).onclick = () => go('correct');
      document.getElementById('fb-no-' + n).onclick = () => go('reject');
    }
  }, 0);
  return row;
}

function submitFeedback(d, action, n) {
  const msg = document.getElementById('fb-msg-' + n);
  const name = document.getElementById('fb-name-' + n).value.trim();
  const numTxt = document.getElementById('fb-num-' + n).value.trim();
  const priceTxt = document.getElementById('fb-price-' + n).value.trim();
  const body = { identification_id: d.identification_id, action: action };
  if (action === 'correct') body.name = name;
  if (numTxt) body.number = parseInt(numTxt, 10);
  if (priceTxt) body.price = { text: priceTxt, source: document.getElementById('fb-src-' + n).value };
  msg.textContent = 'saving…';
  api('/feedback', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
    .then(r => r.json())
    .then(async r => {
      if (!r.ok) { msg.textContent = 'not saved: ' + (r.error || 'error'); return; }
      let learned = r.learned && r.learned.device_prior ? 'this device now favours ' + r.learned.device_prior : 'recorded';
      if (action !== 'reject' && d.vote && d.vote.available && r.name) {
        const l = await swAsk({ type: 'learn', id: d.identification_id, name: r.name, number: r.number });
        if (l && l.ok) learned += '; on-device index ' + l.index_size + ' cards';
      }
      if (r.price) learned += '; price ' + r.price.amount + ' ' + r.price.currency;
      if (r.inbox) learned += '; photo saved to your private inbox';
      msg.textContent = learned;
    })
    .catch(e => { msg.textContent = 'not saved: ' + e; });
}

// ---- Second vote: ONNX embedding in the service worker, on-device index ----
let swReg = null;
let embedStatus = { available: false, reason: 'service worker not registered' };
function swAsk(msg) {
  return new Promise((resolve) => {
    if (!swReg || !swReg.active) { resolve({ available: false, ok: false, reason: 'service worker not active' }); return; }
    const ch = new MessageChannel();
    const timer = setTimeout(() => resolve({ available: false, ok: false, reason: 'service worker timed out' }), 30000);
    ch.port1.onmessage = (e) => { clearTimeout(timer); resolve(e.data); };
    swReg.active.postMessage(msg, [ch.port2]);
  });
}
async function setupEmbedWorker() {
  if (!('serviceWorker' in navigator) || isCrossOrigin()) {
    embedStatus = { available: false, reason: 'service worker unavailable in this shell' };
    return;
  }
  try {
    await navigator.serviceWorker.register('/sw.js', { type: 'module', scope: '/' });
    swReg = await navigator.serviceWorker.ready;
    embedStatus = await swAsk({ type: 'status' });
  } catch (e) {
    embedStatus = { available: false, reason: 'service worker failed: ' + e };
  }
}
async function secondVote(d) {
  if (!d || !d.ok || !d.identification_id) return d;
  if (!embedStatus.available || !d.embed_crop) {
    d.vote = { available: false, reason: embedStatus.reason || 'no crop for the embedding model' };
    return d;
  }
  try {
    const bin = atob(d.embed_crop);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    const bmp = await createImageBitmap(new Blob([bytes], { type: 'image/png' }));
    const c = document.createElement('canvas');
    c.width = bmp.width; c.height = bmp.height;
    const ctx = c.getContext('2d');
    ctx.drawImage(bmp, 0, 0);
    const image = ctx.getImageData(0, 0, c.width, c.height);
    const vote = await swAsk({ type: 'vote', id: d.identification_id, image: image });
    d.vote = vote;
    const r = await api('/identify/vote', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ identification_id: d.identification_id, vote: vote }),
    }).then(x => x.json());
    if (r.ok) { d.decision = r.decision; d.decision_reason = r.decision_reason; }
  } catch (e) {
    d.vote = { available: false, reason: String(e) };
  }
  return d;
}
setupEmbedWorker();

function esc(s) { return String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }

// Settings & Audio buttons
$('#btn-sound-test').onclick = () => { synth.playSettle(); haptic('settle'); };
$('#btn-ar-reset').onclick = () => {
  api('/ar/reset', { method: 'POST' });
  haptic('tap');
};
$('#audio-toggle').onchange = (e) => { synth.enabled = (e.target.value === 'on'); };

$('#btn-host-settings').onclick = () => {
  $('#setting-api-url').value = backendUrl;
  $('#setting-device-id').value = deviceId;
  try { $('#setting-owner-token').value = localStorage.getItem('bakugo_owner_token') || ''; } catch (e) {}
  $('#modal-settings').classList.remove('hidden');
};
$('#modal-close').onclick = () => { $('#modal-settings').classList.add('hidden'); };
$('#setting-save').onclick = () => {
  backendUrl = $('#setting-api-url').value.trim().replace(/\\/+$/, '');
  localStorage.setItem('bakugo_backend_url', backendUrl);
  try {
    const owner = $('#setting-owner-token').value.trim();
    if (owner) localStorage.setItem('bakugo_owner_token', owner);
    else localStorage.removeItem('bakugo_owner_token');
  } catch (e) {}
  $('#modal-settings').classList.add('hidden');
  refreshConfig();
};

$('#zoom').oninput = (e) => { userZoomAt = Date.now(); setZoom(e.target.value); };
$('#zoom-in').onclick = () => { userZoomAt = Date.now(); $('#zoom').value = Math.min(8, zoomLevel + 0.5); setZoom($('#zoom').value); };
$('#zoom-out').onclick = () => { userZoomAt = Date.now(); $('#zoom').value = Math.max(1, zoomLevel - 0.5); setZoom($('#zoom').value); };
$('#setting-camera').onchange = (e) => {
  cameraId = e.target.value || null;
  try {
    if (cameraId) localStorage.setItem('bakugo_camera_id', cameraId);
    else localStorage.removeItem('bakugo_camera_id');
  } catch (err) {}
  stopARStream();
  startARStream();
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
    sys_version = ""
    # StreamRequestHandler applies this to the socket: reads and writes that
    # stall longer than this raise instead of pinning the thread.
    timeout = SOCKET_TIMEOUT_S
    _minted_token: Optional[str] = None

    def log_message(self, fmt, *args):
        pass

    # -- origin / CORS ------------------------------------------------------
    def _allowed_origin(self) -> Optional[str]:
        origin = (self.headers.get("Origin") or "").strip().rstrip("/")
        return origin if origin and origin in cors_origins() else None

    def _is_cross_origin(self) -> bool:
        from urllib.parse import urlparse

        origin = self.headers.get("Origin")
        if not origin:
            return False
        return urlparse(origin).netloc.lower() != (self.headers.get("Host") or "").lower()

    def _cors_headers(self) -> None:
        origin = self._allowed_origin()
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Allow-Credentials", "true")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, HEAD")
            # X-Device-ID / X-Client-ID stay allowed so older clients pass the
            # preflight; the server ignores them (see _extract_device_id).
            self.send_header(
                "Access-Control-Allow-Headers",
                "Content-Type, Authorization, Range, X-Device-ID, X-Client-ID, X-Bakugo-Owner",
            )
            self.send_header(
                "Access-Control-Expose-Headers", "Content-Length, Content-Type, X-Device-Token"
            )
        self.send_header("Vary", "Origin")

    def _cookie_secure(self) -> bool:
        mode = os.environ.get("CARDCENTER_COOKIE_SECURE", "auto").strip().lower()
        if mode in ("1", "true", "yes", "on"):
            return True
        if mode in ("0", "false", "no", "off"):
            return False
        proto = (self.headers.get("X-Forwarded-Proto") or "").strip().lower()
        return proto == "https" or '"https"' in (self.headers.get("Cf-Visitor") or "")

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        self._cors_headers()
        if self._minted_token:
            cookie = (
                f"{DEVICE_COOKIE}={self._minted_token}; Path=/; Max-Age=63072000; "
                "HttpOnly; SameSite=Lax"
            )
            if self._cookie_secure():
                cookie += "; Secure"
            self.send_header("Set-Cookie", cookie)
            # Cross-origin clients (mobile shell) never see the cookie, so they
            # get the token once in a header and send it back as a Bearer token.
            if self._is_cross_origin() and self._allowed_origin():
                self.send_header("X-Device-Token", self._minted_token)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_static(self, body: bytes, ctype: str, cache: str, extra: Optional[dict] = None) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(self, code: int, obj: dict) -> None:
        self._send(code, _dumps(obj).encode(), "application/json")

    def _reject(self, code: int, message: str) -> None:
        self.close_connection = True
        self._send_json(code, {"ok": False, "error": message})

    def _internal_error(self, path: str) -> dict:
        ref = secrets.token_hex(4)
        sys.stderr.write(f"[cardcenter] internal error ref={ref} path={path}\n")
        traceback.print_exc()
        return {"ok": False, "error": "internal error", "ref": ref}

    def do_OPTIONS(self) -> None:
        """Handle CORS preflight requests for allowed cross-origin clients."""
        self.send_response(204)
        self._cors_headers()
        if self._allowed_origin():
            self.send_header("Access-Control-Max-Age", "86400")
        self.send_header("Content-Length", "0")
        self.end_headers()

    # -- identity -----------------------------------------------------------
    def _request_token(self) -> Optional[str]:
        auth = (self.headers.get("Authorization") or "").strip()
        if auth[:7].lower() == "bearer ":
            tok = auth[7:].strip()
            if _valid_token(tok):
                return tok
        raw = self.headers.get("Cookie")
        if raw:
            try:
                jar = SimpleCookie()
                jar.load(raw)
                morsel = jar.get(DEVICE_COOKIE)
                if morsel is not None and _valid_token(morsel.value):
                    return morsel.value
            except CookieError:
                pass
        return None

    def _extract_device_id(self, fields: Optional[dict] = None) -> str:
        """Resolve the caller's tenant id from its server-issued device token.

        Client-supplied ids (X-Device-ID, ?device_id=, a device_id form field)
        are ignored: any caller can send any value there, so trusting them let
        one device read another's scans. Callers without a valid token get a
        new one (Set-Cookie, plus X-Device-Token for allowed cross-origin
        clients). CARDCENTER_TRUST_DEVICE_HEADER=1 restores the old behaviour
        for a private LAN-only server.
        """
        if _flag("CARDCENTER_TRUST_DEVICE_HEADER"):
            legacy = self._legacy_device_id(fields)
            if legacy:
                return legacy
        tok = self._request_token()
        if tok is None:
            if self._minted_token is None:
                self._minted_token = new_device_token()
            tok = self._minted_token
        return device_id_for_token(tok)

    def _legacy_device_id(self, fields: Optional[dict] = None) -> Optional[str]:
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
        return None

    def do_HEAD(self) -> None:
        self.do_GET()

    def do_GET(self) -> None:
        from urllib.parse import urlparse

        self._minted_token = None
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif path == "/manifest.json":
            self._send(200, MANIFEST_JSON.encode(), "application/json")
        elif path == "/sw.js":
            from .embed import sw_source

            self._send_static(sw_source().encode(), "text/javascript; charset=utf-8",
                              cache="no-cache", extra={"Service-Worker-Allowed": "/"})
        elif path == "/embed/status":
            from .embed import status as embed_status

            self._send(200, _dumps(embed_status()).encode(), "application/json")
        elif path.startswith("/embed/"):
            from .embed import served_file

            found = served_file(path[len("/embed/"):])
            if found is None:
                self._send(404, b"not found", "text/plain")
            else:
                fpath, ctype = found
                self._send_static(fpath.read_bytes(), ctype, cache="public, max-age=86400")
        elif path == "/health":
            db = os.environ.get("CARDCENTER_DB", "cardcenter.db")
            self._send(
                200,
                _dumps(
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
                _dumps(
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
                _dumps({"holders": holders, "version": __version__}).encode(),
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
                _dumps(
                    {
                        "ok": True,
                        "version": __version__,
                        "graders": available_graders(),
                        "holders": holders,
                        "ip": local_ip(),
                        "device_id": self._extract_device_id(),
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
                _dumps({"ok": True, "device_id": device_id, "scans": scans}).encode(),
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
            except Exception:
                res = dict(self._internal_error(path), device_id=device_id)
            self._send(200, _dumps(res).encode(), "application/json")
        elif path == "/quipu":
            # Mesh guidance, calibration and lexicon are model internals; they
            # are served only when explicitly enabled.
            if not _flag("CARDCENTER_EXPOSE_QUIPU"):
                self._send(404, b"not found", "text/plain")
                return
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
            except Exception:  # pragma: no cover - observer is optional
                payload = dict(self._internal_error(path), enabled=False)
            self._send(200, _dumps(payload).encode(), "application/json")
        elif path == "/marketplace/assets" and _flag("CARDCENTER_ENABLE_MARKETPLACE"):
            try:
                from .marketplace_client import get_marketplace_assets
                assets = get_marketplace_assets(axis="touch")
                self._send(200, _dumps({"ok": True, "assets": assets}).encode(), "application/json")
            except Exception:
                self._send_json(200, self._internal_error(path))
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        from urllib.parse import urlparse

        self._minted_token = None
        parsed = urlparse(self.path)
        path = parsed.path

        routes = {"/measure", "/ar/push", "/identify", "/ar/session", "/ar/reset",
                  "/identify/vote", "/feedback"}
        if _flag("CARDCENTER_ENABLE_MARKETPLACE"):
            routes.add("/marketplace/tokenize")
        if path not in routes:
            self.close_connection = True
            self._send(404, b"not found", "text/plain")
            return

        # Size is checked from the header before any of the body is read.
        if self.headers.get("Transfer-Encoding"):
            self._reject(411, "send a Content-Length; chunked bodies are not accepted")
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._reject(400, "invalid Content-Length")
            return
        if length < 0:
            self._reject(400, "invalid Content-Length")
            return
        if length > MAX_BODY_BYTES:
            self._reject(413, f"request body is over the {MAX_BODY_BYTES // (1024 * 1024)} MB limit")
            return
        try:
            body = self.rfile.read(length) if length > 0 else b""
        except OSError:  # includes socket timeout
            self.close_connection = True
            return
        if len(body) != length:
            self.close_connection = True
            return

        field = None                       # (kind, image bytes, meta) to keep
        try:
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
                field = ("measure", image, {
                    "holder": fields.get("holder", b"raw").decode("utf-8", "replace"),
                    "lens": fields.get("lens", b"main").decode("utf-8", "replace"),
                    "parent": _parent_crop(fields, image),
                    "device": str(device_id)[:8]})
                payload = _measure_payload(
                    image,
                    fields.get("holder", b"raw").decode("utf-8", "replace"),
                    fields.get("lens", b"main").decode("utf-8", "replace"),
                    _parent_crop(fields, image),
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

                push_t0 = time.monotonic()
                frame = decode_image(image_bytes, "frame")

                session = _get_or_create_ar_session(device_id, holder=holder, lens=lens)
                # A tap on the preview picks the card: a shop counter has
                # several, and the one under the reticle is not always it.
                aim = (_field_float(fields, "aim_x"), _field_float(fields, "aim_y"))
                if None not in aim and all(0.0 <= v <= 1.0 for v in aim):
                    session.select(aim)
                # One push at a time per session. A push the phone gave up on
                # (4 s) keeps running here; the next one used to start beside
                # it, on the same session, and on two CPUs both then ran late
                # -- the "PUSH TIMED OUT" runs of the 2.20.0 field report.
                # Wait briefly for the previous frame, then say "busy" rather
                # than queue: the phone just sends a newer frame.
                lock = getattr(session, "_serve_lock", None)
                if lock is not None and not lock.acquire(timeout=AR_BUSY_WAIT_S):
                    raise _Busy()
                try:
                    measured_before = session.measured
                    status: ARStatus = session.push(
                        frame, source_scale=_field_float(fields, "source_scale"),
                        zoom=_field_float(fields, "zoom"))
                finally:
                    if lock is not None:
                        lock.release()
                # keep the frames that were measured, at most one per 2 s,
                # and one in 6 s of the rest: the 2.21.0 session measured
                # nothing live, so it left no frames to replay
                measured_now = status.measured_frames > measured_before
                since = time.time() - getattr(session, "_field_at", 0.0)
                if (measured_now and since >= 2.0) or since >= 6.0:
                    session._field_at = time.time()
                    res = session.last_result if measured_now else None
                    field = ("live" if measured_now else "live_unmeasured", image_bytes, {
                        "tracking": status.tracking,
                        "guidance": list(status.guidance),
                        "server_ms": int(1000 * (time.monotonic() - push_t0)),
                        "holder": holder, "lens": lens, "device": str(device_id)[:8],
                        "quad": status.quad.tolist() if status.quad is not None else None,
                        "measured_frames": status.measured_frames,
                        "ratio": status.ratio.value if status.ratio else None,
                        "settled": status.settled,
                        "last_borders_mm": None if res is None else {
                            "left": res.horizontal.low_mm.value, "right": res.horizontal.high_mm.value,
                            "top": res.vertical.low_mm.value, "bottom": res.vertical.high_mm.value},
                        "last_warnings": [] if res is None else list(res.quality.warnings)})

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
                    "grade_estimate": status.grade_estimate,
                    "grade_confidence": (
                        round(status.grade_confidence, 2)
                        if status.grade_confidence is not None
                        else None
                    ),
                    "bands": status.bands,
                    "verdict": (
                        session.verdict.name
                        if hasattr(session.verdict, "name")
                        else str(session.verdict)
                    ),
                    "scale": round(status.scale.value, 2) if status.scale else None,
                    # Information floor of the tracked outline and the
                    # auto-accept decision (confidence.gate) for the fused ratio.
                    "information": status.information,
                    "decision": status.decision,
                    "decision_reason": status.decision_reason,
                    # time spent here: the phone's round trip minus this is
                    # the network (the debug inset shows both)
                    "server_ms": int(1000 * (time.monotonic() - push_t0)),
                    "live_px_per_mm": (round(status.px_per_mm, 2)
                                       if status.px_per_mm is not None else None),
                    "card_frac": (round(status.card_frac, 3)
                                  if status.card_frac is not None else None),
                }

            elif path == "/identify":
                # Naming a card and measuring it have different error budgets.
                # Metrology needs the cut edge to a fraction of a millimetre;
                # OCR needs legible glyphs. This endpoint exists so a photo
                # that /measure rightly refuses -- a sleeved card over a bulk
                # bin -- is still worth something to the user.
                image_bytes = fields.get("image") or (body if not fields else None)
                if not image_bytes:
                    raise DetectionError("no image provided")
                from .insitu import owner_token_matches

                field = ("identify", image_bytes, {"device": str(device_id)[:8]})
                payload = _identify_payload(
                    image_bytes, device_id, owner_token_matches(self.headers.get("X-Bakugo-Owner"))
                )

            elif path == "/identify/vote":
                payload = _vote_payload(_json_body(body), device_id)

            elif path == "/feedback":
                from .insitu import owner_token_matches

                payload = _feedback_payload(
                    _json_body(body), device_id,
                    owner_token_matches(self.headers.get("X-Bakugo-Owner")),
                )

            elif path == "/ar/session" or path == "/ar/reset":
                holder = fields.get("holder", b"raw").decode("utf-8", "replace")
                lens = fields.get("lens", b"main").decode("utf-8", "replace")
                session = _get_or_create_ar_session(device_id, holder=holder, lens=lens)
                session.reset()
                payload = {"ok": True, "device_id": device_id, "status": "reset"}

            elif path == "/marketplace/tokenize":
                # Only routed when CARDCENTER_ENABLE_MARKETPLACE=1 (see above).
                # Keep that off on any publicly reachable server.
                try:
                    data = json.loads(body.decode("utf-8")) if body else {}
                except ValueError:
                    raise DetectionError("request body must be JSON")
                if not isinstance(data, dict):
                    raise DetectionError("request body must be a JSON object")
                wallet = str(data.get("wallet") or "").strip()
                if not wallet:
                    raise DetectionError("wallet is required")
                try:
                    from .marketplace_client import tokenize_metrology_scan
                    payload = tokenize_metrology_scan(
                        scan_id=str(data.get("scan_id", f"scan-{int(time.time())}")),
                        title=str(data.get("title", "Centering Metrology Scan")),
                        contributor_wallet=wallet,
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
                except Exception:
                    payload = self._internal_error(path)

            else:  # pragma: no cover - guarded by `routes` above
                self._send(404, b"not found", "text/plain")
                return

        except _Busy:
            payload = {"ok": False, "busy": True,
                       "error": "still working on the previous frame"}
        except DetectionError as exc:
            payload = {"ok": False, "error": str(exc)}
        except Exception:
            # Details go to the server log only; the client gets a reference.
            self._send_json(500, self._internal_error(path))
            return

        if field is not None:
            from .fieldlog import record

            kind, data, meta = field
            outcome = {k: payload.get(k) for k in (
                "ok", "error", "ratio", "ratio_lo", "ratio_hi", "axis", "borders", "px_per_mm",
                "inner_confidence", "warnings", "identified", "name") if k in payload}
            record(kind, data, {**meta, "version": __version__, "outcome": outcome})
        self._send(200, _dumps(payload).encode(), "application/json")


def local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


class BoundedThreadingHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer with a cap on concurrent connections.

    The stock server starts a thread per connection with no limit. Here a
    connection waits up to two seconds for a free slot and is then dropped.
    """

    daemon_threads = True
    request_queue_size = 64

    def __init__(self, *args, max_connections: int = MAX_CONNECTIONS, **kwargs):
        super().__init__(*args, **kwargs)
        self._slots = threading.BoundedSemaphore(max(1, max_connections))

    def process_request(self, request, client_address):
        if not self._slots.acquire(timeout=2.0):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self._slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()


def serve(host: str = "0.0.0.0", port: int = 8765) -> None:
    if not os.environ.get("CARDCENTER_DB"):
        os.environ["CARDCENTER_DB"] = "cardcenter.db"
    httpd = BoundedThreadingHTTPServer((host, port), Handler)
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
