"""The standard metrology record: one measurement of one card face as JSON (schema ``bakugo.metrology_record/1``).

The schema is ``cardcenter/schemas/metrology_record.v1.schema.json``. Everything here is built from the pipeline's own
objects (CenteringResult, the condition view), so every measured length keeps its 1-sigma uncertainty instead of the
rounded numbers /measure shows a person.

What v1 can and cannot say, plainly:

* Lens distortion is reported as ``not_corrected``: the camera model is a pinhole (focal length from the lens's field of
  view). A lens profile database and the correction that uses it are the next step; the fields for the Brown-Conrady
  coefficients are already in the schema.
* Holder optics come from the presets in ``types.SLAB_PRESETS`` (standard refractive indices with their uncertainty,
  thicknesses that are estimates) and are marked ``source: preset`` until someone measures their own holder.
* The four cut edges are probed for straightness (sagitta, cardcenter/edge_probe.py) and for the split between the
  red, green and blue channels. Bowed edges are lens distortion or a curved holder face; which one shows in the pattern
  (``edge_probe.bow``). A colour split is the lens's lateral chromatic aberration; the plastic's own dispersion is a
  few micrometres.
* Holder error is measured, not assumed: the same card (``subject.card_ref``) measured bare and in a holder by one
  device gives that holder's bias and excess scatter per border (``holder_study``,
  ``python -m cardcenter.metrology_record holder-study``).
* Defects are what the condition readings resolve: corner whitening and loss, edge whitening and nicks. Silvering, holo
  scratches and smudges are defined for human labels; a single view cannot tell a scratch from artwork or a smudge, so
  the pipeline does not emit them on its own.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from . import __version__
from .types import STANDARD_CARD_H_MM, STANDARD_CARD_W_MM

SCHEMA_ID = "bakugo.metrology_record/1"
SCHEMA_PATH = Path(__file__).with_name("schemas") / "metrology_record.v1.schema.json"
CAVEAT = "Estimates from a photo, not a grade issued by any grading company."

_CORNER_REGION = {"top_left": "corner_tl", "top_right": "corner_tr", "bottom_right": "corner_br", "bottom_left": "corner_bl"}
_EDGE_LEN_MM = {"top": STANDARD_CARD_W_MM, "bottom": STANDARD_CARD_W_MM, "left": STANDARD_CARD_H_MM, "right": STANDARD_CARD_H_MM}


def schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _m(measured: Any) -> dict:
    return {"value": round(float(measured.value), 4), "sigma": round(float(measured.sigma), 4)}


def _mv(value: float, sigma: float = 0.0) -> dict:
    return {"value": round(float(value), 4), "sigma": round(max(0.0, float(sigma)), 4)}


def exif_device(image_bytes: bytes) -> Optional[dict]:
    """Camera make and model from the photo's EXIF, when the photo has any (a camera photo does; a video frame drawn
    to a canvas does not). Pillow is optional at run time."""
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        exif = Image.open(io.BytesIO(image_bytes)).getexif()
    except Exception:
        return None
    make, model = (str(exif.get(271) or "").strip() or None), (str(exif.get(272) or "").strip() or None)
    if not (make or model):
        return None
    return {"make": make, "model": model, "source": "exif"}


def _holder(slab: Any, quality: Any) -> dict:
    layers = [{"name": layer.name,
               "refractive_index": _mv(layer.refractive_index, layer.refractive_index_sigma),
               "thickness_mm": _mv(layer.acrylic_thickness_mm, layer.acrylic_thickness_sigma_mm),
               "source": "preset"}
              for layer in slab.layers if layer.is_optically_active]
    return {"preset": slab.name, "layers": layers, "refraction_applied": bool(quality.refraction_applied),
            "max_refraction_shift_mm": round(float(quality.max_refraction_shift_mm or 0.0), 4)}


def _defects(view: Any, version: str) -> list[dict]:
    """The defects the condition readings resolve, with their magnitudes. Nothing unresolved is reported."""
    if view is None:
        return []
    label = {"source": "model", "model": f"cardcenter.condition/{version}"}
    out: list[dict] = []
    for name, c in (getattr(view, "corners", None) or {}).items():
        region = _CORNER_REGION.get(name)
        if region is None or not c.resolvable:
            continue
        if c.whitening_resolvable and c.whitening_mm > 0:
            out.append({"defect_id": f"{region}:whitening", "type": "whitening", "region": region,
                        "magnitude": {"depth_mm": _mv(c.whitening_mm, c.sigma_mm), "length_mm": _mv(c.whitening_len_mm)},
                        "evidence": {"views": 1}, "label": dict(label)})
        if c.loss_mm > 0:
            out.append({"defect_id": f"{region}:corner_loss", "type": "corner_loss", "region": region,
                        "magnitude": {"depth_mm": _mv(c.loss_mm, c.sigma_mm)},
                        "evidence": {"views": 1}, "label": dict(label)})
    for side, e in (getattr(view, "edges", None) or {}).items():
        region = f"edge_{side}"
        if side not in _EDGE_LEN_MM or not e.resolvable:
            continue
        if e.whitening_resolvable and e.whitening_frac > 0:
            frac = min(1.0, max(0.0, float(e.whitening_frac)))
            out.append({"defect_id": f"{region}:whitening", "type": "whitening", "region": region,
                        "magnitude": {"fraction_of_edge": round(frac, 4), "length_mm": _mv(frac * _EDGE_LEN_MM[side])},
                        "evidence": {"views": 1}, "label": dict(label)})
        if e.nick_count > 0:
            out.append({"defect_id": f"{region}:nick", "type": "nick", "region": region,
                        "magnitude": {"depth_mm": _mv(e.nick_max_mm, e.noise_mm)},
                        "evidence": {"views": e.nick_count}, "label": dict(label)})
    return out


BOW_MIN_MM = 0.05     # a bow smaller than this is below what matters at grading scale
BOW_MIN_Z = 3.0       # and it must be this many sigma from straight


def _edge_probe(probe: Optional[Mapping[str, Any]]) -> Optional[dict]:
    """The probe's per-side readings, plus which edges bow and the pattern they make:
    ``none``; ``all_outward`` / ``all_inward`` (every edge the same way: radial lens distortion);
    ``top_bottom`` / ``left_right`` (one opposing pair only: a face curved across that direction, e.g. a bowed
    toploader); ``mixed`` (anything else, or too few edges located to say)."""
    if not probe:
        return None
    sides: dict = {}
    bowed: dict = {}
    for side, v in probe.items():
        if not v or not v.get("points"):
            sides[side] = {"points": int((v or {}).get("points") or 0)}
            continue
        sag, sig = float(v["sagitta_mm"]), float(v.get("sagitta_sigma_mm") or 0.0)
        sides[side] = {"points": int(v["points"]), "sagitta_mm": _mv(sag, sig), "line_rms_mm": round(float(v["line_rms_mm"]), 4),
                       "red_minus_green_px": v.get("red_minus_green_px"), "blue_minus_green_px": v.get("blue_minus_green_px")}
        if abs(sag) >= BOW_MIN_MM and (sig <= 0 or abs(sag) / sig >= BOW_MIN_Z):
            bowed[side] = 1 if sag > 0 else -1
    located = [s for s, v in sides.items() if v.get("points")]
    if len(located) < 4:
        pattern = "mixed" if bowed else ("none" if len(located) >= 2 else "mixed")
    elif not bowed:
        pattern = "none"
    elif len(bowed) == 4 and len(set(bowed.values())) == 1:
        pattern = "all_outward" if next(iter(bowed.values())) > 0 else "all_inward"
    elif set(bowed) == {"top", "bottom"}:
        pattern = "top_bottom"
    elif set(bowed) == {"left", "right"}:
        pattern = "left_right"
    else:
        pattern = "mixed"
    return {"sides": sides, "bowed": sorted(bowed), "bow": pattern}


def _predicted(preds: Mapping[str, Any]) -> dict:
    out = {}
    for grader, p in (preds or {}).items():
        out[grader] = {"grade": p.grade_label, "score": p.grade_score if p.graded else None, "ceiling": p.grade_score,
                       "range": list(p.range) if p.range else None, "graded": bool(p.graded),
                       "probability": p.confidence, "missing": list(p.missing),
                       "subgrades": {"centering": p.centering_subgrade, "corners": p.estimated_corners,
                                     "edges": p.estimated_edges, "surface": p.estimated_surface}}
    return out


def build_record(*, image_bytes: bytes, image_shape: tuple, result: Any, bands: Mapping[str, Any], preds: Mapping[str, Any],
                 view: Any = None, face: str = "front", lens: str = "main", fov_deg: Optional[float] = None,
                 capture: Any = None, warnings: Optional[list] = None, image_source: str = "unknown",
                 device_id: Optional[str] = None, identification: Optional[Mapping[str, Any]] = None,
                 asset_id: Optional[str] = None, convergence: Optional[Mapping[str, Any]] = None,
                 edge_probe: Optional[Mapping[str, Any]] = None, card_ref: Optional[str] = None,
                 now: Optional[datetime] = None) -> dict:
    """The record for one measurement. ``result`` is the CenteringResult, ``view`` the ConditionView (or None),
    ``bands`` the centering grade bands and ``preds`` the predicted grades, as /measure computes them."""
    sha = hashlib.sha256(image_bytes).hexdigest()
    rid = "mr_" + hashlib.sha256(f"{sha}|{__version__}|{face}".encode()).hexdigest()[:24]
    now = now or datetime.now(timezone.utc)
    h, w = int(image_shape[0]), int(image_shape[1])
    worst = result.worst_axis
    wr = worst.ratio_pct
    lo, hi = wr.interval()
    focal = getattr(capture, "focal_px", None)
    pp = getattr(capture, "principal_point", None)
    card = None
    if identification and identification.get("name"):
        dex = identification.get("dex")
        card = {"name": str(identification["name"]), "dex": int(dex) if isinstance(dex, (int, float)) else None,
                "set_code": identification.get("set_code"), "number": identification.get("number"),
                "rarity": identification.get("rarity")}
    record = {
        "schema": SCHEMA_ID,
        "record_id": rid,
        "created_utc": now.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "software": {"name": "cardcenter", "version": __version__},
        "subject": {"face": face if face in ("front", "back") else "front", "asset_id": asset_id,
                    "card_ref": clean_card_ref(card_ref),
                    "identification_id": (identification or {}).get("identification_id"), "card": card},
        "capture": {
            "image": {"width_px": w, "height_px": h, "sha256": sha,
                      "source": image_source if image_source in ("video_frame", "camera_photo", "upload") else "unknown"},
            "device": exif_device(image_bytes),
            "lens": {"label": lens, "fov_deg": fov_deg, "focal_px": round(float(focal), 2) if focal else None,
                     "principal_point_px": [round(float(pp[0]), 2), round(float(pp[1]), 2)] if pp else None,
                     "distortion": {"status": "not_corrected"}},
            "px_per_mm": round(float(result.px_per_mm), 3),
            "tilt_deg": None if view is None or view.tilt_deg is None else round(float(view.tilt_deg), 2),
        },
        "holder": _holder(result.slab, result.quality),
        "centering": {
            "borders_mm": {"left": _m(result.horizontal.low_mm), "right": _m(result.horizontal.high_mm),
                           "top": _m(result.vertical.low_mm), "bottom": _m(result.vertical.high_mm)},
            "ratios": {"horizontal": _m(result.horizontal.ratio_pct), "vertical": _m(result.vertical.ratio_pct)},
            "worst": {"axis": worst.axis, "ratio": _m(wr), "interval95": [round(lo, 3), round(hi, 3)], "wider": worst.skew_toward},
            "inner_confidence": round(min(1.0, max(0.0, float(result.quality.inner_confidence))), 4),
        },
        "condition": None,
        "edge_probe": _edge_probe(edge_probe),
        "defects": _defects(view, __version__),
        "convergence": dict(convergence) if convergence else None,
        "grading": {
            "centering_bands": {g: {"label": b.best if b.is_single else f"{b.worst}–{b.best}", "limited_by": b.limited_by}
                                for g, b in (bands or {}).items()},
            "predicted": _predicted(preds),
            "caveat": CAVEAT,
        },
        "quality": {"warnings": [str(x) for x in (warnings if warnings is not None else result.quality.warnings)],
                    "outer_residual_px": round(float(result.quality.outer_residual_px), 3)},
        "provenance": {"device_id": device_id},
    }
    if view is not None:
        d = view.to_dict()
        record["condition"] = {"corners": d.get("corners") or {}, "edges": d.get("edges") or {}, "surface": d.get("surface") or {}}
    return record


# ---------------------------------------------------------------------------- the same card, bare and in holders

_CARD_REF = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
BORDERS = ("left", "right", "top", "bottom")


def clean_card_ref(ref: Any) -> Optional[str]:
    """The user's own label for one physical card (e.g. "probopass-1"), so measurements of it can be paired."""
    s = str(ref or "").strip()
    return s if _CARD_REF.match(s) else None


def _db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=10)
    conn.execute("CREATE TABLE IF NOT EXISTS metrology_records (record_id TEXT PRIMARY KEY, created_utc TEXT NOT NULL, "
                 "device_id TEXT, card_ref TEXT, holder TEXT, face TEXT, scan_id INTEGER, record_json TEXT NOT NULL)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_mr_card ON metrology_records (device_id, card_ref, face)")
    return conn


def save_record(db_path: str, record: Mapping[str, Any], scan_id: Optional[int] = None) -> None:
    """Keeps the record beside the scans. A re-measure of the same photo by the same release replaces its row."""
    if not db_path or not record:
        return
    prov = record.get("provenance") or {}
    conn = _db(db_path)
    try:
        conn.execute("INSERT OR REPLACE INTO metrology_records VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                     (record["record_id"], record["created_utc"], prov.get("device_id"), (record.get("subject") or {}).get("card_ref"),
                      (record.get("holder") or {}).get("preset"), (record.get("subject") or {}).get("face"), scan_id,
                      json.dumps(record, separators=(",", ":"))))
        conn.commit()
    finally:
        conn.close()


def load_records(db_path: str) -> list[dict]:
    if not db_path or not os.path.exists(db_path):
        return []
    conn = _db(db_path)
    try:
        return [json.loads(r[0]) for r in conn.execute(
            "SELECT record_json FROM metrology_records WHERE card_ref IS NOT NULL ORDER BY created_utc").fetchall()]
    finally:
        conn.close()


def _mean(xs: list) -> float:
    return sum(xs) / len(xs)


def holder_study(records: list[Mapping[str, Any]], min_cards: int = 1) -> dict:
    """Per holder, against the same cards measured bare (holder ``raw``) by the same device on the same face:

    * ``bias_mm``: mean (in holder - bare) per border; what the refraction model leaves uncorrected.
    * ``extra_sigma_mm``: scatter of those differences beyond what both measurements' own sigmas explain
      (sqrt(max(0, var(diff) - mean(sigma_bare^2 + sigma_holder^2)))); the honest extra uncertainty for that holder.
    * ``ratio_bias_pp``: mean change in the worst-axis centering ratio, in percentage points.

    A holder needs ``min_cards`` paired cards to be reported; with fewer than 3 differences per border the scatter is
    reported as None rather than guessed."""
    groups: dict = {}
    for r in records:
        subj, prov = r.get("subject") or {}, r.get("provenance") or {}
        ref = subj.get("card_ref")
        if not ref:
            continue
        key = (prov.get("device_id"), ref, subj.get("face"))
        groups.setdefault(key, {}).setdefault((r.get("holder") or {}).get("preset") or "raw", []).append(r)
    per: dict = {}
    for key, by_holder in groups.items():
        bare = by_holder.get("raw")
        if not bare:
            continue
        bare_b = {s: _mean([x["centering"]["borders_mm"][s]["value"] for x in bare]) for s in BORDERS}
        bare_s2 = {s: _mean([x["centering"]["borders_mm"][s]["sigma"] ** 2 for x in bare]) / len(bare) for s in BORDERS}
        bare_ratio = _mean([x["centering"]["worst"]["ratio"]["value"] for x in bare])
        for holder, recs in by_holder.items():
            if holder == "raw":
                continue
            acc = per.setdefault(holder, {"cards": set(), "n": 0, "diff": {s: [] for s in BORDERS},
                                          "s2": {s: [] for s in BORDERS}, "ratio": []})
            acc["cards"].add(key)
            for x in recs:
                acc["n"] += 1
                for s in BORDERS:
                    m = x["centering"]["borders_mm"][s]
                    acc["diff"][s].append(m["value"] - bare_b[s])
                    acc["s2"][s].append(m["sigma"] ** 2 + bare_s2[s])
                acc["ratio"].append(x["centering"]["worst"]["ratio"]["value"] - bare_ratio)
    out: dict = {}
    for holder, acc in sorted(per.items()):
        if len(acc["cards"]) < min_cards:
            continue
        bias, extra = {}, {}
        for s in BORDERS:
            d = acc["diff"][s]
            bias[s] = round(_mean(d), 4)
            if len(d) >= 3:
                mu = _mean(d)
                var = sum((v - mu) ** 2 for v in d) / (len(d) - 1)
                extra[s] = round(math.sqrt(max(0.0, var - _mean(acc["s2"][s]))), 4)
            else:
                extra[s] = None
        out[holder] = {"cards": len(acc["cards"]), "measurements": acc["n"], "bias_mm": bias, "extra_sigma_mm": extra,
                       "ratio_bias_pp": round(_mean(acc["ratio"]), 3)}
    return out


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m cardcenter.metrology_record", description="metrology records")
    sub = ap.add_subparsers(dest="cmd", required=True)
    hs = sub.add_parser("holder-study", help="per-holder bias and extra scatter from cards measured bare and in holders")
    hs.add_argument("--db", default=os.environ.get("CARDCENTER_DB", "cardcenter.db"))
    hs.add_argument("--min-cards", type=int, default=1)
    sub.add_parser("schema", help="print the JSON Schema")
    args = ap.parse_args(argv)
    if args.cmd == "schema":
        print(json.dumps(schema(), indent=2))
    else:
        recs = load_records(args.db)
        print(json.dumps({"records_with_card_ref": len(recs), "holders": holder_study(recs, args.min_cards)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
