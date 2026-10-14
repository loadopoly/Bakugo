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
* Defects are what the condition readings resolve: corner whitening and loss, edge whitening and nicks. Silvering, holo
  scratches and smudges are defined for human labels; a single view cannot tell a scratch from artwork or a smudge, so
  the pipeline does not emit them on its own.
"""

from __future__ import annotations

import hashlib
import io
import json
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
