"""Static files for the service-worker embedding recognizer.

The model and the onnxruntime-web WASM build are not part of this package.
Put them in ``CARDCENTER_EMBED_DIR``::

    model.json                      manifest (see ``validate_manifest``)
    model.onnx                      the embedding model
    ort/ort.wasm.bundle.min.mjs     from onnxruntime-web/dist (>= 1.19)
    ort/ort-wasm-simd-threaded.wasm from the same release

The bundle build inlines its WASM glue, so the module service worker never
needs ``import()`` (which service workers do not allow). ``wasmPaths`` is
left unset on purpose: setting it makes onnxruntime-web load the glue with a
dynamic import. The .wasm file is found next to the bundle.

Only the files above are served. Nothing under this directory is an index of
anyone's cards: each device's index lives in that device's IndexedDB.
"""

from __future__ import annotations

import json
import os
from importlib import resources
from pathlib import Path
from typing import Optional

MANIFEST = "model.json"
MODEL = "model.onnx"
ORT_BUNDLE = "ort/ort.wasm.bundle.min.mjs"
ORT_WASM = "ort/ort-wasm-simd-threaded.wasm"
SERVED = {
    MANIFEST: "application/json",
    MODEL: "application/octet-stream",
    ORT_BUNDLE: "text/javascript; charset=utf-8",
    ORT_WASM: "application/wasm",
}
MAX_MODEL_BYTES = 256 * 1024 * 1024


class EmbedConfigError(ValueError):
    pass


def embed_dir() -> Optional[Path]:
    raw = os.environ.get("CARDCENTER_EMBED_DIR")
    return Path(raw) if raw else None


def validate_manifest(m: dict) -> dict:
    if not isinstance(m, dict):
        raise EmbedConfigError("model.json must be an object")
    mid = m.get("model_id")
    if not isinstance(mid, str) or not (1 <= len(mid) <= 80) or not all(
            c.isalnum() or c in "._-" for c in mid):
        raise EmbedConfigError("model_id must be 1-80 chars of [A-Za-z0-9._-]")
    size = m.get("input_size")
    if (not isinstance(size, list) or len(size) != 2
            or not all(isinstance(v, int) and 8 <= v <= 1024 for v in size)):
        raise EmbedConfigError("input_size must be [height, width], 8..1024")
    if m.get("layout", "NCHW") not in ("NCHW", "NHWC"):
        raise EmbedConfigError("layout must be NCHW or NHWC")
    if m.get("channels", "RGB") not in ("RGB", "BGR"):
        raise EmbedConfigError("channels must be RGB or BGR")
    for key in ("mean", "std"):
        v = m.get(key, [0, 0, 0] if key == "mean" else [1, 1, 1])
        if (not isinstance(v, list) or len(v) != 3
                or not all(isinstance(x, (int, float)) for x in v)
                or (key == "std" and any(x == 0 for x in v))):
            raise EmbedConfigError(f"{key} must be three numbers")
    if not isinstance(m.get("scale", 255), (int, float)) or m.get("scale", 255) <= 0:
        raise EmbedConfigError("scale must be positive")
    for key in ("input_name", "output_name", "license", "source"):
        if key in m and not isinstance(m[key], str):
            raise EmbedConfigError(f"{key} must be a string")
    return m


def status() -> dict:
    d = embed_dir()
    if d is None:
        return {"available": False, "reason": "CARDCENTER_EMBED_DIR is not set"}
    missing = [f for f in SERVED if not (d / f).is_file()]
    if missing:
        return {"available": False, "reason": f"missing in embed dir: {', '.join(missing)}"}
    try:
        m = validate_manifest(json.loads((d / MANIFEST).read_text(encoding="utf-8")))
    except (ValueError, OSError) as exc:
        return {"available": False, "reason": f"model.json invalid: {exc}"}
    if (d / MODEL).stat().st_size > MAX_MODEL_BYTES:
        return {"available": False, "reason": "model.onnx is larger than 256 MB"}
    return {"available": True, "model_id": m["model_id"], "input_size": m["input_size"]}


def served_file(rel: str) -> Optional[tuple[Path, str]]:
    """(path, content type) for an allowlisted file, else None."""
    ctype = SERVED.get(rel)
    d = embed_dir()
    if ctype is None or d is None:
        return None
    p = (d / rel).resolve()
    try:
        p.relative_to(d.resolve())
    except ValueError:
        return None
    return (p, ctype) if p.is_file() else None


def sw_source() -> str:
    tpl = resources.files("cardcenter").joinpath("static/sw.js").read_text(encoding="utf-8")
    st = status()
    public = {k: st[k] for k in ("available", "reason", "model_id", "input_size") if k in st}
    imp = ("import * as ort from '/embed/ort/ort.wasm.bundle.min.mjs';"
           if st["available"] else "const ort = null;")
    return tpl.replace("/*__ORT_IMPORT__*/", imp).replace("/*__STATUS__*/", json.dumps(public))
