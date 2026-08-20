"""Bakugo's link to the QUIPU Observer — feed structured observations up,
pull cross-corpus learnings down.

Bakugo's images are RELATIVELY STRUCTURED (trading cards: fixed geometry,
closed catalog vocabulary), so its observations route onto QUIPU's *touch*
axis, while Loadopoly-OCR's unstructured archival scans route onto *vision*.
QUIPU trains one shared mesh across both and enacts the result back here:
token frequencies learned from EITHER corpus become priors that break ties
between otherwise-ambiguous collector-number readings.

Everything is best-effort and non-blocking:
  * observations post from a daemon thread with a short timeout
  * guidance is cached with a TTL and returns {} when the Observer is away
  * disabled entirely unless CARDCENTER_QUIPU_URL is set

Stdlib only, matching the rest of the package.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Optional

from . import world_model_grounding

_SOURCE = "bakugo"
_TIMEOUT_S = 3.0
_GUIDANCE_TTL_S = 300.0

_guidance_lock = threading.Lock()
_guidance_cache: dict[str, Any] = {}
_guidance_fetched_at = 0.0


def base_url() -> str:
    return os.environ.get("CARDCENTER_QUIPU_URL", "").rstrip("/")


def enabled() -> bool:
    return bool(base_url())


def _post_json(path: str, payload: dict[str, Any]) -> Optional[dict[str, Any]]:
    url = f"{base_url()}{path}"
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return None


def _get_json(path: str) -> Optional[dict[str, Any]]:
    try:
        with urllib.request.urlopen(f"{base_url()}{path}", timeout=_TIMEOUT_S) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return None


def observe_async(
    text: str,
    *,
    confidence: Optional[float] = None,
    meta: Optional[dict[str, Any]] = None,
) -> None:
    """Fire-and-forget observation. Never blocks or raises."""
    if not enabled() or not text:
        return

    def _worker() -> None:
        _post_json(
            "/observe",
            {
                "source": _SOURCE,
                "kind": "structured",
                "text": text,
                "confidence": confidence,
                "meta": meta or {},
            },
        )

    threading.Thread(target=_worker, daemon=True, name="quipu-observe").start()


def observe_measure_async(payload: dict[str, Any]) -> None:
    """Distil a successful /measure payload into a concept-dense observation."""
    if not enabled() or not payload.get("ok"):
        return
    parts = [
        "card centering scan",
        f"holder {payload.get('holder', 'raw')}",
        f"axis {payload.get('axis', '')}",
        f"wider {payload.get('wider', '')}",
        f"ratio {payload.get('ratio', '')}",
    ]
    for grader, band in (payload.get("bands") or {}).items():
        parts.append(f"{grader} band {band.get('label', '')}")
    collector = payload.get("collector_number")
    if collector:
        parts.append(f"collector number {collector}")
        
    grounding = world_model_grounding.grounding_annotation(
        channel_conditions=payload.get("channel_conditions"),
        ratio=payload.get("ratio"),
        sigma=payload.get("sigma"),
        crb_sigma=payload.get("crb_sigma"),
        holder=payload.get("holder")
    )
    
    meta = {
        "holder": payload.get("holder"),
        "ratio": payload.get("ratio"),
        "px_per_mm": payload.get("px_per_mm"),
    }
    meta.update(grounding)
    
    observe_async(
        " ".join(str(p) for p in parts if p),
        confidence=payload.get("inner_confidence"),
        meta=meta,
    )
    
    world_model_grounding.accumulate_physical_priors(grounding)


def feedback_async(expected: str, observed: str = "") -> None:
    """Report ground truth back to the Observer (reinforcement)."""
    if not enabled() or not expected:
        return

    def _worker() -> None:
        _post_json(
            "/feedback",
            {"source": _SOURCE, "expected": expected, "observed": observed},
        )

    threading.Thread(target=_worker, daemon=True, name="quipu-feedback").start()


def guidance(force: bool = False) -> dict[str, Any]:
    """Cached learnings from the Observer. {} when unavailable."""
    global _guidance_fetched_at
    if not enabled():
        return {}
    with _guidance_lock:
        fresh = (time.monotonic() - _guidance_fetched_at) < _GUIDANCE_TTL_S
        if _guidance_cache and fresh and not force:
            return _guidance_cache
    fetched = _get_json(f"/guidance?source={_SOURCE}") or {}
    with _guidance_lock:
        if fetched:
            _guidance_cache.clear()
            _guidance_cache.update(fetched)
            _guidance_fetched_at = time.monotonic()
        return dict(_guidance_cache)


def number_priors() -> dict[str, float]:
    """Cross-corpus frequency priors for numeric tokens.

    Built from the mesh's numeric lexicon — numbers observed by BOTH the
    unstructured Loadopoly-OCR corpus and prior Bakugo scans. Used to break
    ties between catalog collector numbers that OCR alone cannot separate.
    """
    g = guidance()
    entries = g.get("numeric_lexicon") or []
    total = sum(int(e.get("freq", 0)) for e in entries) or 1
    return {
        str(e.get("token", "")).lower(): int(e.get("freq", 0)) / total
        for e in entries
        if e.get("token")
    }
