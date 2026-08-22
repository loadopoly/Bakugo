"""World model grounding for Bakugo observations.

Connects to the active QUIPU Docker container to synchronize physical grounding,
lossy channel invariants, and refractive indices without local disk dependencies.
"""

from __future__ import annotations

import os
import threading
from typing import Any

# Standard physical invariants fallback
_DEFAULT_REFRACTIVE_INDICES: dict[str, float] = {
    "psa": 1.491,   # PMMA
    "bgs": 1.586,   # PC
    "cgc": 1.586,   # PC
    "raw": 1.000,   # Air
}


def grounding_annotation(
    channel_conditions: dict[str, Any] | None = None,
    ratio: float | None = None,
    sigma: float | None = None,
    crb_sigma: float | None = None,
    holder: str | None = None,
) -> dict[str, Any]:
    """Build a world-model grounding annotation for a metrology observation.
    
    Quantifies how much information survived the physical channel:
    - information_efficiency: ratio of achieved vs theoretical CRB limit (0..1)
    - lossy_channel_profile: what physical processes destroyed information
    - physical_invariants: geometric truths learned from this measurement
    - grounding_confidence: how reliable this physical grounding is
    """
    channel = channel_conditions or {}
    
    # Calculate information efficiency
    efficiency = 0.0
    if crb_sigma is not None and sigma is not None:
        efficiency = min(1.0, crb_sigma / max(sigma, 1e-12))
        
    # Determine lossy channel profile
    profile = []
    if isinstance(channel, dict):
        for k, v in channel.items():
            if k in ("blur", "noise", "refraction", "glare", "quantization") and v:
                profile.append(k)
    
    invariants: dict[str, Any] = {}
    if holder:
        holder_lower = str(holder).lower()
        for k, n in _DEFAULT_REFRACTIVE_INDICES.items():
            if k in holder_lower:
                invariants["refractive_index"] = n
                break
            
    confidence = 0.8 if efficiency > 0.5 else 0.4
    
    return {
        "information_efficiency": efficiency,
        "lossy_channel_profile": profile,
        "physical_invariants": invariants,
        "grounding_confidence": confidence,
    }


def accumulate_physical_priors(grounding: dict[str, Any]) -> dict[str, Any]:
    """Accumulate physical-space priors.
    
    Grounding annotations are sent directly to the QUIPU Docker container
    via observe_measure_async.
    """
    return grounding


def physical_world_summary() -> dict[str, Any]:
    """Return a summary of what has been learned about physical space
    from the active QUIPU container.
    """
    try:
        from . import quipu_client
        g = quipu_client.guidance()
        wm = g.get("world_model") or {}
        if wm:
            return wm
        state = quipu_client.fetch_state()
        return state or {}
    except Exception:
        return {}
