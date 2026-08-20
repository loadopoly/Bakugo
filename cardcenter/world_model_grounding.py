"""World model grounding for Bakugo observations."""

from __future__ import annotations

import json
import os
import threading
from typing import Any

_lock = threading.Lock()

def _get_db_path() -> str:
    base = os.environ.get("CARDCENTER_DB", ".")
    return os.path.join(base, ".world_model.json")

def _load_state() -> dict[str, Any]:
    path = _get_db_path()
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {
            "measurements_count": 0,
            "mean_information_efficiency": 0.0,
            "lossy_channel_profiles": {},
            "physical_invariants": {}
        }

def _save_state(state: dict[str, Any]) -> None:
    path = _get_db_path()
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f)
    except OSError:
        pass

def grounding_annotation(
    channel_conditions: dict[str, Any] | None = None,
    ratio: float | None = None,
    sigma: float | None = None,
    crb_sigma: float | None = None,
    holder: str | None = None
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
    
    invariants = {}
    if holder:
        holder_lower = str(holder).lower()
        if "psa" in holder_lower:
            invariants["refractive_index"] = 1.491  # PMMA
        elif "bgs" in holder_lower or "cgc" in holder_lower:
            invariants["refractive_index"] = 1.586  # PC
            
    confidence = 0.8 if efficiency > 0.5 else 0.4
    
    return {
        "information_efficiency": efficiency,
        "lossy_channel_profile": profile,
        "physical_invariants": invariants,
        "grounding_confidence": confidence
    }

def accumulate_physical_priors(grounding: dict[str, Any]) -> dict[str, Any]:
    """Accumulate physical-space priors over time.
    
    Tracks running statistics of:
    - Mean information efficiency across measurements
    - Distribution of lossy channel profiles (blur, noise, refraction, glare)
    - Invariant laws discovered (e.g., Snell angles for specific holder types)
    
    Returns the accumulated physical priors dict.
    """
    with _lock:
        state = _load_state()
        
        count = state.get("measurements_count", 0)
        mean_eff = state.get("mean_information_efficiency", 0.0)
        
        # Update mean information efficiency
        eff = grounding.get("information_efficiency", 0.0)
        new_mean = (mean_eff * count + eff) / (count + 1)
        state["mean_information_efficiency"] = new_mean
        state["measurements_count"] = count + 1
        
        # Update lossy channel profiles
        profiles = state.setdefault("lossy_channel_profiles", {})
        for p in grounding.get("lossy_channel_profile", []):
            profiles[p] = profiles.get(p, 0) + 1
            
        # Update physical invariants
        invariants = state.setdefault("physical_invariants", {})
        for k, v in grounding.get("physical_invariants", {}).items():
            invariants[k] = v
            
        _save_state(state)
        return state

def physical_world_summary() -> dict[str, Any]:
    """Return a summary of what Bakugo has learned about physical space.
    
    This is included in QUIPU observations so the Observer can build
    a world model that understands information loss in physical reality.
    """
    with _lock:
        return _load_state()
