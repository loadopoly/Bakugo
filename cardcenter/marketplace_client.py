"""Bakugo CardCenter Shared Marketplace & Web3 GARD-Shard Client.

Bridges CardCenter structured metrology (Touch axis) into the unified
HubCore Web3 GARD-Shard Shared Marketplace.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

logger = logging.getLogger("cardcenter.marketplace")

DEFAULT_MARKETPLACE_URL = "http://127.0.0.1:8600"


def get_marketplace_url() -> str:
    return os.environ.get("MARKETPLACE_URL") or DEFAULT_MARKETPLACE_URL


def tokenize_metrology_scan(
    scan_id: str,
    title: str,
    contributor_wallet: str,
    user_id: str,
    centering_ratio: float,
    ratio_ci: list[float],
    cramer_rao_floor_px: float,
    grade_ceiling: str,
    holder: str = "raw",
    refraction: bool = False,
    shard_count: int = 218,
    shard_price_base: float = 10.0,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Publish and tokenize a physical card metrology scan to the Shared Marketplace."""
    url = f"{get_marketplace_url()}/api/v1/marketplace/tokenize"
    meta = metadata or {}
    meta.update(
        {
            "worst_ratio": centering_ratio,
            "ratio_ci": ratio_ci,
            "cramer_rao_floor_px": cramer_rao_floor_px,
            "grade_ceiling": grade_ceiling,
            "holder": holder,
            "refraction_applied": refraction,
            "source_app": "Bakugo",
        }
    )

    payload = {
        "asset_id": f"card-{scan_id}",
        "token_id": f"tok-dcc1-card-{scan_id}",
        "axis": "touch",
        "title": title,
        "category": "cardcenter_metrology",
        "contributor_wallet": contributor_wallet,
        "user_id": user_id,
        "shard_count": shard_count,
        "shard_price_base": shard_price_base,
        "royalty_rate": 0.05,
        "is_genesis": False,
        "quality_score": 0.98,
        "precision_score": 0.99,
        "metadata": meta,
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "Bakugo-CardCenter/2.7.0"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        logger.warning("Failed to tokenize to marketplace: %s", exc)
        return {"ok": False, "error": str(exc)}


def get_marketplace_assets(axis: str = "touch", limit: int = 50) -> List[Dict[str, Any]]:
    """Fetch assets from the shared marketplace."""
    url = f"{get_marketplace_url()}/api/v1/marketplace/catalog?axis={axis}&limit={limit}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("assets", [])
    except Exception as exc:
        logger.warning("Failed to fetch marketplace catalog: %s", exc)
        return []
