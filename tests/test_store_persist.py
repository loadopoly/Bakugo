"""Local ScanStore persist helpers used by serve / CLI / Pages."""

from __future__ import annotations

from cardcenter.store import ScanStore


def test_add_scan_from_measure_and_unsynced(tmp_path) -> None:
    db = tmp_path / "scans.db"
    payload = {
        "ok": True,
        "card_key": "sv01-mew",
        "holder": "raw",
        "ratio": 55.0,
        "ratio_lo": 53.0,
        "ratio_hi": 57.0,
        "axis": "horizontal",
        "borders": {"left": 2.7, "right": 3.3, "top": 3.0, "bottom": 3.0},
        "px_per_mm": 12.0,
        "inner_confidence": 0.91,
        "refraction": False,
        "warnings": ["glare"],
    }
    with ScanStore(str(db)) as store:
        scan_id = store.add_scan_from_measure(payload, source="test")
        assert scan_id == 1
        row = store.get_scan(scan_id)
        assert row is not None
        assert row["card_key"] == "sv01-mew"
        assert row["worst_ratio_pct"] == 55.0
        assert abs(row["worst_ratio_sigma"] - (4.0 / 3.92)) < 1e-9
        assert row["source"] == "test"
        pending = store.unsynced_scans()
        assert len(pending) == 1
        store.mark_synced(scan_id)
        assert store.unsynced_scans() == []
        assert store.scan_count() == 1
