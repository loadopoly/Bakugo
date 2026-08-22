"""Tests for the DuckDB analytical layer (``cardcenter.analytics``).

These tests create a throwaway SQLite WAL database, populate it with synthetic
scans and labels, then verify every ``AnalyticsEngine`` query against known
expected results.
"""

from __future__ import annotations

import os
import tempfile
import time

import pytest

# Skip entire module when duckdb is not installed.
duckdb = pytest.importorskip("duckdb")

from cardcenter.analytics import AnalyticsEngine, available
from cardcenter.store import LabelKind, ScanStore


# ── fixtures ──────────────────────────────────────────────────────────


@pytest.fixture()
def populated_store(tmp_path):
    """Return a ``ScanStore`` seeded with deterministic test data."""
    db_path = str(tmp_path / "test_cardcenter.db")
    store = ScanStore(db_path)
    # Insert 10 scans across 2 devices with varying centering ratios.
    for i in range(10):
        device = "device-A" if i < 6 else "device-B"
        ratio = 50.0 + i * 1.5  # 50.0, 51.5, …, 63.5
        store.conn.execute(
            "INSERT INTO scans "
            "(card_key, holder, worst_ratio_pct, worst_ratio_sigma, worst_axis, "
            " h_ratio_pct, v_ratio_pct, px_per_mm, inner_confidence, "
            " refraction_applied, warnings, phash, source, device_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                f"card-{i:03d}",
                "psa",
                ratio,
                1.0 + i * 0.1,
                "horizontal" if i % 2 == 0 else "vertical",
                ratio,
                50.0,
                28.0,
                0.95,
                0,
                "",
                1000 + i,
                "test",
                device,
                time.time() - (10 - i) * 86400,
            ),
        )
    store.conn.commit()

    # Add a certified label and a self-reported label.
    store.add_label(1, "PSA", "9", LabelKind.CERTIFIED, cert_number="PSA-00001")
    store.add_label(2, "PSA", "8", LabelKind.SELF_REPORTED)

    yield store, db_path
    store.close()


@pytest.fixture()
def engine(populated_store):
    """Return a connected ``AnalyticsEngine`` over the test database."""
    _, db_path = populated_store
    eng = AnalyticsEngine(db_path)
    yield eng
    eng.close()


# ── tests ─────────────────────────────────────────────────────────────


def test_available():
    assert available() is True


def test_scan_summary(engine):
    summary = engine.scan_summary()
    assert summary["total_scans"] == 10
    assert summary["avg_centering"] is not None
    assert 50.0 <= summary["min_centering"] <= 51.0
    assert 63.0 <= summary["max_centering"] <= 64.0


def test_centering_distribution(engine):
    hist = engine.centering_distribution(bins=5)
    assert len(hist) == 5
    assert all("bin_lo" in b and "bin_hi" in b and "count" in b for b in hist)
    total = sum(b["count"] for b in hist)
    assert total == 10


def test_device_leaderboard(engine):
    board = engine.device_leaderboard()
    # Two devices: A (6 scans) and B (4 scans).
    assert len(board) == 2
    assert board[0]["device_id"] == "device-A"
    assert board[0]["scan_count"] == 6
    assert board[1]["device_id"] == "device-B"
    assert board[1]["scan_count"] == 4


def test_label_provenance(engine):
    prov = engine.label_provenance()
    assert prov.get("certified", 0) == 1
    assert prov.get("self_reported", 0) == 1


def test_training_export_default_certified_only(engine):
    rows = engine.training_export()
    assert len(rows) == 1
    assert rows[0]["kind"] == "certified"
    assert rows[0]["cert_number"] == "PSA-00001"


def test_training_export_custom_kinds(engine):
    rows = engine.training_export(include_kinds=["certified", "self_reported"])
    assert len(rows) == 2


def test_export_parquet(engine, tmp_path):
    out = engine.export_parquet(str(tmp_path / "parquet_out"))
    # The output directory should exist and contain partitioned files.
    assert os.path.isdir(out)
    # At least one year= partition should have been created.
    subdirs = list(os.listdir(out))
    assert any(d.startswith("year=") for d in subdirs)


def test_detach_reattach(engine):
    engine.detach()
    engine.reattach()
    summary = engine.scan_summary()
    assert summary["total_scans"] == 10


def test_context_manager(populated_store):
    _, db_path = populated_store
    with AnalyticsEngine(db_path) as eng:
        assert eng.scan_summary()["total_scans"] == 10


def test_import_error_when_duckdb_missing(monkeypatch):
    """Verify graceful ImportError message when duckdb is absent."""
    import cardcenter.analytics as mod

    original = mod.duckdb
    monkeypatch.setattr(mod, "duckdb", None)
    with pytest.raises(ImportError, match="duckdb is required"):
        AnalyticsEngine("/nonexistent.db")
    monkeypatch.setattr(mod, "duckdb", original)
