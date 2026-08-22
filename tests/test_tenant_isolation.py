"""Tests for multi-tenant user data isolation and firewalling in Bakugo."""

from __future__ import annotations

import json
import time
import pytest

from cardcenter.store import ScanStore, LabelKind


@pytest.fixture()
def isolated_store(tmp_path):
    db_path = str(tmp_path / "tenant_test.db")
    store = ScanStore(db_path)
    
    # User A scans
    store.add_scan_record({
        "card_key": "userA_card_1",
        "holder": "raw",
        "worst_ratio_pct": 52.0,
        "worst_ratio_sigma": 1.0,
        "worst_axis": "horizontal",
        "h_ratio_pct": 52.0,
        "v_ratio_pct": 50.0,
        "source": "serve",
        "device_id": "device_user_AAA",
        "created_at": time.time() - 100,
    })
    store.add_scan_record({
        "card_key": "userA_card_2",
        "holder": "raw",
        "worst_ratio_pct": 54.0,
        "worst_ratio_sigma": 1.0,
        "worst_axis": "horizontal",
        "h_ratio_pct": 54.0,
        "v_ratio_pct": 50.0,
        "source": "serve",
        "device_id": "device_user_AAA",
        "created_at": time.time() - 50,
    })
    
    # User B scans
    store.add_scan_record({
        "card_key": "userB_card_1",
        "holder": "raw",
        "worst_ratio_pct": 60.0,
        "worst_ratio_sigma": 1.0,
        "worst_axis": "vertical",
        "h_ratio_pct": 50.0,
        "v_ratio_pct": 60.0,
        "source": "serve",
        "device_id": "device_user_BBB",
        "created_at": time.time() - 20,
    })
    
    yield store, db_path
    store.close()


def test_tenant_scans_isolation(isolated_store):
    store, _ = isolated_store
    
    # User A query only returns User A scans
    user_a_scans = store.scans_for_device("device_user_AAA")
    assert len(user_a_scans) == 2
    assert all(s["device_id"] == "device_user_AAA" for s in user_a_scans)
    assert set(s["card_key"] for s in user_a_scans) == {"userA_card_1", "userA_card_2"}
    
    # User B query only returns User B scans
    user_b_scans = store.scans_for_device("device_user_BBB")
    assert len(user_b_scans) == 1
    assert user_b_scans[0]["card_key"] == "userB_card_1"
    
    # Empty / unknown device returns empty list (no data leak)
    assert store.scans_for_device("") == []
    assert store.scans_for_device("unknown_device") == []


def test_tenant_duckdb_analytics_scoping(isolated_store):
    duckdb = pytest.importorskip("duckdb")
    from cardcenter.analytics import AnalyticsEngine
    
    _, db_path = isolated_store
    with AnalyticsEngine(db_path) as engine:
        # Query scoped to User A
        row_a = engine._con.execute("""
            SELECT COUNT(*), AVG(worst_ratio_pct) 
            FROM cc.scans 
            WHERE device_id = 'device_user_AAA'
        """).fetchone()
        assert row_a[0] == 2
        assert row_a[1] == 53.0
        
        # Query scoped to User B
        row_b = engine._con.execute("""
            SELECT COUNT(*), AVG(worst_ratio_pct) 
            FROM cc.scans 
            WHERE device_id = 'device_user_BBB'
        """).fetchone()
        assert row_b[0] == 1
        assert row_b[1] == 60.0
