"""Tests for Supabase to DuckDB transfer engine (``cardcenter.transfer_supabase``)."""

from __future__ import annotations

import json
import os
import tempfile
import pytest

duckdb = pytest.importorskip("duckdb")

from cardcenter.transfer_supabase import SupabaseTransferEngine, SUPABASE_PUBLIC_TABLES


def test_public_tables_list():
    assert len(SUPABASE_PUBLIC_TABLES) >= 28
    assert "bakugo_scans" in SUPABASE_PUBLIC_TABLES
    assert "bakugo_labels" in SUPABASE_PUBLIC_TABLES
    assert "historical_documents_global" in SUPABASE_PUBLIC_TABLES
    assert "community_fund" in SUPABASE_PUBLIC_TABLES


def test_transfer_engine_local_table_creation(tmp_path):
    duckdb_p = str(tmp_path / "test_vault.duckdb")
    parquet_p = str(tmp_path / "parquet")
    
    engine = SupabaseTransferEngine(duckdb_path=duckdb_p, parquet_dir=parquet_p)
    
    # Mocking rows for transfer_table
    engine._run_psql_json = lambda sql: (
        [{"id": "123", "card_key": "pikachu", "worst_ratio_pct": 52.5}]
        if "bakugo_scans" in sql
        else []
    )
    
    res = engine.transfer_table("bakugo_scans")
    assert res["table"] == "bakugo_scans"
    assert res["rows"] == 1
    assert os.path.exists(res["parquet"])
    
    # Verify DuckDB query
    val = engine.con.execute("SELECT card_key, worst_ratio_pct FROM bakugo_scans").fetchone()
    assert val[0] == "pikachu"
    assert val[1] == 52.5
    
    engine.close()


def test_transfer_empty_table_schema(tmp_path):
    duckdb_p = str(tmp_path / "test_empty_vault.duckdb")
    parquet_p = str(tmp_path / "parquet_empty")
    
    engine = SupabaseTransferEngine(duckdb_path=duckdb_p, parquet_dir=parquet_p)
    
    def fake_psql(sql):
        if "information_schema.columns" in sql:
            return [
                {"column_name": "id", "data_type": "text"},
                {"column_name": "count", "data_type": "integer"},
                {"column_name": "active", "data_type": "boolean"},
            ]
        return []
        
    engine._run_psql_json = fake_psql
    
    res = engine.transfer_table("archive_partnerships")
    assert res["table"] == "archive_partnerships"
    assert res["rows"] == 0
    assert os.path.exists(res["parquet"])
    
    # Verify table schema was created in DuckDB
    cols = [col[0] for col in engine.con.execute("DESCRIBE archive_partnerships").fetchall()]
    assert "id" in cols
    assert "count" in cols
    assert "active" in cols
    
    engine.close()
