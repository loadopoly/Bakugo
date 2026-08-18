"""Tests for Semantic Versioning, Capability Matrix, Schema Migrations, and Update Checks."""

import sqlite3
import pytest

from cardcenter import __version__
from cardcenter.versioning import (
    CURRENT_SCHEMA_VERSION,
    SCHEMA_V1,
    SCHEMA_V2,
    EngineCapabilities,
    SchemaMigrationError,
    SemVer,
    UpdateStatus,
    check_for_updates,
    get_db_schema_version,
    get_engine_capabilities,
    get_version_info,
    migrate_database,
    set_db_schema_version,
)


def test_semver_parsing() -> None:
    sv = SemVer.parse("1.9.0")
    assert sv.major == 1
    assert sv.minor == 9
    assert sv.patch == 0
    assert sv.prerelease is None
    assert str(sv) == "1.9.0"

    sv_pre = SemVer.parse("v2.0.0-rc.1+build42")
    assert sv_pre.major == 2
    assert sv_pre.minor == 0
    assert sv_pre.patch == 0
    assert sv_pre.prerelease == "rc.1"
    assert sv_pre.build == "build42"
    assert sv_pre.is_prerelease

    sv_short = SemVer.parse("1.8")
    assert sv_short.major == 1
    assert sv_short.minor == 8
    assert sv_short.patch == 0


def test_semver_comparison() -> None:
    v1 = SemVer.parse("1.8.0")
    v2 = SemVer.parse("1.9.0")
    v3 = SemVer.parse("2.0.0-alpha")
    v4 = SemVer.parse("2.0.0")

    assert v1 < v2
    assert v2 <= v2
    assert v2 < v3
    assert v3 < v4
    assert v4 > v1
    assert v2 == SemVer(1, 9, 0)
    assert v1 != v2


def test_semver_compatibility() -> None:
    v1_9 = SemVer.parse("1.9.0")
    v1_9_1 = SemVer.parse("1.9.1")
    v1_10 = SemVer.parse("1.10.0")
    v2_0 = SemVer.parse("2.0.0")

    assert v1_9.is_compatible_with(v1_9_1)
    assert v1_9.is_compatible_with(v1_10)
    assert not v1_9.is_compatible_with(v2_0)


def test_version_info_structure() -> None:
    info = get_version_info()
    assert info.version == __version__
    assert info.schema_version == CURRENT_SCHEMA_VERSION
    assert info.python_version != ""
    assert info.numpy_version != ""
    desc = info.describe()
    assert "Bakugo / CardCenter" in desc
    assert "Python" in desc


def test_engine_capabilities_matrix() -> None:
    caps = get_engine_capabilities()
    assert isinstance(caps, EngineCapabilities)
    assert caps.has_subpixel_edge_detection is True
    assert caps.has_stack_refraction_solver is True
    assert caps.has_provenance_firewall is True
    assert caps.has_remote_sync_protocol is True
    assert CURRENT_SCHEMA_VERSION in caps.supported_schemas
    desc = caps.describe()
    assert "has_subpixel_edge_detection" in desc


def test_schema_version_and_migration(tmp_path) -> None:
    db_path = str(tmp_path / "test_migration.db")

    # Create a legacy v1 style database
    conn = sqlite3.connect(db_path)
    conn.execute(
        """CREATE TABLE scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            card_key TEXT NOT NULL,
            holder TEXT NOT NULL,
            worst_ratio_pct REAL NOT NULL,
            worst_ratio_sigma REAL NOT NULL,
            worst_axis TEXT NOT NULL,
            h_ratio_pct REAL NOT NULL,
            v_ratio_pct REAL NOT NULL,
            left_mm REAL, right_mm REAL, top_mm REAL, bottom_mm REAL,
            px_per_mm REAL,
            inner_confidence REAL,
            refraction_applied INTEGER,
            warnings TEXT,
            phash INTEGER,
            source TEXT,
            created_at REAL NOT NULL
        )"""
    )
    conn.execute(
        """CREATE TABLE labels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_id INTEGER NOT NULL,
            grader TEXT NOT NULL,
            grade TEXT NOT NULL,
            centering_subgrade TEXT,
            kind TEXT NOT NULL,
            cert_number TEXT,
            attributed_to TEXT,
            created_at REAL NOT NULL
        )"""
    )
    conn.commit()

    # Verify detected as schema v1
    v = get_db_schema_version(conn)
    assert v == SCHEMA_V1
    conn.close()

    # Run migration to V2
    final_ver = migrate_database(db_path)
    assert final_ver == SCHEMA_V2

    # Verify columns were added
    conn2 = sqlite3.connect(db_path)
    conn2.row_factory = sqlite3.Row
    cols = {r["name"] for r in conn2.execute("PRAGMA table_info(scans)").fetchall()}
    assert "client_uuid" in cols
    assert "integrity_hash" in cols
    assert "synced_at" in cols

    lbl_cols = {r["name"] for r in conn2.execute("PRAGMA table_info(labels)").fetchall()}
    assert "provenance_hash" in lbl_cols

    assert get_db_schema_version(conn2) == SCHEMA_V2
    conn2.close()


def test_check_for_updates() -> None:
    res = check_for_updates(timeout_sec=2.0)
    assert res.current_version == __version__
    assert res.status in (
        UpdateStatus.UP_TO_DATE,
        UpdateStatus.UPDATE_AVAILABLE,
        UpdateStatus.PRE_RELEASE_AHEAD,
        UpdateStatus.OFFLINE_OR_UNAVAILABLE,
    )
    desc = res.describe()
    assert "Installed version" in desc
