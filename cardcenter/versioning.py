"""Versioning, capability matrix, schema migrations, and remote repository integration.

This module provides:
1. Semantic version parsing, validation, and compatibility comparison (SemVer).
2. Engine capability introspection across computer vision, metrology, hardware, and APIs.
3. Database schema version tracking and forward migration engine for ScanStore SQLite databases.
4. Remote update checking against the GitHub repository (https://github.com/PoodlesOfWar/Bakugo).
"""

from __future__ import annotations

import json
import os
import platform
import re
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import __version__

# Canonical repository and API endpoints
GITHUB_REPO = "PoodlesOfWar/Bakugo"
GITHUB_API_RELEASES_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
GITHUB_TAGS_URL = f"https://api.github.com/repos/{GITHUB_REPO}/tags"
USER_AGENT = f"cardcenter-versioning/{__version__} (+https://github.com/{GITHUB_REPO})"

# Schema versions
SCHEMA_V1 = "cardcenter/1"
SCHEMA_V2 = "cardcenter/2"
CURRENT_SCHEMA_VERSION = SCHEMA_V2


# ---------------------------------------------------------------------------
# Semantic Versioning
# ---------------------------------------------------------------------------


@dataclass(frozen=True, order=False)
class SemVer:
    """Standard Semantic Version (SemVer 2.0.0 compliant)."""

    major: int
    minor: int
    patch: int
    prerelease: Optional[str] = None
    build: Optional[str] = None

    @classmethod
    def parse(cls, version_str: str) -> "SemVer":
        """Parse a version string (e.g. '1.9.0', 'v2.0.0-rc.1+build123')."""
        clean = version_str.strip().lstrip("vV")
        pattern = r"^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)(?:-(?P<prerelease>[0-9A-Za-z.-]+))?(?:\+(?P<build>[0-9A-Za-z.-]+))?$"
        match = re.match(pattern, clean)
        if not match:
            # Fallback for 2-part versions (e.g. '1.9')
            two_part = re.match(r"^(?P<major>\d+)\.(?P<minor>\d+)(?:-(?P<prerelease>[0-9A-Za-z.-]+))?$", clean)
            if two_part:
                return cls(
                    major=int(two_part.group("major")),
                    minor=int(two_part.group("minor")),
                    patch=0,
                    prerelease=two_part.group("prerelease"),
                )
            raise ValueError(f"Invalid semantic version string: '{version_str}'")

        groups = match.groupdict()
        return cls(
            major=int(groups["major"]),
            minor=int(groups["minor"]),
            patch=int(groups["patch"]),
            prerelease=groups.get("prerelease"),
            build=groups.get("build"),
        )

    @property
    def is_prerelease(self) -> bool:
        return bool(self.prerelease)

    def tuple(self) -> Tuple[int, int, int]:
        return (self.major, self.minor, self.patch)

    def __str__(self) -> str:
        s = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            s += f"-{self.prerelease}"
        if self.build:
            s += f"+{self.build}"
        return s

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SemVer):
            try:
                other = SemVer.parse(str(other))
            except Exception:
                return False
        return (
            self.major == other.major
            and self.minor == other.minor
            and self.patch == other.patch
            and self.prerelease == other.prerelease
        )

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, SemVer):
            other = SemVer.parse(str(other))
        if self.tuple() != other.tuple():
            return self.tuple() < other.tuple()
        # Non-prerelease is higher precedence than prerelease
        if self.prerelease is None and other.prerelease is not None:
            return False
        if self.prerelease is not None and other.prerelease is None:
            return True
        if self.prerelease is not None and other.prerelease is not None:
            return self.prerelease < other.prerelease
        return False

    def __le__(self, other: object) -> bool:
        return self == other or self < other

    def __gt__(self, other: object) -> bool:
        return not (self <= other)

    def __ge__(self, other: object) -> bool:
        return not (self < other)

    def is_compatible_with(self, other: "SemVer") -> bool:
        """Major versions must match for 1.x+; 0.x versions require minor version match."""
        if self.major != other.major:
            return False
        if self.major == 0 and self.minor != other.minor:
            return False
        return True


@dataclass
class VersionInfo:
    """Comprehensive environment and build version information."""

    version: str
    semver: SemVer
    schema_version: str
    git_commit: Optional[str]
    python_version: str
    platform_system: str
    platform_release: str
    platform_machine: str
    opencv_version: str
    numpy_version: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "semver": str(self.semver),
            "schema_version": self.schema_version,
            "git_commit": self.git_commit,
            "python_version": self.python_version,
            "platform": {
                "system": self.platform_system,
                "release": self.platform_release,
                "machine": self.platform_machine,
            },
            "dependencies": {
                "opencv": self.opencv_version,
                "numpy": self.numpy_version,
            },
        }

    def describe(self) -> str:
        lines = [
            f"Bakugo / CardCenter v{self.version} ({self.schema_version})",
            f"  Python       : {self.python_version} ({self.platform_system} {self.platform_machine})",
            f"  OpenCV       : {self.opencv_version}",
            f"  NumPy        : {self.numpy_version}",
            f"  Git Commit   : {self.git_commit or 'n/a (packaged release)'}",
            f"  Repository   : https://github.com/{GITHUB_REPO}",
        ]
        return "\n".join(lines)


def get_git_commit() -> Optional[str]:
    """Retrieve git HEAD hash if running from a git repository."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return None


def get_version_info() -> VersionInfo:
    """Assemble runtime version info."""
    import cv2
    import numpy as np

    sv = SemVer.parse(__version__)
    return VersionInfo(
        version=__version__,
        semver=sv,
        schema_version=CURRENT_SCHEMA_VERSION,
        git_commit=get_git_commit(),
        python_version=platform.python_version(),
        platform_system=platform.system(),
        platform_release=platform.release(),
        platform_machine=platform.machine(),
        opencv_version=getattr(cv2, "__version__", "unknown"),
        numpy_version=getattr(np, "__version__", "unknown"),
    )


# ---------------------------------------------------------------------------
# Capability Matrix & Feature Flags
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EngineCapabilities:
    """Flags describing what this build can measure, solve, and connect to."""

    # Metrology & CV features
    has_subpixel_edge_detection: bool = True
    has_stack_refraction_solver: bool = True
    has_multicard_batch_scan: bool = True
    has_ar_live_tracking: bool = True
    has_caliper_scale_calibration: bool = True
    has_two_view_fusion: bool = True
    has_tesseract_ocr: bool = False
    
    # Financial & Market features
    has_almgren_chriss_liquidation: bool = True
    has_grade_premium_valuation: bool = True
    has_active_learning_curator: bool = True
    
    # Connection & Storage features
    has_provenance_firewall: bool = True
    has_remote_sync_protocol: bool = True
    has_github_update_checker: bool = True
    has_sqlite_store: bool = True
    supported_schemas: Tuple[str, ...] = (SCHEMA_V1, SCHEMA_V2)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def describe(self) -> str:
        lines = ["Engine Capabilities:"]
        for k, v in sorted(asdict(self).items()):
            if isinstance(v, (list, tuple)):
                val_str = ", ".join(str(x) for x in v)
                lines.append(f"  {k:<32}: [{val_str}]")
            else:
                mark = "[+]" if v else "[-]"
                lines.append(f"  {k:<32}: {mark} {v}")
        return "\n".join(lines)



def get_engine_capabilities() -> EngineCapabilities:
    """Inspect current environment for available optional capabilities."""
    # Check OCR (pytesseract or tesseract binary)
    has_tess = False
    try:
        import pytesseract  # type: ignore
        has_tess = True
    except ImportError:
        try:
            r = subprocess.run(["tesseract", "--version"], capture_output=True, timeout=1.0, check=False)
            has_tess = (r.returncode == 0)
        except Exception:
            has_tess = False

    return EngineCapabilities(
        has_tesseract_ocr=has_tess,
        has_subpixel_edge_detection=True,
        has_stack_refraction_solver=True,
        has_multicard_batch_scan=True,
        has_ar_live_tracking=True,
        has_caliper_scale_calibration=True,
        has_two_view_fusion=True,
        has_almgren_chriss_liquidation=True,
        has_grade_premium_valuation=True,
        has_active_learning_curator=True,
        has_provenance_firewall=True,
        has_remote_sync_protocol=True,
        has_github_update_checker=True,
        has_sqlite_store=True,
        supported_schemas=(SCHEMA_V1, SCHEMA_V2),
    )


# ---------------------------------------------------------------------------
# Database Schema Versioning & Forward Migrations
# ---------------------------------------------------------------------------


class SchemaMigrationError(Exception):
    """Raised when a schema migration fails or encountering an incompatible future schema."""


def get_db_schema_version(conn: sqlite3.Connection) -> str:
    """Query current database schema version from metadata table."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS _schema_metadata ("
        "  key TEXT PRIMARY KEY, "
        "  value TEXT NOT NULL, "
        "  updated_at REAL NOT NULL"
        ")"
    )
    row = conn.execute("SELECT value FROM _schema_metadata WHERE key = 'schema_version'").fetchone()
    if row is None:
        # Check if legacy scans table exists
        table = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='scans'"
        ).fetchone()
        if table:
            # Legacy v1 database without metadata table
            return SCHEMA_V1
        return CURRENT_SCHEMA_VERSION
    return str(row[0])


def set_db_schema_version(conn: sqlite3.Connection, version: str) -> None:
    """Store schema version in metadata table."""
    conn.execute(
        "INSERT INTO _schema_metadata (key, value, updated_at) VALUES ('schema_version', ?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
        (version, time.time()),
    )
    conn.commit()


def migrate_database(db_path: str) -> str:
    """Apply all pending migrations to an existing SQLite database.
    
    Returns the final schema version string.
    """
    path = Path(db_path)
    if not path.exists():
        # Fresh database will be created directly at latest schema
        return CURRENT_SCHEMA_VERSION

    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        current = get_db_schema_version(conn)
        if current == CURRENT_SCHEMA_VERSION:
            return current

        if current == SCHEMA_V1:
            # Migration from V1 to V2:
            # 1. Add provenance integrity hash to scans if missing
            # 2. Add client_uuid, sync_id and integrity columns
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(scans)")
            columns = {row["name"] for row in cursor.fetchall()}
            
            if "client_uuid" not in columns:
                cursor.execute("ALTER TABLE scans ADD COLUMN client_uuid TEXT DEFAULT ''")
            if "integrity_hash" not in columns:
                cursor.execute("ALTER TABLE scans ADD COLUMN integrity_hash TEXT DEFAULT ''")
            if "synced_at" not in columns:
                cursor.execute("ALTER TABLE scans ADD COLUMN synced_at REAL DEFAULT 0.0")

            # Check labels table
            cursor.execute("PRAGMA table_info(labels)")
            label_cols = {row["name"] for row in cursor.fetchall()}
            if "provenance_hash" not in label_cols:
                cursor.execute("ALTER TABLE labels ADD COLUMN provenance_hash TEXT DEFAULT ''")

            set_db_schema_version(conn, SCHEMA_V2)
            conn.commit()
            return SCHEMA_V2

        raise SchemaMigrationError(
            f"Unsupported or newer database schema version: '{current}'. "
            f"This engine supports up to '{CURRENT_SCHEMA_VERSION}'."
        )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Remote Repository & Release Update Checker
# ---------------------------------------------------------------------------


class UpdateStatus(str, Enum):
    UP_TO_DATE = "up_to_date"
    UPDATE_AVAILABLE = "update_available"
    PRE_RELEASE_AHEAD = "pre_release_ahead"
    OFFLINE_OR_UNAVAILABLE = "offline_or_unavailable"


@dataclass(frozen=True)
class UpdateCheckResult:
    status: UpdateStatus
    current_version: str
    latest_version: Optional[str]
    release_name: Optional[str]
    release_url: Optional[str]
    published_at: Optional[str]
    changelog_summary: Optional[str]
    checked_at: float
    error_message: Optional[str] = None

    def describe(self) -> str:
        lines = [f"Update Status: {self.status.value.upper()}"]
        lines.append(f"  Installed version : v{self.current_version}")
        if self.latest_version:
            lines.append(f"  Latest upstream   : v{self.latest_version}")
        if self.release_name:
            lines.append(f"  Release title     : {self.release_name}")
        if self.release_url:
            lines.append(f"  Release URL       : {self.release_url}")
        if self.error_message:
            lines.append(f"  Notice            : {self.error_message}")
        if self.status == UpdateStatus.UPDATE_AVAILABLE:
            lines.append("  Action required   : Run 'git pull' or 'pip install --upgrade .'")
        return "\n".join(lines)


def check_for_updates(
    repo: str = GITHUB_REPO,
    timeout_sec: float = 4.0,
) -> UpdateCheckResult:
    """Check GitHub repository for newer releases or tags.

    Connects to the official repository at https://github.com/PoodlesOfWar/Bakugo.
    Gracefully handles offline environments, rate limits, and network errors.
    """
    now = time.time()
    current_sv = SemVer.parse(__version__)

    url = f"https://api.github.com/repos/{repo}/releases/latest"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/vnd.github.v3+json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            tag = data.get("tag_name", "").lstrip("vV")
            if not tag:
                tag = data.get("name", "").lstrip("vV")

            if tag:
                remote_sv = SemVer.parse(tag)
                status = (
                    UpdateStatus.UPDATE_AVAILABLE
                    if remote_sv > current_sv
                    else UpdateStatus.PRE_RELEASE_AHEAD
                    if remote_sv < current_sv
                    else UpdateStatus.UP_TO_DATE
                )
                return UpdateCheckResult(
                    status=status,
                    current_version=__version__,
                    latest_version=str(remote_sv),
                    release_name=data.get("name"),
                    release_url=data.get("html_url", f"https://github.com/{repo}/releases"),
                    published_at=data.get("published_at"),
                    changelog_summary=data.get("body", "")[:300] if data.get("body") else None,
                    checked_at=now,
                )
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            # Fallback to tags if no official github release published yet
            return _check_tags_fallback(repo, current_sv, now, timeout_sec)
        return UpdateCheckResult(
            status=UpdateStatus.OFFLINE_OR_UNAVAILABLE,
            current_version=__version__,
            latest_version=None,
            release_name=None,
            release_url=f"https://github.com/{repo}",
            published_at=None,
            changelog_summary=None,
            checked_at=now,
            error_message=f"GitHub API returned HTTP {exc.code}: {exc.reason}",
        )
    except Exception as exc:
        return UpdateCheckResult(
            status=UpdateStatus.OFFLINE_OR_UNAVAILABLE,
            current_version=__version__,
            latest_version=None,
            release_name=None,
            release_url=f"https://github.com/{repo}",
            published_at=None,
            changelog_summary=None,
            checked_at=now,
            error_message=f"Could not reach GitHub: {exc}",
        )

    return _check_tags_fallback(repo, current_sv, now, timeout_sec)


def _check_tags_fallback(
    repo: str,
    current_sv: SemVer,
    now: float,
    timeout_sec: float,
) -> UpdateCheckResult:
    """Fallback query to /tags endpoint."""
    tags_url = f"https://api.github.com/repos/{repo}/tags"
    req = urllib.request.Request(
        tags_url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/vnd.github.v3+json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            tags = json.loads(resp.read().decode("utf-8"))
            if isinstance(tags, list) and len(tags) > 0:
                tag_name = tags[0].get("name", "").lstrip("vV")
                remote_sv = SemVer.parse(tag_name)
                status = (
                    UpdateStatus.UPDATE_AVAILABLE
                    if remote_sv > current_sv
                    else UpdateStatus.PRE_RELEASE_AHEAD
                    if remote_sv < current_sv
                    else UpdateStatus.UP_TO_DATE
                )
                return UpdateCheckResult(
                    status=status,
                    current_version=__version__,
                    latest_version=str(remote_sv),
                    release_name=f"Tag {tag_name}",
                    release_url=f"https://github.com/{repo}/tree/{tag_name}",
                    published_at=None,
                    changelog_summary=None,
                    checked_at=now,
                )
    except Exception as exc:
        return UpdateCheckResult(
            status=UpdateStatus.OFFLINE_OR_UNAVAILABLE,
            current_version=__version__,
            latest_version=None,
            release_name=None,
            release_url=f"https://github.com/{repo}",
            published_at=None,
            changelog_summary=None,
            checked_at=now,
            error_message=f"GitHub repository check: {exc}",
        )

    return UpdateCheckResult(
        status=UpdateStatus.UP_TO_DATE,
        current_version=__version__,
        latest_version=__version__,
        release_name="Current Head",
        release_url=f"https://github.com/{repo}",
        published_at=None,
        changelog_summary=None,
        checked_at=now,
    )
