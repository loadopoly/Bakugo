"""DuckDB analytical layer — zero-copy OLAP over the live SQLite store.

The transactional path (``ScanStore``) writes scans through SQLite in WAL mode.
This module attaches the *same* database file through DuckDB's native SQLite
scanner, giving instant columnar-vectorised aggregations without any ETL or
data copying.  Because SQLite WAL allows unlimited concurrent readers, DuckDB
reads never block active scans and vice-versa.

Optional dependency — ``pip install cardcenter[analytics]`` or
``pip install duckdb>=1.0.0``.  The rest of cardcenter works without it.

Parquet lakehouse
-----------------
``export_parquet()`` sinks the scans table into Hive-partitioned Parquet files
(``year=YYYY/month=MM/*.parquet``) for completely lock-free downstream queries,
dashboard loading, or cross-service federation via Cloudflare-served URLs.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Optional

try:
    import duckdb
except ImportError:  # pragma: no cover
    duckdb = None  # type: ignore[assignment]


class AnalyticsEngine:
    """Attach the live ``cardcenter.db`` via DuckDB's SQLite scanner.

    Parameters
    ----------
    sqlite_path:
        Absolute path to the SQLite WAL database (e.g. ``/data/cardcenter.db``).

    Raises
    ------
    ImportError
        If ``duckdb`` is not installed.
    """

    def __init__(self, sqlite_path: str) -> None:
        if duckdb is None:
            raise ImportError(
                "duckdb is required for the analytics engine.  "
                "Install it with: pip install cardcenter[analytics]"
            )
        self._sqlite_path = sqlite_path
        self._con = duckdb.connect()
        self._con.execute("INSTALL sqlite; LOAD sqlite;")
        self._con.execute(
            f"ATTACH '{sqlite_path}' AS cc (TYPE SQLITE, READ_ONLY);"
        )

    # -- high-level queries ------------------------------------------------

    def scan_summary(self) -> dict[str, Any]:
        """Aggregate scan statistics across the entire store.

        Returns a dict with ``total_scans``, ``distinct_cards``,
        ``avg_centering``, ``min_centering``, ``max_centering``.
        """
        row = self._con.execute("""
            SELECT
                COUNT(*)                          AS total_scans,
                APPROX_COUNT_DISTINCT(phash)      AS distinct_cards,
                ROUND(AVG(worst_ratio_pct), 2)    AS avg_centering,
                ROUND(MIN(worst_ratio_pct), 2)    AS min_centering,
                ROUND(MAX(worst_ratio_pct), 2)    AS max_centering
            FROM cc.scans
        """).fetchone()
        if row is None:
            return {
                "total_scans": 0,
                "distinct_cards": 0,
                "avg_centering": None,
                "min_centering": None,
                "max_centering": None,
            }
        return {
            "total_scans": row[0],
            "distinct_cards": row[1],
            "avg_centering": row[2],
            "min_centering": row[3],
            "max_centering": row[4],
        }

    def centering_distribution(self, bins: int = 20) -> list[dict[str, Any]]:
        """Histogram of ``worst_ratio_pct`` across all scans.

        Returns a list of ``{"bin_lo", "bin_hi", "count"}`` dicts.
        """
        rows = self._con.execute(
            f"""
            WITH bounds AS (
                SELECT
                    MIN(worst_ratio_pct) AS lo,
                    MAX(worst_ratio_pct) AS hi
                FROM cc.scans
            ),
            bins AS (
                SELECT
                    lo + (hi - lo) * i / {bins}       AS bin_lo,
                    lo + (hi - lo) * (i + 1) / {bins}  AS bin_hi,
                    i                                  AS bin_idx
                FROM bounds, generate_series(0, {bins - 1}) AS t(i)
            )
            SELECT
                ROUND(b.bin_lo, 2)  AS bin_lo,
                ROUND(b.bin_hi, 2)  AS bin_hi,
                COUNT(s.id)         AS cnt
            FROM bins b
            LEFT JOIN cc.scans s
                ON s.worst_ratio_pct >= b.bin_lo
               AND (s.worst_ratio_pct < b.bin_hi
                    OR (b.bin_idx = {bins - 1} AND s.worst_ratio_pct <= b.bin_hi))
            GROUP BY b.bin_lo, b.bin_hi
            ORDER BY b.bin_lo
            """
        ).fetchall()
        return [
            {"bin_lo": r[0], "bin_hi": r[1], "count": r[2]} for r in rows
        ]

    def device_leaderboard(self) -> list[dict[str, Any]]:
        """Per-device scan counts and average centering quality.

        Returns a list of ``{"device_id", "scan_count", "avg_centering"}``
        dicts, ordered by scan count descending.
        """
        rows = self._con.execute("""
            SELECT
                COALESCE(NULLIF(device_id, ''), 'local') AS device_id,
                COUNT(*)                                 AS scan_count,
                ROUND(AVG(worst_ratio_pct), 2)           AS avg_centering
            FROM cc.scans
            GROUP BY device_id
            ORDER BY scan_count DESC
        """).fetchall()
        return [
            {"device_id": r[0], "scan_count": r[1], "avg_centering": r[2]}
            for r in rows
        ]

    def label_provenance(self) -> dict[str, int]:
        """Count labels by provenance kind (certified, self_reported, etc.)."""
        rows = self._con.execute("""
            SELECT kind, COUNT(*) AS cnt
            FROM cc.labels
            GROUP BY kind
            ORDER BY cnt DESC
        """).fetchall()
        return {r[0]: r[1] for r in rows}

    def training_export(
        self,
        include_kinds: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """Export scans joined to labels, contamination-firewalled.

        By default only ``certified`` labels are included.  Overriding this
        is possible but the caller must acknowledge the contamination risk.
        """
        kinds = include_kinds or ["certified"]
        placeholders = ", ".join(f"'{k}'" for k in kinds)
        rows = self._con.execute(f"""
            SELECT
                s.*,
                l.grader, l.grade, l.centering_subgrade,
                l.kind, l.cert_number
            FROM cc.scans s
            JOIN cc.labels l ON l.scan_id = s.id
            WHERE l.kind IN ({placeholders})
        """).fetchall()
        columns = [desc[0] for desc in self._con.description]
        return [dict(zip(columns, row)) for row in rows]

    # -- Parquet lakehouse -------------------------------------------------

    def export_parquet(self, output_dir: str) -> str:
        """Sink scans into Hive-partitioned Parquet files.

        Creates ``<output_dir>/year=YYYY/month=MM/scans.parquet``.
        Returns the output directory path.
        """
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        self._con.execute(f"""
            COPY (
                SELECT
                    *,
                    EXTRACT(YEAR  FROM to_timestamp(created_at)) AS year,
                    EXTRACT(MONTH FROM to_timestamp(created_at)) AS month
                FROM cc.scans
            )
            TO '{output}' (
                FORMAT PARQUET,
                PARTITION_BY (year, month),
                OVERWRITE_OR_IGNORE
            )
        """)
        return str(output)

    # -- lifecycle ---------------------------------------------------------

    def detach(self) -> None:
        """Detach the SQLite database without closing DuckDB."""
        try:
            self._con.execute("DETACH cc")
        except Exception:
            pass

    def reattach(self) -> None:
        """Re-attach after the SQLite file may have been replaced."""
        self.detach()
        self._con.execute(
            f"ATTACH '{self._sqlite_path}' AS cc (TYPE SQLITE, READ_ONLY);"
        )

    def close(self) -> None:
        """Close the DuckDB connection entirely."""
        try:
            self._con.close()
        except Exception:
            pass

    def __enter__(self) -> "AnalyticsEngine":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def available() -> bool:
    """Return True if duckdb is installed and the analytics engine can run."""
    return duckdb is not None
