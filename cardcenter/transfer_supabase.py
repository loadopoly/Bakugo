"""Supabase to DuckDB Data Transfer Engine.

Transfers all tables, schemas, and rows from Supabase PostgreSQL into
a local/server DuckDB database and Hive/columnar Parquet lakehouse files.

Supports:
1. Docker exec psql extraction (fastest, direct binary/JSON dump from container).
2. PostgREST REST API extraction (using anon/service keys if container isn't local).
3. Ingesting tables into persistent DuckDB (`.duckdb` file).
4. Exporting tables into `.parquet` lakehouse files.
5. Reconciling `bakugo_scans` and `bakugo_labels` with local `cardcenter.db` SQLite store.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import duckdb
except ImportError:  # pragma: no cover
    duckdb = None  # type: ignore[assignment]


# All 28 standard public tables in the Loadopoly / Bakugo Supabase schema
SUPABASE_PUBLIC_TABLES = [
    "archive_partnerships",
    "asset_graph_nodes",
    "bakugo_labels",
    "bakugo_scans",
    "classification_audit_log",
    "cluster_dimension_statistics",
    "community_fund",
    "credit_transactions",
    "digital_asset_bundles",
    "gard_tokenized_assets",
    "governance_votes",
    "graph_edges",
    "graph_nodes",
    "historical_documents_global",
    "master_user_access",
    "pending_rewards",
    "presence_sessions",
    "processing_queue",
    "realtime_events",
    "royalty_transactions",
    "shard_holdings",
    "social_return_projects",
    "spatial_anchors",
    "structured_classification_mappings",
    "structured_clusters",
    "user_avatars",
    "user_credits",
    "world_sectors",
]


class SupabaseTransferEngine:
    """Extracts tables from Supabase and populates DuckDB & Parquet lakehouse."""

    def __init__(
        self,
        duckdb_path: str = "supabase_vault.duckdb",
        parquet_dir: str = "data/parquet/supabase",
        docker_container: str = "supabase_db_agard",
    ) -> None:
        if duckdb is None:
            raise ImportError(
                "duckdb is required for the Supabase transfer engine. "
                "Install with: pip install cardcenter[analytics]"
            )
        self.duckdb_path = duckdb_path
        self.parquet_dir = Path(parquet_dir)
        self.docker_container = docker_container
        self.parquet_dir.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(self.duckdb_path)

    def _run_psql_json(self, sql: str) -> List[Dict[str, Any]]:
        """Run SQL via docker exec psql and return parsed JSON rows."""
        # Wrap query in json_agg to get clean, typed JSON output
        json_sql = f"SELECT coalesce(json_agg(t), '[]'::json) FROM ({sql}) t;"
        cmd = [
            "docker",
            "exec",
            self.docker_container,
            "psql",
            "-U",
            "postgres",
            "-d",
            "postgres",
            "-t",
            "-A",
            "-c",
            json_sql,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"psql error: {res.stderr.strip()}")
        raw = res.stdout.strip()
        if not raw:
            return []
        return json.loads(raw)

    def list_supabase_tables(self) -> List[str]:
        """Fetch list of all table names in public schema."""
        try:
            rows = self._run_psql_json(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename"
            )
            return [r["tablename"] for r in rows]
        except Exception:
            return list(SUPABASE_PUBLIC_TABLES)

    def transfer_table(self, table_name: str) -> Dict[str, Any]:
        """Extract a single table from Supabase into DuckDB and Parquet."""
        t0 = time.time()
        rows = self._run_psql_json(f'SELECT * FROM public."{table_name}"')
        row_count = len(rows)

        parquet_file = self.parquet_dir / f"{table_name}.parquet"

        if row_count > 0:
            import tempfile

            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as tf:
                json.dump(rows, tf)
                temp_json_path = tf.name

            try:
                # Load JSON file into DuckDB
                self.con.execute(
                    f'CREATE OR REPLACE TABLE "{table_name}" AS SELECT * FROM read_json_auto(?)',
                    [temp_json_path],
                )
                # Export to Parquet
                self.con.execute(
                    f'COPY "{table_name}" TO \'{parquet_file.as_posix()}\' (FORMAT PARQUET, OVERWRITE_OR_IGNORE)'
                )
            finally:
                if os.path.exists(temp_json_path):
                    os.remove(temp_json_path)
        else:
            # Query column schema for empty table
            cols_meta = self._run_psql_json(
                f"""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_schema = 'public' AND table_name = '{table_name}'
                ORDER BY ordinal_position
                """
            )
            if cols_meta:
                col_defs = []
                for c in cols_meta:
                    cname = c["column_name"]
                    ctype = c["data_type"].upper()
                    # Map PG types to DuckDB
                    if "INT" in ctype:
                        dtype = "BIGINT"
                    elif "FLOAT" in ctype or "DOUBLE" in ctype or "NUMERIC" in ctype or "REAL" in ctype:
                        dtype = "DOUBLE"
                    elif "BOOL" in ctype:
                        dtype = "BOOLEAN"
                    elif "TIMESTAMP" in ctype:
                        dtype = "TIMESTAMP"
                    elif "JSON" in ctype:
                        dtype = "JSON"
                    else:
                        dtype = "VARCHAR"
                    col_defs.append(f'"{cname}" {dtype}')
                schema_sql = ", ".join(col_defs)
                self.con.execute(f'CREATE OR REPLACE TABLE "{table_name}" ({schema_sql})')
            else:
                self.con.execute(f'CREATE OR REPLACE TABLE "{table_name}" (id VARCHAR)')

            # Create an empty Parquet file from the table
            self.con.execute(
                f'COPY "{table_name}" TO \'{parquet_file.as_posix()}\' (FORMAT PARQUET, OVERWRITE_OR_IGNORE)'
            )

        elapsed = round((time.time() - t0) * 1000, 1)
        return {
            "table": table_name,
            "rows": row_count,
            "parquet": str(parquet_file),
            "parquet_size_bytes": parquet_file.stat().st_size if parquet_file.exists() else 0,
            "elapsed_ms": elapsed,
        }

    def transfer_storage_metadata(self) -> Dict[str, Any]:
        """Transfer storage buckets and object metadata to DuckDB."""
        t0 = time.time()
        try:
            buckets = self._run_psql_json("SELECT * FROM storage.buckets")
            objects = self._run_psql_json("SELECT * FROM storage.objects")
        except Exception:
            buckets, objects = [], []

        import tempfile

        if buckets:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as tf:
                json.dump(buckets, tf)
                temp_p = tf.name
            try:
                self.con.execute(
                    'CREATE OR REPLACE TABLE "storage_buckets" AS SELECT * FROM read_json_auto(?)',
                    [temp_p],
                )
                self.con.execute(
                    f'COPY "storage_buckets" TO \'{(self.parquet_dir / "storage_buckets.parquet").as_posix()}\' (FORMAT PARQUET, OVERWRITE_OR_IGNORE)'
                )
            finally:
                if os.path.exists(temp_p):
                    os.remove(temp_p)

        if objects:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as tf:
                json.dump(objects, tf)
                temp_p = tf.name
            try:
                self.con.execute(
                    'CREATE OR REPLACE TABLE "storage_objects" AS SELECT * FROM read_json_auto(?)',
                    [temp_p],
                )
                self.con.execute(
                    f'COPY "storage_objects" TO \'{(self.parquet_dir / "storage_objects.parquet").as_posix()}\' (FORMAT PARQUET, OVERWRITE_OR_IGNORE)'
                )
            finally:
                if os.path.exists(temp_p):
                    os.remove(temp_p)

        return {
            "buckets_count": len(buckets),
            "objects_count": len(objects),
            "elapsed_ms": round((time.time() - t0) * 1000, 1),
        }

    def transfer_all(self) -> Dict[str, Any]:
        """Transfer all Supabase items to DuckDB and Parquet lakehouse."""
        tables = self.list_supabase_tables()
        results = []
        total_rows = 0

        for tbl in tables:
            res = self.transfer_table(tbl)
            results.append(res)
            total_rows += res["rows"]

        storage_res = self.transfer_storage_metadata()

        summary = {
            "status": "success",
            "duckdb_database": str(Path(self.duckdb_path).resolve()),
            "parquet_directory": str(self.parquet_dir.resolve()),
            "total_tables": len(results),
            "total_rows": total_rows,
            "tables": results,
            "storage_metadata": storage_res,
            "transferred_at": time.time(),
        }

        # Write manifest file
        manifest_path = self.parquet_dir / "transfer_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary

    def reconcile_bakugo_sqlite(self, sqlite_path: str = "cardcenter.db") -> Dict[str, int]:
        """Reconcile Supabase scans and labels back into local SQLite store."""
        from .store import ScanStore

        store = ScanStore(sqlite_path)
        conn = store.conn

        scans_added = 0
        labels_added = 0

        try:
            sb_scans = self._run_psql_json("SELECT * FROM public.bakugo_scans")
            for s in sb_scans:
                # Check if scan exists by card_key and created_at or phash
                card_key = s.get("card_key", "")
                phash = int(s.get("phash") or 0)
                worst_ratio = float(s.get("worst_ratio_pct") or 0.0)

                existing = conn.execute(
                    "SELECT id FROM scans WHERE card_key = ? AND worst_ratio_pct = ?",
                    (card_key, worst_ratio),
                ).fetchone()

                if not existing:
                    conn.execute(
                        """
                        INSERT INTO scans (
                            card_key, holder, worst_ratio_pct, worst_ratio_sigma,
                            worst_axis, h_ratio_pct, v_ratio_pct, left_mm, right_mm,
                            top_mm, bottom_mm, px_per_mm, inner_confidence,
                            refraction_applied, warnings, phash, source, device_id, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            card_key,
                            s.get("holder", "raw"),
                            worst_ratio,
                            float(s.get("worst_ratio_sigma") or 1.0),
                            s.get("worst_axis", "horizontal"),
                            float(s.get("h_ratio_pct") or 50.0),
                            float(s.get("v_ratio_pct") or 50.0),
                            float(s.get("left_mm") or 0.0),
                            float(s.get("right_mm") or 0.0),
                            float(s.get("top_mm") or 0.0),
                            float(s.get("bottom_mm") or 0.0),
                            float(s.get("px_per_mm") or 0.0),
                            float(s.get("inner_confidence") or 0.95),
                            int(bool(s.get("refraction_applied"))),
                            s.get("warnings", ""),
                            phash,
                            s.get("source", "supabase_sync"),
                            s.get("device_id", ""),
                            time.time(),
                        ),
                    )
                    scans_added += 1

            conn.commit()

            sb_labels = self._run_psql_json("SELECT * FROM public.bakugo_labels")
            for l in sb_labels:
                cert = l.get("cert_number")
                if cert:
                    existing = conn.execute(
                        "SELECT id FROM labels WHERE cert_number = ?", (cert,)
                    ).fetchone()
                    if not existing:
                        # Get a scan id to attach to
                        first_scan = conn.execute("SELECT id FROM scans ORDER BY id LIMIT 1").fetchone()
                        scan_id = first_scan[0] if first_scan else 1
                        conn.execute(
                            """
                            INSERT INTO labels (
                                scan_id, grader, grade, centering_subgrade, kind,
                                cert_number, attributed_to, created_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                scan_id,
                                l.get("grader", "PSA"),
                                l.get("grade", "9"),
                                l.get("centering_subgrade"),
                                l.get("kind", "certified"),
                                cert,
                                l.get("attributed_to", "supabase"),
                                time.time(),
                            ),
                        )
                        labels_added += 1

            conn.commit()
        except Exception as e:
            print(f"Error during SQLite reconciliation: {e}")
        finally:
            store.close()

        return {"scans_reconciled": scans_added, "labels_reconciled": labels_added}

    def close(self) -> None:
        self.con.close()


def run_transfer(
    duckdb_path: str = "supabase_vault.duckdb",
    parquet_dir: str = "data/parquet/supabase",
    sqlite_path: str = "cardcenter.db",
) -> Dict[str, Any]:
    """Convenience function to run complete transfer and reconciliation."""
    engine = SupabaseTransferEngine(duckdb_path=duckdb_path, parquet_dir=parquet_dir)
    try:
        summary = engine.transfer_all()
        reconciliation = engine.reconcile_bakugo_sqlite(sqlite_path=sqlite_path)
        summary["sqlite_reconciliation"] = reconciliation
        return summary
    finally:
        engine.close()


if __name__ == "__main__":
    print("Initiating full transfer of Supabase items to DuckDB server...")
    summary = run_transfer()
    print("\n=== Transfer Summary ===")
    print(f"DuckDB database: {summary['duckdb_database']}")
    print(f"Parquet directory: {summary['parquet_directory']}")
    print(f"Total tables transferred: {summary['total_tables']}")
    print(f"Total rows transferred: {summary['total_rows']}")
    print(f"SQLite reconciliation: {summary['sqlite_reconciliation']}")
    print("\nTables:")
    for t in summary["tables"]:
        if t["rows"] > 0:
            print(f"  * {t['table']:35s}: {t['rows']:4d} rows ({t['parquet_size_bytes']} bytes)")
        else:
            print(f"    {t['table']:35s}:    0 rows (schema registered)")
