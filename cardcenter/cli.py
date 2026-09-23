"""Command line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from . import __version__
from .centering import measure_centering
from .grading import (
    all_grade_bands,
    available_graders,
    caveat_text,
    grade_band,
    predict_all_grades,
    predict_overall_grade,
)
from .learning import GradeOutcomeModel, maybe_load_grade_model
from .render import save_annotated
from .types import SLAB_PRESETS, CaptureSpec, CenteringResult, DetectionError

DISCLAIMER = (
    "This measures centering only. It does not assess corners, edges, or "
    "surface, and it is not a grade prediction. Centering sets a ceiling on "
    "the grade; the other three attributes decide where under that ceiling a "
    "card lands. A tool that reports a single grade from one photograph is "
    "overclaiming, and if that number is wired to anything that moves money, "
    "the overclaim becomes someone else's loss."
)


def _result_dict(res: CenteringResult, bands: dict, model: Optional[GradeOutcomeModel] = None) -> dict:
    def pair(p):
        rp = p.ratio_pct
        lo, hi = rp.interval()
        return {
            "axis": p.axis,
            "low_name": p.low_name,
            "high_name": p.high_name,
            "low_mm": round(p.low_mm.value, 4),
            "low_sigma_mm": round(p.low_mm.sigma, 4),
            "high_mm": round(p.high_mm.value, 4),
            "high_sigma_mm": round(p.high_mm.sigma, 4),
            "ratio_pct": round(rp.value, 3),
            "ratio_sigma_pct": round(rp.sigma, 3),
            "ratio_ci95": [round(lo, 3), round(hi, 3)],
            "wider_side": p.skew_toward,
        }

    return {
        "schema": "cardcenter/1",
        "version": __version__,
        "horizontal": pair(res.horizontal),
        "vertical": pair(res.vertical),
        "worst_axis": res.worst_axis.axis,
        "worst_ratio_pct": round(res.worst_ratio.value, 3),
        "worst_ratio_ci95": [round(x, 3) for x in res.worst_ratio.interval()],
        "holder": res.slab.name,
        "px_per_mm": round(res.px_per_mm, 3),
        "quality": {
            "outer_residual_px": round(res.quality.outer_residual_px, 3),
            "inner_confidence": round(res.quality.inner_confidence, 3),
            "inner_confidence_per_side": {
                k: round(v, 3) for k, v in res.quality.inner_confidence_per_side.items()
            },
            "refraction_applied": res.quality.refraction_applied,
            "max_refraction_shift_mm": round(res.quality.max_refraction_shift_mm, 4),
            "warnings": res.quality.warnings,
        },
        "grade_bands": {
            name: {
                "best": b.best,
                "worst": b.worst,
                "limited_by": b.limited_by,
                "measurement_span": b.measurement_span,
                "standards_span": b.standards_span,
                "table_confidence": b.grader_confidence,
            }
            for name, b in bands.items()
        },
        "predicted_grades": {
            name: {
                "grade": pred.grade_label,
                "score": pred.grade_score,
                "condition": pred.condition_name,
                "subgrades": {
                    "centering": pred.centering_subgrade,
                    "corners": pred.estimated_corners,
                    "edges": pred.estimated_edges,
                    "surface": pred.estimated_surface,
                },
                "probabilities": pred.probabilities,
                "confidence": pred.confidence,
                "used_learned": pred.used_learned,
                "n_observations": pred.n_observations,
            }
            for name, pred in {
                g: predict_overall_grade(
                    res.worst_ratio, quality=res.quality, grader=g, model=model
                )
                for g in bands.keys()
            }.items()
        },
        "disclaimer": DISCLAIMER,
    }


def _print_human(
    res: CenteringResult,
    bands: dict,
    face: str,
    model: Optional[GradeOutcomeModel] = None,
) -> None:
    print()
    print("=" * 66)
    print("  CENTERING MEASUREMENT")
    print("=" * 66)
    print()
    print(f"  {res.horizontal.describe()}")
    print(f"  {res.vertical.describe()}")
    print()
    print(
        f"  borders (mm):  left {res.horizontal.low_mm}   right {res.horizontal.high_mm}"
    )
    print(
        f"                 top  {res.vertical.low_mm}   bottom {res.vertical.high_mm}"
    )
    print()
    print(f"  worst axis: {res.worst_axis.axis} at {res.worst_ratio.value:.1f}/"
          f"{100 - res.worst_ratio.value:.1f}")
    print(f"  holder: {res.slab.name}    rectified scale: {res.px_per_mm:.1f} px/mm")
    if res.quality.refraction_applied:
        print(
            f"  refraction corrected, max shift "
            f"{res.quality.max_refraction_shift_mm:.3f} mm"
        )
    print(f"  detection confidence: {res.quality.inner_confidence:.2f}")

    print()
    print("-" * 66)
    print(f"  ESTIMATED CARD GRADES & SUBGRADES ({face.upper()})")
    print("-" * 66)
    from .grading import predict_overall_grade
    for name in bands.keys():
        pred = predict_overall_grade(
            res.worst_ratio, quality=res.quality, grader=name, face=face, model=model
        )
        learned = (
            f"  [learned from {pred.n_observations} certs]" if pred.used_learned else ""
        )
        print(f"  {name:<5} -> {pred.grade_label:<8} ({pred.condition_name:<16})  [Confidence: {int(pred.confidence*100)}%]{learned}")
        print(f"        Subgrades: Centering {pred.centering_subgrade:.1f} | Corners {pred.estimated_corners:.1f} | Edges {pred.estimated_edges:.1f} | Surface {pred.estimated_surface:.1f}")
        prob_str = ", ".join(f"{g}: {int(p*100)}%" for g, p in sorted(pred.probabilities.items(), key=lambda kv: -kv[1]))
        print(f"        Probabilities: {prob_str}")
    print()

    print("-" * 66)
    print(f"  GRADE CEILING FROM CENTERING ALONE ({face})")
    print("-" * 66)
    for name, b in bands.items():
        label = b.best if b.is_single else f"{b.worst} - {b.best}"
        conf = "" if b.grader_confidence == "medium" else f"  [table confidence: {b.grader_confidence}]"
        print(f"  {name:<5} {label}{conf}")
        print(f"        limited by {b.limited_by}")
    print()


    if res.quality.warnings:
        print("-" * 66)
        print("  WARNINGS")
        print("-" * 66)
        for w in res.quality.warnings:
            print(f"  * {w}")
        print()

    print("-" * 66)
    for chunk in DISCLAIMER.split(". "):
        if chunk.strip():
            print(f"  {chunk.strip().rstrip('.')}.")
    print("-" * 66)
    print()


SCAN_DISCLAIMER = (
    "Centering only. At display-case distance a phone resolves roughly 50-250um "
    "per pixel, while edge whitening and corner wear are 50-200um features, so "
    "corners, edges and surface are not measurable here and are not being "
    "assessed. This tool can rule a card OUT on centering. It can never rule one "
    "IN."
)


def _run_identify(image, path) -> int:
    """Name the card in a photograph, or say why it cannot be named.

    Identification is rectification-tolerant in a way metrology is not, so this
    path deliberately keeps going where measure_centering would refuse: it
    rectifies from the FULL-resolution pixels (glyph size is the whole game for
    OCR, unlike centering where accuracy is flat above 1200 px) and reads the
    card as sparse text.
    """
    import numpy as np

    from .geometry import find_card_quad, rectify
    from .recognise import recognise_card
    from .types import STANDARD_CARD_W_MM

    detect_long = 1400
    scale = detect_long / max(image.shape[:2])
    small = (
        cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else image
    )
    h, w = small.shape[:2]
    try:
        quad, _, _ = find_card_quad(small, prefer_point=(w / 2.0, h / 2.0))
    except DetectionError as exc:
        print(f"\nNOT IDENTIFIED\n\n  could not locate a card: {exc}\n", file=sys.stderr)
        return 1

    quad_full = quad / (scale if scale < 1.0 else 1.0)
    px_per_mm = float(np.linalg.norm(quad_full[1] - quad_full[0])) / STANDARD_CARD_W_MM
    try:
        rect, _ = rectify(image, quad_full, px_per_mm=min(px_per_mm, 24.0))
    except DetectionError as exc:
        print(f"\nNOT IDENTIFIED\n\n  could not rectify: {exc}\n", file=sys.stderr)
        return 1

    result = recognise_card(rect)
    print()
    print(f"  {path.name}")
    print(f"  {result.describe()}")
    print(f"  read {result.tokens_considered} text tokens via {result.engine}")
    print()
    return 0 if result.resolved else 1


def _run_scan(args, video: bool) -> int:
    from .multicard import scan_image, scan_video
    from .store import ScanStore

    capture = CaptureSpec(focal_px=args.focal_px) if args.focal_px else CaptureSpec()

    if video:
        print(f"  scanning {args.video} (every {args.stride} frames)...")
        report = scan_video(
            args.video,
            holder=args.slab,
            capture=capture,
            stride=args.stride,
            max_frames=args.max_frames,
        )
    else:
        image = cv2.imread(args.image, cv2.IMREAD_COLOR)
        if image is None:
            print(f"error: could not decode image: {args.image}", file=sys.stderr)
            return 2
        if args.fov:
            capture = CaptureSpec.from_fov(args.fov, image.shape)
        report = scan_image(
            image,
            holder=args.slab,
            capture=capture,
            enforce_quality=not args.no_quality_gate,
        )

    print()
    print("=" * 70)
    print("  BATCH SCAN")
    print("=" * 70)
    print(report.summary())
    print()

    measured = sorted(report.measured(), key=lambda c: c.worst_ratio.value)
    if measured:
        print("  ranked best-centred first (ceiling is the pessimistic end):")
        print()
        for c in measured:
            w = c.worst_ratio
            band = grade_band(w, "PSA", args.face)
            lo, hi = w.interval()
            label = band.best if band.is_single else f"{band.worst}-{band.best}"
            print(
                f"    card {c.card_id:<3} {w.value:5.1f}/{100 - w.value:4.1f}  "
                f"CI {lo:5.1f}-{hi:5.1f}  PSA ceiling {label:<7} "
                f"({c.n_measured} frame(s))"
            )
    unmeasured = report.unmeasured()
    if unmeasured:
        print()
        print(f"  {len(unmeasured)} card(s) not measured:")
        for c in unmeasured:
            reasons = c.failure_reasons
            print(f"    card {c.card_id:<3} {reasons[0][:80] if reasons else 'no usable frame'}")

    if args.db:
        from .cloud import sync_store

        with ScanStore(args.db) as st:
            n = 0
            for c in report.measured():
                best = max(
                    (o for o in c.observations if o.result is not None),
                    key=lambda o: o.result.quality.inner_confidence,
                    default=None,
                )
                if best is None:
                    continue
                st.add_scan(
                    args.card_key, best.result, phash=c.phash,
                    source=args.video or args.image or "scan",
                )
                n += 1
            print()
            print(f"  recorded {n} scan(s) in {args.db}")
            print("  no grade labels were written. Labels must come from a grader,")
            print("  with a cert number, or they cannot train anything.")
            cloud = sync_store(st)
            if cloud.skipped:
                print("  cloud mirror skipped (set CARDCENTER_SUPABASE_URL + ANON_KEY).")
            elif cloud.ok:
                print(f"  cloud mirror wrote {cloud.count} scan(s) to Loadopoly-OCR Supabase.")
            else:
                print(f"  cloud mirror failed: {cloud.error}")

    print()
    print("-" * 70)
    for chunk in SCAN_DISCLAIMER.split(". "):
        if chunk.strip():
            print(f"  {chunk.strip().rstrip('.')}.")
    print("-" * 70)
    print()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cardcenter",
        description="Measure trading card centering from a photograph.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Capture guidance:\n"
            "  * Fill the frame with the card; leave a small margin of plain,\n"
            "    contrasting background on all four sides.\n"
            "  * Shoot as close to straight-on as you can. Tilt is corrected but\n"
            "    it costs effective resolution.\n"
            "  * Even, diffuse light. Cross-polarise if the card is slabbed or\n"
            "    foiled; glare is the single most common cause of refusal.\n"
            "  * Pass --fov or --focal-px so tilt can be recovered. Without it,\n"
            "    slabbed measurements carry an inflated error bar."
        ),
    )
    p.add_argument("image", nargs="?", help="path to the card photograph")
    p.add_argument(
        "--slab",
        "--holder",
        dest="slab",
        default="raw",
        help="holder type; use case_* variants when shooting through display glass",
    )
    p.add_argument("--scan", action="store_true",
                   help="wide shot or binder page: measure every card in the image")
    p.add_argument("--video", metavar="PATH",
                   help="scan a video pan across a display case")
    p.add_argument("--stride", type=int, default=6,
                   help="process every Nth video frame (default 6)")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--db", metavar="PATH", help="record scans in this SQLite store")
    p.add_argument("--card-key", default="unidentified",
                   help="identifier to file scans under in the store")
    p.add_argument("--ar", action="store_true",
                   help="start Perceptopoly, the AR capture front end")
    p.add_argument("--serve", action="store_true",
                   help="start the local web UI (use this on a phone)")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--lan", action="store_true",
                   help="also listen on the local network, so another device can reach it")
    p.add_argument("--circularity", action="store_true",
                   help="report how much of the label pool is independent")
    p.add_argument(
        "--ingest-grades",
        action="store_true",
        help="rebuild the grade-outcome model from certified labels in --db",
    )
    p.add_argument("--no-quality-gate", action="store_true",
                   help="measure every detection, even blurred or glared ones")
    p.add_argument("--face", default="front", choices=["front", "back"])
    p.add_argument("--fov", type=float, help="horizontal field of view in degrees")
    p.add_argument("--focal-px", type=float, help="focal length in pixels")
    p.add_argument(
        "--grader",
        action="append",
        help="restrict to one grader (repeatable). Default: all.",
    )
    p.add_argument("--json", metavar="PATH", help="write machine-readable results")
    p.add_argument("--overlay", metavar="PATH", help="write an annotated image")
    p.add_argument("--quiet", action="store_true", help="suppress the human report")
    p.add_argument("--list-slabs", action="store_true")
    p.add_argument("--list-graders", action="store_true")
    p.add_argument(
        "--self-test",
        type=int,
        metavar="N",
        help="render N synthetic cards with known centering and report accuracy",
    )
    p.add_argument(
        "--identify",
        action="store_true",
        help=(
            "name the card instead of measuring it. Works on photographs too "
            "rough for metrology -- bulk bins, sleeves, glare -- because "
            "reading a name needs legible glyphs, not a sub-millimetre boundary"
        ),
    )
    p.add_argument("--capabilities", action="store_true", help="list supported engine capabilities")
    p.add_argument("--check-updates", action="store_true", help="check upstream GitHub repository for updates")
    p.add_argument("--check-health", metavar="URL", help="check health and latency of a remote cardcenter endpoint")
    p.add_argument("--sync-url", metavar="URL", help="synchronize local database with a remote endpoint")
    p.add_argument(
        "--sync-cloud",
        action="store_true",
        help="push unsynced local scans to the Loadopoly-OCR Supabase project",
    )
    p.add_argument(
        "--transfer-supabase",
        action="store_true",
        help="extract and transfer all Supabase items/tables into DuckDB and Parquet lakehouse",
    )
    p.add_argument(
        "--target-duckdb",
        metavar="PATH",
        default=None,
        help="target DuckDB database path (default: $BAKUGO_VAULT_DB, else supabase_vault.duckdb)",
    )
    p.add_argument(
        "--analytics-summary",
        action="store_true",
        help="query and display DuckDB OLAP scan summary and device leaderboard",
    )
    p.add_argument("--auth-token", metavar="TOKEN", help="bearer authentication token for remote sync")
    p.add_argument("--migrate-db", metavar="PATH", help="migrate SQLite database to the latest schema version")
    p.add_argument("--info", action="store_true", help="show detailed build, runtime, and schema version info")
    p.add_argument("--version", action="version", version=f"cardcenter {__version__}")
    return p



def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_slabs:
        for name, s in sorted(SLAB_PRESETS.items()):
            if s.is_optically_active:
                print(
                    f"  {name:<14} t={s.acrylic_thickness_mm:.2f}"
                    f"±{s.acrylic_thickness_sigma_mm:.2f}mm  n={s.refractive_index:.3f}"
                )
            else:
                print(f"  {name:<14} no optical correction")
        print()
        print("  Thicknesses are estimates of the SOLID PLASTIC face wall, not")
        print("  total slab depth. Measure your own holders with calipers and")
        print("  pass a custom spec for best results.")
        return 0

    if args.list_graders:
        for name in available_graders():
            print(f"  {name}")
            for ln in caveat_text(name).splitlines():
                if ln.strip():
                    print(f"      {ln.strip()}")
        return 0

    if args.ar:
        from .perceptopoly import serve as ar_serve

        ar_serve(host="0.0.0.0" if args.lan else "127.0.0.1", port=args.port or 8766)
        return 0

    if args.serve:
        from .serve import serve

        if args.db:
            import os

            os.environ["CARDCENTER_DB"] = args.db
        serve(host="0.0.0.0" if args.lan else "127.0.0.1", port=args.port)
        return 0

    if args.circularity:
        if not args.db:
            print("error: --circularity needs --db", file=sys.stderr)
            return 2
        from .store import ScanStore

        with ScanStore(args.db) as st:
            print()
            print(st.circularity_report())
            print()
        return 0

    if args.ingest_grades:
        if not args.db:
            print("error: --ingest-grades needs --db", file=sys.stderr)
            return 2
        from .learning import LearningStore, ingest_certified_labels
        from .store import ScanStore

        with ScanStore(args.db) as scans, LearningStore(args.db) as learned:
            model = ingest_certified_labels(scans, learned)
        print()
        print(
            f"  ingested {model.observations()} certified grade label(s) "
            f"into {args.db}"
        )
        print("  only CERTIFIED rows with a cert number were used.")
        print()
        return 0

    if args.info:
        from .versioning import get_version_info
        print()
        print(get_version_info().describe())
        print()
        return 0

    if args.capabilities:
        from .versioning import get_engine_capabilities
        print()
        print(get_engine_capabilities().describe())
        print()
        return 0

    if args.check_updates:
        from .versioning import check_for_updates
        print("\nChecking upstream GitHub repository for updates...")
        res = check_for_updates()
        print()
        print(res.describe())
        print()
        return 0

    if args.check_health:
        from .connection import ConnectionManager, ConnectionSpec
        spec = ConnectionSpec(url=args.check_health, auth_token=args.auth_token)
        mgr = ConnectionManager()
        print(f"\nChecking health of {spec.normalized_url()}...")
        health = mgr.check_health(spec)
        print()
        print(f"Status       : {health.status.value.upper()}")
        print(f"Latency      : {health.latency_ms:.1f} ms")
        if health.server_version:
            print(f"Server Ver   : v{health.server_version} ({health.schema_version})")
            print(f"Total Scans  : {health.scan_count}")
        if health.error_message:
            print(f"Error Detail : {health.error_message}")
        print()
        return 0 if health.is_healthy() else 1

    if args.sync_cloud:
        if not args.db:
            print("error: --sync-cloud requires --db <PATH>", file=sys.stderr)
            return 2
        from .cloud import sync_store
        from .store import ScanStore

        with ScanStore(args.db) as store:
            print(f"\nMirroring {args.db} to Loadopoly-OCR Supabase...")
            result = sync_store(store)
            if result.skipped:
                print("  skipped: set CARDCENTER_SUPABASE_URL and CARDCENTER_SUPABASE_ANON_KEY.")
                print()
                return 2
            if result.ok:
                print(f"  wrote {result.count} scan(s). Photos were not uploaded.")
                print()
                return 0
    if args.transfer_supabase:
        from .transfer_supabase import run_transfer
        from .vault import vault_path

        args.target_duckdb = vault_path(args.target_duckdb)
        print(f"\nInitiating Supabase -> DuckDB transfer into '{args.target_duckdb}'...")
        summary = run_transfer(
            duckdb_path=args.target_duckdb,
            sqlite_path=args.db or "cardcenter.db",
        )
        print(f"\n[OK] Transferred {summary['total_tables']} table(s), {summary['total_rows']} row(s) to DuckDB & Parquet.")
        print(f"  DuckDB Vault : {summary['duckdb_database']}")
        print(f"  Parquet Sinks: {summary['parquet_directory']}")
        print(f"  Reconciled   : {summary['sqlite_reconciliation']}\n")
        return 0

    if args.analytics_summary:
        from .analytics import AnalyticsEngine
        sqlite_p = args.db or "cardcenter.db"
        with AnalyticsEngine(sqlite_p) as engine:
            s = engine.scan_summary()
            print("\n=== CardCenter DuckDB OLAP Summary ===")
            print(f"Total Scans   : {s['total_scans']}")
            print(f"Distinct Cards: {s['distinct_cards']}")
            print(f"Avg Centering : {s['avg_centering']}% (min: {s['min_centering']}%, max: {s['max_centering']}%)")
            board = engine.device_leaderboard()
            if board:
                print("\nDevice Leaderboard:")
                for b in board:
                    print(f"  * {b['device_id']:20s}: {b['scan_count']:4d} scans (avg: {b['avg_centering']}%)")
            print()
        return 0

    if args.sync_url:
        if not args.db:
            print("error: --sync-url requires --db <PATH> to specify local store", file=sys.stderr)
            return 2
        from .connection import ConnectionManager, ConnectionSpec
        from .store import ScanStore
        spec = ConnectionSpec(url=args.sync_url, auth_token=args.auth_token)
        mgr = ConnectionManager()
        with ScanStore(args.db) as store:
            print(f"\nSynchronizing {args.db} with {spec.normalized_url()}...")
            sync_res = mgr.sync(store, spec)
            print()
            print(sync_res.describe())
            from .learning import LearningStore

            with LearningStore(args.db) as learned:
                n = learned.load_grade_model().observations()
            print(f"  Grade model now holds {n} certified observation(s).")
            print()
            return 0 if sync_res.success else 1

    if args.migrate_db:
        from .versioning import migrate_database
        print(f"\nMigrating database '{args.migrate_db}'...")
        final_ver = migrate_database(args.migrate_db)
        print(f"Database migrated to schema version '{final_ver}'.\n")
        return 0

    if args.video:
        return _run_scan(args, video=True)


    if args.self_test is not None:
        from validate import report, run  # noqa: PLC0415

        report(run(args.self_test))
        return 0

    if not args.image:
        build_parser().print_help()
        return 2

    if args.scan:
        return _run_scan(args, video=False)

    path = Path(args.image)
    if not path.exists():
        print(f"error: no such file: {path}", file=sys.stderr)
        return 2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"error: could not decode image: {path}", file=sys.stderr)
        return 2

    if args.identify:
        return _run_identify(image, path)

    if args.focal_px:
        capture = CaptureSpec(focal_px=args.focal_px)
    elif args.fov:
        capture = CaptureSpec.from_fov(args.fov, image.shape)
    else:
        capture = CaptureSpec()

    try:
        res = measure_centering(image, slab=args.slab, capture=capture)
    except DetectionError as exc:
        print(f"\nMEASUREMENT REFUSED\n\n  {exc}\n", file=sys.stderr)
        print(
            "  No number is reported because none would be trustworthy.\n",
            file=sys.stderr,
        )
        return 1

    names = args.grader or available_graders()
    bands = {}
    for n in names:
        try:
            bands[n] = grade_band(res.worst_ratio, n, args.face)
        except KeyError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    model = maybe_load_grade_model(args.db)

    if not args.quiet:
        _print_human(res, bands, args.face, model=model)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(_result_dict(res, bands, model=model), fh, indent=2)
        print(f"  wrote {args.json}")

    if args.overlay:
        save_annotated(args.overlay, res, bands)
        print(f"  wrote {args.overlay}")

    if args.db:
        from .cloud import sync_scan_id
        from .store import ScanStore

        with ScanStore(args.db) as store:
            scan_id = store.add_scan(
                args.card_key, res, source=str(path),
            )
            print(f"  recorded scan {scan_id} in {args.db}")
            cloud = sync_scan_id(store, scan_id)
            if cloud.skipped:
                print("  cloud mirror skipped (set CARDCENTER_SUPABASE_URL + ANON_KEY).")
            elif cloud.ok:
                print("  cloud mirror wrote this scan to Loadopoly-OCR Supabase.")
            else:
                print(f"  cloud mirror failed: {cloud.error}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
