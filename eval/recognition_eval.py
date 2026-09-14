#!/usr/bin/env python3
"""
recognition_eval.py -- standalone evaluator for the Bakugo card-recognition path.

WHAT IT CONFIRMS (grounded in the actual backend, not assumptions):
  - OCR: cardcenter.recognise.recognise_card -> Tesseract (SparseTextEngine),
    species snap (Levenshtein, refuses on ties), collector-number read.
  - "VLM" integration: there is NO vision-LLM in the card path -- ocr.py uses
    Tesseract on purpose ("a vision LLM fails by returning something plausible
    but wrong"). The real model integration is cardcenter.quipu_client, which
    supplies OCR number PRIORS from QUIPU. This harness measures its actual
    contribution by running with it enabled vs disabled. If/when a true
    VLM or embedding recognizer is added behind recognise_card, this same
    harness measures whether it beats the Tesseract baseline -- until then the
    VLM row reports "not integrated," honestly.
  - The accept/abstain tradeoff: accuracy on the auto-accepted subset vs the
    fraction abstained, so you get "auto-accepts N% of cards at M% accuracy"
    instead of a hope. Sliced by capture resolution (px/mm) so the AR distance
    dependence is visible.

STANDALONE BY DESIGN:
  - Own entrypoint. Imports cardcenter READ-ONLY; never touches the serving app,
    the database, or the live AR loop.
  - No network by default (Tesseract is local; the catalog/price lookup that
    needs Scryfall/PokemonTCG is opt-in).
  - Runs where cardcenter + Tesseract + the labeled data live.

USAGE:
  python recognition_eval.py --data labels.jsonl
  python recognition_eval.py --data labels.csv --quipu both --report out.json
  python recognition_eval.py --data scans.parquet --image-col image_path \
        --name-col true_name --number-col true_number

MANIFEST (jsonl or csv), one card per row, images are RECTIFIED card crops
(the same thing the AR loop feeds recognise_card); pass --detect to run
detection+crop on raw photos instead:
  {"image": "cards/lechonk.png", "name": "Lechonk", "number": 182, "px_per_mm": 11.4}
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


def _load_app():
    try:
        import numpy as np  # noqa: F401
        import cv2  # noqa: F401
        from cardcenter.recognise import recognise_card, load_species, SparseTextEngine
    except Exception as exc:  # pragma: no cover - environment guard
        sys.exit(
            f"cannot import the app: {exc}\n"
            "Run this from the Bakugo repo root (so 'cardcenter' imports), "
            "with numpy, opencv and Tesseract installed."
        )
    return recognise_card, load_species, SparseTextEngine


@dataclass
class Sample:
    image: str
    name: Optional[str] = None
    number: Optional[int] = None
    printing: Optional[str] = None
    px_per_mm: Optional[float] = None


@dataclass
class Row:
    image: str
    truth_name: Optional[str]
    pred_name: Optional[str]
    resolved: bool
    ambiguous: bool
    truth_number: Optional[int]
    pred_number: Optional[int]
    px_per_mm: Optional[float]
    engine: str
    name_correct: Optional[bool]
    number_correct: Optional[bool]
    error: str = ""


def load_manifest(path: str, cols: dict) -> list:
    p = Path(path)
    if not p.exists():
        sys.exit(f"data file not found: {path}")
    ext = p.suffix.lower()
    if ext == ".jsonl":
        rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    elif ext == ".csv":
        with p.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
    elif ext == ".parquet":
        try:
            import pandas as pd
        except Exception:
            sys.exit("reading .parquet needs pandas: pip install pandas pyarrow")
        rows = pd.read_parquet(p).to_dict("records")
    else:
        sys.exit(f"unsupported data format: {ext} (use .jsonl, .csv or .parquet)")

    out = []
    for r in rows:
        out.append(
            Sample(
                image=str(r[cols["image"]]),
                name=_opt_str(r.get(cols["name"])) if cols.get("name") else None,
                number=_opt_int(r.get(cols["number"])) if cols.get("number") else None,
                printing=_opt_str(r.get(cols.get("printing", ""))) if cols.get("printing") else None,
                px_per_mm=_opt_float(r.get(cols.get("px_per_mm", ""))) if cols.get("px_per_mm") else None,
            )
        )
    return out


def _opt_str(v):
    return None if v is None or str(v).strip() == "" else str(v).strip()


def _opt_int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _opt_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _maybe_detect_and_crop(img, detect: bool):
    if not detect:
        return img, ""
    try:
        from cardcenter.geometry import find_card_quad  # type: ignore
        from cardcenter.multicard import _crop_quad  # type: ignore
        quad, _, _ = find_card_quad(img)
        crop, _ = _crop_quad(img, quad)
        return crop, ""
    except Exception as exc:
        return img, f"detect failed ({exc}); used full frame"


def run_one(sample, recognise_card, vocab, engine, detect: bool):
    import cv2

    img = cv2.imread(sample.image)
    if img is None:
        return Row(sample.image, sample.name, None, False, False, sample.number,
                   None, sample.px_per_mm, "n/a", None, None, "image unreadable")

    crop, note = _maybe_detect_and_crop(img, detect)
    try:
        rec = recognise_card(crop, vocabulary=vocab, engine=engine)
    except Exception as exc:
        return Row(sample.image, sample.name, None, False, False, sample.number,
                   None, sample.px_per_mm, "error", None, None, f"recognise raised: {exc}")

    pred_name = rec.name
    pred_num = rec.dex
    resolved = bool(rec.resolved()) if callable(getattr(rec, "resolved", None)) else bool(pred_name)
    ambiguous = bool(getattr(rec, "alternatives", ()))

    name_correct = None
    if sample.name is not None:
        name_correct = (pred_name is not None
                        and pred_name.strip().lower() == sample.name.strip().lower())
    number_correct = None
    if sample.number is not None:
        number_correct = (pred_num is not None and int(pred_num) == int(sample.number))

    return Row(
        image=sample.image, truth_name=sample.name, pred_name=pred_name,
        resolved=resolved, ambiguous=ambiguous, truth_number=sample.number,
        pred_number=pred_num, px_per_mm=sample.px_per_mm,
        engine=getattr(rec, "engine", "?"),
        name_correct=name_correct, number_correct=number_correct, error=note,
    )


# Accept policies, loosest -> strictest. Coverage falls, accuracy rises: that
# tradeoff IS the answer to "how much can it fire-and-forget."
POLICIES = {
    "resolved":            lambda r: r.resolved,
    "resolved+number":     lambda r: r.resolved and r.pred_number is not None,
    "resolved+number+unambiguous": lambda r: r.resolved and r.pred_number is not None and not r.ambiguous,
}


def summarize(rows: list) -> dict:
    labeled = [r for r in rows if r.name_correct is not None]
    n = len(labeled)
    out = {
        "total": len(rows),
        "labeled": n,
        "read_errors": sum(1 for r in rows if r.error and "unreadable" in r.error),
        "engine": _mode([r.engine for r in rows]),
    }
    if n == 0:
        out["note"] = "no ground-truth labels present; only coverage was measured"
        out["resolved_rate"] = round(_safe_div(sum(r.resolved for r in rows), len(rows)), 4)
        return out

    curve = {}
    for name, accept in POLICIES.items():
        acc = [r for r in labeled if accept(r)]
        coverage = _safe_div(len(acc), n)
        accuracy = _safe_div(sum(1 for r in acc if r.name_correct), len(acc)) if acc else None
        num_acc = [r for r in acc if r.number_correct is not None]
        number_accuracy = _safe_div(sum(1 for r in num_acc if r.number_correct), len(num_acc)) if num_acc else None
        curve[name] = {
            "coverage": round(coverage, 4),
            "name_accuracy_on_accepted": _r(accuracy),
            "number_accuracy_on_accepted": _r(number_accuracy),
            "abstained": round(1 - coverage, 4),
        }
    out["accept_curve"] = curve
    out["raw_name_accuracy"] = round(_safe_div(sum(1 for r in labeled if r.name_correct), n), 4)

    buckets = {"<6 px/mm": [], "6-9": [], "9-12": [], ">=12": [], "unknown": []}
    for r in labeled:
        buckets[_bucket(r.px_per_mm)].append(r)
    out["by_resolution"] = {
        b: {
            "n": len(rs),
            "name_accuracy": _r(_safe_div(sum(1 for r in rs if r.name_correct), len(rs))) if rs else None,
            "resolved_rate": _r(_safe_div(sum(r.resolved for r in rs), len(rs))) if rs else None,
        }
        for b, rs in buckets.items() if rs
    }
    return out


def _bucket(ppm):
    if ppm is None:
        return "unknown"
    if ppm < 6:
        return "<6 px/mm"
    if ppm < 9:
        return "6-9"
    if ppm < 12:
        return "9-12"
    return ">=12"


def _safe_div(a, b):
    return (a / b) if b else 0.0


def _r(x):
    return None if x is None else round(x, 4)


def _mode(xs):
    from collections import Counter
    return Counter(xs).most_common(1)[0][0] if xs else "?"


def evaluate(samples: list, detect: bool, quipu_mode: str) -> dict:
    recognise_card, load_species, SparseTextEngine = _load_app()
    vocab = list(load_species())

    def one_pass(force_quipu):
        prev = os.environ.get("QUIPU_URL")
        if force_quipu is False:
            os.environ.pop("QUIPU_URL", None)
        try:
            try:
                from cardcenter.quipu_client import enabled
                quipu_on = bool(enabled())
            except Exception:
                quipu_on = False
            engine = SparseTextEngine()
            rows = [run_one(s, recognise_card, vocab, engine, detect) for s in samples]
            summary = summarize(rows)
            summary["quipu_enabled"] = quipu_on
            summary["rows"] = [asdict(r) for r in rows]
            return summary
        finally:
            if prev is not None:
                os.environ["QUIPU_URL"] = prev

    result = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "n_samples": len(samples)}
    if quipu_mode == "both":
        result["quipu_off"] = one_pass(force_quipu=False)
        result["quipu_on"] = one_pass(force_quipu=None)
        result["quipu_delta_name_accuracy"] = _delta(result["quipu_on"], result["quipu_off"])
    else:
        result["run"] = one_pass(force_quipu=(False if quipu_mode == "off" else None))
    return result


def _delta(on: dict, off: dict):
    a = on.get("raw_name_accuracy")
    b = off.get("raw_name_accuracy")
    return None if a is None or b is None else round(a - b, 4)


def print_summary(result: dict) -> None:
    def show(tag, s):
        print(f"\n== {tag} ==  (quipu_enabled={s.get('quipu_enabled')})")
        print(f"  samples: {s['total']}  labeled: {s['labeled']}  engine: {s['engine']}")
        if "raw_name_accuracy" in s:
            print(f"  raw name accuracy (no abstention): {s['raw_name_accuracy']:.1%}")
        for name, c in s.get("accept_curve", {}).items():
            acc = c["name_accuracy_on_accepted"]
            acc_s = f"{acc:.1%}" if acc is not None else "n/a"
            print(f"    accept[{name:<28}] covers {c['coverage']:.0%} at {acc_s} accuracy")
        for b, d in s.get("by_resolution", {}).items():
            acc = d["name_accuracy"]
            acc_s = f"{acc:.1%}" if acc is not None else "n/a"
            print(f"    {b:<10} n={d['n']:<4} name acc {acc_s}")

    print(f"\nrecognition eval  {result['generated']}  n={result['n_samples']}")
    if "run" in result:
        show("run", result["run"])
    else:
        show("QUIPU off", result["quipu_off"])
        show("QUIPU on", result["quipu_on"])
        d = result.get("quipu_delta_name_accuracy")
        if d is not None:
            print(f"\n  QUIPU integration contribution: {d:+.1%} name accuracy")


def main() -> None:
    ap = argparse.ArgumentParser(description="Standalone Bakugo recognition evaluator")
    ap.add_argument("--data", required=True, help="labels .jsonl / .csv / .parquet")
    ap.add_argument("--image-col", default="image")
    ap.add_argument("--name-col", default="name")
    ap.add_argument("--number-col", default="number")
    ap.add_argument("--printing-col", default="")
    ap.add_argument("--pxmm-col", default="px_per_mm")
    ap.add_argument("--detect", action="store_true",
                    help="run detection+crop on raw photos (default: inputs are rectified crops)")
    ap.add_argument("--quipu", choices=["on", "off", "both"], default="both",
                    help="run with the QUIPU prior integration on, off, or both to measure its delta")
    ap.add_argument("--report", default="", help="write full JSON report here")
    args = ap.parse_args()

    cols = {"image": args.image_col, "name": args.name_col, "number": args.number_col,
            "printing": args.printing_col, "px_per_mm": args.pxmm_col}
    samples = load_manifest(args.data, cols)
    if not samples:
        sys.exit("no samples loaded")

    result = evaluate(samples, detect=args.detect, quipu_mode=args.quipu)
    print_summary(result)
    if args.report:
        Path(args.report).write_text(json.dumps(result, indent=2, default=str))
        print(f"\nfull report -> {args.report}")


if __name__ == "__main__":
    main()
