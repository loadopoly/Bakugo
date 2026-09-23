"""The three phases of the nightly loop.

``day``   (broaden) -- diff the manifest, assign splits, pseudo-label, export
                       the review queue, file QUIPU proposals, export the
                       recognition test set.
``night`` (deepen)  -- fit a candidate per task on confirmed train labels and
                       evaluate it against the champion on the frozen test set.
``dawn``            -- one promotion decision per evaluated candidate.

``sync`` (the network step) is separate and runs in its own container before
``day``.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from typing import Optional, Sequence

from . import artifacts, exports, insitu_import, manifest, pseudo, quipu_outbox, split
from .paths import Layout
from .promote import (PromotionConfig, decide, evaluate_candidate, mark_decided,
                      pending_reports)
from .tasks import TASKS, get_task


def log(layout: Layout, event: str, **fields) -> None:
    layout.logs.mkdir(parents=True, exist_ok=True)
    rec = {"at": _dt.datetime.now().isoformat(timespec="seconds"), "event": event, **fields}
    with open(layout.logs / "trainer.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")


def day(layout: Layout, predictors: Optional[Sequence] = None,
        split_cfg: split.SplitConfig = split.SplitConfig()) -> dict:
    con = manifest.connect(layout)
    try:
        diff = manifest.refresh(layout, con)
        inbox = insitu_import.import_inbox(con, layout)
        splits = split.assign(con, split_cfg)
        leaks = split.leakage_report(con, split_cfg)
        if leaks:
            raise RuntimeError(f"split leakage after assign: {len(leaks)} components")
        n_pseudo = pseudo.pseudo_label(con, layout, predictors) if predictors else 0
        queue = pseudo.build_queue(con, layout)
        queue += insitu_import.public_feedback_queue(os.environ.get("BAKUGO_APP_DB"))
        queue_path = pseudo.export_queue(layout, queue)
        proposal = quipu_outbox.propose_number_priors(con, layout)
        rec_path = exports.recognition_manifest(con, layout)
        price_path = exports.price_manifest(con, layout)
        out = {"new": len(diff["new"]), "missing": len(diff["missing"]),
               "inbox": inbox,
               "raster_errors": diff["raster_errors"], "splits": splits,
               "pseudo_labelled": n_pseudo, "queued": len(queue),
               "queue": str(queue_path), "quipu_proposal": str(proposal) if proposal else None,
               "recognition_manifest": str(rec_path), "price_manifest": str(price_path)}
        log(layout, "day", **out)
        return out
    finally:
        con.close()


def night(layout: Layout, tasks: Optional[Sequence] = None,
          cfg: PromotionConfig = PromotionConfig()) -> list[dict]:
    tasks = list(tasks) if tasks is not None else [get_task(n) for n in TASKS]
    con = manifest.connect(layout)
    results = []
    try:
        for task in tasks:
            params, train_shas = task.fit(con, layout)
            champ = artifacts.champion(layout, task.name)
            current = champ.params if champ else task.default_params()
            note = getattr(task, "last_fit_note", "")
            if params is None:
                results.append({"task": task.name,
                                "skipped": note or "no confirmed training data"})
                continue
            if params == current:
                results.append({"task": task.name, "skipped": "fit reproduced the champion"})
                continue
            cand = artifacts.write_candidate(layout, task=task.name, exposure=task.exposure,
                                             params=params, train_shas=train_shas)
            report = evaluate_candidate(con, layout, task, cand, cfg)
            results.append({k: report.get(k) for k in
                             ("task", "candidate", "champion", "promote", "reason",
                              "mean_diff", "ci", "n_test", "report_path")})
        log(layout, "night", results=results)
        return results
    finally:
        con.close()


def dawn(layout: Layout) -> list[dict]:
    con = manifest.connect(layout)
    decisions = []
    try:
        seen_tasks = set()
        # newest report per task decides; older pending ones are stale
        for report in sorted(pending_reports(layout), key=lambda r: r.get("evaluated_at", ""),
                             reverse=True):
            task = report["task"]
            champ = artifacts.champion(layout, task)
            champ_id = champ.model_id if champ else f"{task}-code-default"
            if task in seen_tasks or report.get("champion") != champ_id:
                report["promote"] = False
                report["reason"] = "stale: superseded or champion changed since evaluation"
                promoted = decide(con, layout, report)
            else:
                promoted = decide(con, layout, report)
                seen_tasks.add(task)
            mark_decided(report, promoted)
            decisions.append({"task": task, "candidate": report["candidate"],
                              "promoted": promoted, "reason": report.get("reason")})
        log(layout, "dawn", decisions=decisions)
        return decisions
    finally:
        con.close()


def status(layout: Layout) -> dict:
    con = manifest.connect(layout)
    try:
        q = lambda sql: con.execute(sql).fetchall()
        return {
            "root": str(layout.root),
            "files": dict(q("SELECT kind, count(*) FROM files GROUP BY kind")),
            "splits": {str(k): v for k, v in q(
                "SELECT split, count(*) FROM files WHERE kind <> 'pdf' GROUP BY split")},
            "labels": {str(k): v for k, v in q(
                "SELECT label_strength, count(*) FROM files WHERE kind <> 'pdf' GROUP BY label_strength")},
            "frozen_test": q("SELECT count(*) FROM frozen_test")[0][0],
            "quad_labels": q("SELECT count(*) FROM quad_labels")[0][0],
            "test_uses": dict(q("SELECT task, count(*) FROM test_uses GROUP BY task")),
            "champions": {n: (c.model_id if (c := artifacts.champion(layout, n)) else "code default")
                          for n in TASKS},
            "last_decisions": [dict(zip(("task", "candidate", "promoted", "reason", "at"), r))
                               for r in q("SELECT task, candidate_id, promoted, reason, decided_at "
                                          "FROM promotions ORDER BY seq DESC LIMIT 5")],
        }
    finally:
        con.close()
