"""Steps 5-6: evaluate a candidate against the champion on the frozen test set
and decide once.

Promotion rule (``PromotionConfig``):

* at least ``min_items`` paired test items;
* overall: the paired-bootstrap ``1 - alpha`` interval of
  mean(candidate - champion) lies entirely above zero ("outside the error
  bars");
* per resolution slice: the candidate's point estimate may not fall below the
  champion's by more than ``max_slice_drop`` (default 0.0, i.e. no regression
  in any slice, as specified), and no slice may show a regression whose own
  interval lies entirely below zero.

With ``max_slice_drop = 0`` a single flipped item in a two-item slice blocks
promotion. That is the literal rule and it is the default; raise the tolerance
deliberately and the decision log will show the value used.

Every decision -- promote or keep -- is appended to ``promotions`` with its
reason and the report path.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

from . import artifacts
from .paths import Layout


@dataclass(frozen=True)
class PromotionConfig:
    min_items: int = 30
    alpha: float = 0.05
    n_boot: int = 4000
    max_slice_drop: float = 0.0
    seed: int = 0


def paired_bootstrap(diffs, n_boot: int, alpha: float, seed: int) -> tuple[float, float, float]:
    d = np.asarray(diffs, dtype=np.float64)
    if d.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(d.mean()), float(lo), float(hi)


def compare(cand_scores, champ_scores, cfg: PromotionConfig) -> dict:
    champ = {s.key: s for s in champ_scores}
    cand = {s.key: s for s in cand_scores}
    keys = sorted(set(champ) & set(cand))
    report: dict = {"n_test": len(keys), "config": asdict(cfg), "slices": {},
                    "unpaired": len(set(champ) ^ set(cand))}
    if len(keys) < cfg.min_items:
        report.update(promote=False, reason=f"only {len(keys)} paired test items (< {cfg.min_items})")
        return report
    diffs = [cand[k].value - champ[k].value for k in keys]
    mean, lo, hi = paired_bootstrap(diffs, cfg.n_boot, cfg.alpha, cfg.seed)
    report.update(mean_diff=round(mean, 4), ci=[round(lo, 4), round(hi, 4)],
                  candidate_acc=round(float(np.mean([cand[k].value for k in keys])), 4),
                  champion_acc=round(float(np.mean([champ[k].value for k in keys])), 4))
    reasons = []
    if not lo > 0.0:
        reasons.append(f"overall gain {mean:+.3f} not outside error bars (CI {lo:+.3f}..{hi:+.3f})")
    by_slice: dict[str, list[str]] = {}
    for k in keys:
        by_slice.setdefault(champ[k].slice, []).append(k)
    for name, ks in sorted(by_slice.items()):
        d = [cand[k].value - champ[k].value for k in ks]
        m, l, h = paired_bootstrap(d, cfg.n_boot, cfg.alpha, cfg.seed)
        report["slices"][name] = {"n": len(ks), "mean_diff": round(m, 4),
                                  "ci": [round(l, 4), round(h, 4)]}
        if m < -cfg.max_slice_drop:
            reasons.append(f"slice {name} regresses {m:+.3f} (n={len(ks)}, tolerance {cfg.max_slice_drop})")
        elif h < 0.0:
            reasons.append(f"slice {name} regression is significant (CI {l:+.3f}..{h:+.3f})")
    report["promote"] = not reasons
    report["reason"] = "; ".join(reasons) if reasons else (
        f"gain {mean:+.3f} (CI {lo:+.3f}..{hi:+.3f}) with no slice regression")
    return report


def evaluate_candidate(con, layout: Layout, task, candidate: artifacts.Artifact,
                       cfg: PromotionConfig = PromotionConfig()) -> dict:
    champ = artifacts.champion(layout, task.name)
    champ_params = champ.params if champ else task.default_params()
    champ_id = champ.model_id if champ else f"{task.name}-code-default"
    champ_scores = task.score(con, layout, champ_params, champ_id)
    cand_scores = task.score(con, layout, candidate.params, candidate.model_id)
    report = compare(cand_scores, champ_scores, cfg)
    report.update(task=task.name, candidate=candidate.model_id, champion=champ_id,
                  candidate_params=candidate.params, champion_params=champ_params,
                  test_uses=con.execute("SELECT count(*) FROM test_uses WHERE task = ?",
                                        [task.name]).fetchone()[0],
                  evaluated_at=_dt.datetime.now().isoformat(timespec="seconds"))
    layout.reports.mkdir(parents=True, exist_ok=True)
    path = layout.reports / f"eval-{candidate.model_id}.json"
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    report["report_path"] = str(path)
    return report


def decide(con, layout: Layout, report: dict) -> bool:
    """The single promotion decision for one evaluated candidate."""
    task = report["task"]
    promoted = bool(report.get("promote"))
    if promoted:
        cand = artifacts.load(layout.models / task / report["candidate"])
        artifacts.set_champion(layout, cand, {k: report.get(k) for k in
                                              ("mean_diff", "ci", "n_test", "slices", "reason")})
    con.execute(
        "INSERT INTO promotions (task, candidate_id, champion_id, promoted, reason, report, decided_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [task, report["candidate"], report.get("champion"), promoted,
         report.get("reason", ""), report.get("report_path"),
         _dt.datetime.now().replace(microsecond=0)],
    )
    return promoted


def pending_reports(layout: Layout) -> list[dict]:
    out = []
    for p in sorted(layout.reports.glob("eval-*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        if not r.get("decided"):
            r["report_path"] = str(p)
            out.append(r)
    return out


def mark_decided(report: dict, promoted: bool) -> None:
    from pathlib import Path

    p = Path(report["report_path"])
    r = json.loads(p.read_text(encoding="utf-8"))
    r.update(decided=True, promoted=promoted,
             decided_at=_dt.datetime.now().isoformat(timespec="seconds"))
    p.write_text(json.dumps(r, indent=1), encoding="utf-8")
