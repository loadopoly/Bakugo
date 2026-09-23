"""Candidate and champion artifacts in the private tier, and the public copy
the release step writes.

Private artifact (``<private>/models/<task>/<model_id>/``):
  artifact.json  -- full lineage, including the SHA-256 of every training item
  params.json    -- the payload

Public artifact (``cardcenter/data/released/<task>/``):
  artifact.json  -- the same record with the per-item list replaced by its
                    digest and count (the hashes are of private photos)
  params.json

Champion pointer: ``<private>/models/<task>/champion.json``.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from cardcenter.release_guard import SCHEMA, check_artifact, ReleaseRefused, scan

from .paths import Layout, REPO_ROOT


def _digest(shas: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(shas)).encode()).hexdigest()


def _code_version() -> str:
    try:
        from cardcenter import __version__ as cc
    except Exception:
        cc = "?"
    from . import __version__ as tr
    return f"cardcenter {cc}; trainer {tr}"


@dataclass(frozen=True)
class Artifact:
    task: str
    model_id: str
    path: Path
    meta: dict
    params: dict


def write_candidate(layout: Layout, *, task: str, exposure: str, params: dict,
                    train_shas: Sequence[str], notes: str = "") -> Artifact:
    stamp = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    body = json.dumps(params, sort_keys=True, indent=1)
    model_id = f"{task}-{stamp}-{hashlib.sha256(body.encode()).hexdigest()[:8]}"
    d = layout.models / task / model_id
    d.mkdir(parents=True, exist_ok=False)
    (d / "params.json").write_text(body, encoding="utf-8")
    meta = {
        "schema": SCHEMA,
        "task": task,
        "kind": task,
        "model_id": model_id,
        "exposure": exposure,
        "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "code_version": _code_version(),
        "payload": "params.json",
        "payload_sha256": hashlib.sha256(body.encode()).hexdigest(),
        "lineage": {
            "confirmed_only": True,
            "split": "train",
            "n_train": len(train_shas),
            "train_set_digest": _digest(train_shas),
            "train_sha256": sorted(train_shas),
            "notes": notes,
        },
    }
    (d / "artifact.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return Artifact(task, model_id, d, meta, params)


def load(path: Path) -> Artifact:
    meta = json.loads((path / "artifact.json").read_text(encoding="utf-8"))
    params = json.loads((path / meta["payload"]).read_text(encoding="utf-8"))
    return Artifact(meta["task"], meta["model_id"], path, meta, params)


def champion(layout: Layout, task: str) -> Optional[Artifact]:
    ptr = layout.models / task / "champion.json"
    if not ptr.exists():
        return None
    model_id = json.loads(ptr.read_text(encoding="utf-8"))["model_id"]
    return load(layout.models / task / model_id)


def set_champion(layout: Layout, art: Artifact, evaluation: dict) -> None:
    meta_path = art.path / "artifact.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.setdefault("lineage", {})["promotion_eval"] = evaluation
    meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    ptr = layout.models / art.task / "champion.json"
    tmp = ptr.with_suffix(".tmp")
    tmp.write_text(json.dumps({"model_id": art.model_id,
                               "since": _dt.datetime.now().isoformat(timespec="seconds")}),
                   encoding="utf-8")
    tmp.replace(ptr)


def release(layout: Layout, task: str, package_root: Path = REPO_ROOT / "cardcenter") -> Path:
    """Copy the current champion into the public package, then run the guard.
    Any refusal removes what was written."""
    art = champion(layout, task)
    if art is None:
        raise ReleaseRefused(f"no champion for {task}")
    if art.meta.get("exposure") != "generic":
        raise ReleaseRefused(f"{art.model_id} is {art.meta.get('exposure')}; not releasable")
    public = json.loads(json.dumps(art.meta))
    lineage = public.get("lineage", {})
    lineage.pop("train_sha256", None)
    ev = lineage.get("promotion_eval") or {}
    lineage["promotion_eval"] = {k: ev.get(k) for k in ("mean_diff", "ci", "n_test") if k in ev}
    target = package_root / "data" / "released" / task
    staging = target.with_name(task + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    shutil.copyfile(art.path / art.meta["payload"], staging / art.meta["payload"])
    (staging / "artifact.json").write_text(json.dumps(public, indent=1), encoding="utf-8")
    try:
        check_artifact_dir = staging.with_name(task)
        backup = None
        if check_artifact_dir.exists():
            backup = check_artifact_dir.with_name(task + ".previous")
            if backup.exists():
                shutil.rmtree(backup)
            check_artifact_dir.rename(backup)
        staging.rename(check_artifact_dir)
        try:
            check_artifact(check_artifact_dir)
            problems = scan(package_root)
            if problems:
                raise ReleaseRefused("; ".join(problems))
        except ReleaseRefused:
            shutil.rmtree(check_artifact_dir)
            if backup is not None:
                backup.rename(check_artifact_dir)
            raise
        if backup is not None:
            shutil.rmtree(backup)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return target
