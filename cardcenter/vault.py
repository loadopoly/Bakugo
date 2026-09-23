"""Location of the local DuckDB vault (``supabase_vault.duckdb``).

The vault is the local analytical database under development: the Supabase
transfer writes its ``main`` schema, and the private trainer keeps its
manifest, labels and price attributions in a separate ``trainer`` schema of
the same file.

Because the trainer schema holds private collection data, the file must never
be committed or sent to a Docker build. ``vault_ignore_problems`` checks that
for a vault that sits inside the repository.

Resolution order: ``BAKUGO_VAULT_DB``, then ``supabase_vault.duckdb`` in the
current directory (the historical default).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional

DEFAULT_NAME = "supabase_vault.duckdb"
TRAINER_SCHEMA = "trainer"


def vault_path(explicit: Optional[str] = None) -> str:
    return explicit or os.environ.get("BAKUGO_VAULT_DB") or DEFAULT_NAME


def _dockerignore_patterns(repo_root: Path) -> list[str]:
    f = repo_root / ".dockerignore"
    if not f.exists():
        return []
    out = []
    for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def _pattern_regex(pat: str):
    """Docker's matcher: ``*`` and ``?`` stay inside one path segment, ``**``
    spans segments, and a pattern also excludes everything below a matched
    directory."""
    import re

    out, i = [], 0
    while i < len(pat):
        c = pat[i]
        if pat.startswith("**", i):
            out.append(".*")
            i += 2
            if i < len(pat) and pat[i] == "/":
                i += 1
            continue
        out.append({"*": "[^/]*", "?": "[^/]"}.get(c, re.escape(c)))
        i += 1
    return re.compile("^" + "".join(out) + "(/.*)?$")


def _docker_ignored(rel: str, patterns: list[str]) -> bool:
    ignored = False
    for pat in patterns:
        negate = pat.startswith("!")
        p = (pat[1:] if negate else pat).strip().lstrip("/").rstrip("/")
        if p and _pattern_regex(p).match(rel):
            ignored = not negate
    return ignored


def _git_ignored(rel: str, repo_root: Path) -> Optional[bool]:
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    try:
        r = subprocess.run(["git", "-C", str(repo_root), "check-ignore", "-q", "--", rel],
                           capture_output=True, env=env, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode in (0, 1):
        return r.returncode == 0
    return None


def vault_ignore_problems(path: str, repo_root: Path) -> list[str]:
    """Empty when the vault is outside ``repo_root`` or ignored by both git
    and Docker. Each entry is one reason it is not safe."""
    p = Path(path).expanduser().resolve()
    root = repo_root.resolve()
    try:
        rel_path = p.relative_to(root)
    except ValueError:
        return []
    rel = rel_path.as_posix()
    problems = []
    for suffix in ("", ".wal"):
        name = rel + suffix
        if not _docker_ignored(name, _dockerignore_patterns(root)):
            problems.append(f"{name} is not excluded by .dockerignore")
        git = _git_ignored(name, root)
        if git is False:
            problems.append(f"{name} is not ignored by git")
    return problems
