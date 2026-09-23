"""Private data tier layout.

The root comes from ``BAKUGO_PRIVATE_ROOT``. The default is per-user and
outside the repository:

* Windows: ``%LOCALAPPDATA%\\Bakugo\\private``
* elsewhere: ``$XDG_DATA_HOME/bakugo-private`` (``~/.local/share/...``)

Inside a container the compose file mounts the host root at ``/private`` and
sets ``BAKUGO_PRIVATE_ROOT=/private``.

``resolve_layout`` refuses a root that resolves inside the repository, because
the repository is the Docker build context for the public image and is pushed
to a public remote. ``.gitignore``/``.dockerignore`` carry matching patterns as
a second line of defence, not as the control.
"""

from __future__ import annotations

import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent


class UnsafeLocation(RuntimeError):
    """A private path resolved somewhere the public side can read."""


def default_root() -> Path:
    if sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "Bakugo" / "private"
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "bakugo-private"


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def assert_outside_repo(path: Path, what: str, repo_root: Path = REPO_ROOT) -> Path:
    resolved = Path(path).expanduser().resolve()
    repo = repo_root.resolve()
    if _is_within(resolved, repo):
        raise UnsafeLocation(
            f"{what} resolves to {resolved}, inside the repository {repo}. "
            "The repository is the public Docker build context and a public git "
            "remote; put private data and credentials outside it."
        )
    return resolved


@dataclass(frozen=True)
class Layout:
    root: Path
    vault_db: Path = None  # type: ignore[assignment]  # set by resolve_layout

    def __post_init__(self):
        if self.vault_db is None:
            object.__setattr__(self, "vault_db", self.root / "supabase_vault.duckdb")

    @property
    def mirror(self) -> Path:          # rclone copy target (raw Drive files)
        return self.root / "mirror"

    @property
    def raster(self) -> Path:          # rasterised PDF pages, written once
        return self.root / "raster"

    @property
    def inbox(self) -> Path:           # confirmed captures from owner devices (in-situ)
        return self.root / "inbox"

    @property
    def listing(self) -> Path:         # rclone lsjson output (Drive IDs)
        return self.root / "drive_listing.json"

    @property
    def queue(self) -> Path:           # review queue exports
        return self.root / "queue"

    @property
    def models(self) -> Path:          # candidate + champion artifacts
        return self.root / "models"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def quipu_outbox(self) -> Path:    # proposals only; nothing here is realised
        return self.root / "outbox" / "quipu"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    def ensure(self) -> "Layout":
        for p in (self.mirror, self.raster, self.queue, self.models,
                  self.reports, self.quipu_outbox, self.logs, self.inbox):
            p.mkdir(parents=True, exist_ok=True)
        return self


def resolve_vault(vault: Optional[os.PathLike] = None, repo_root: Path = REPO_ROOT) -> Path:
    """The trainer's tables live in the ``trainer`` schema of the local vault
    (``supabase_vault.duckdb``). A vault inside the repository is accepted
    only while git and ``.dockerignore`` both exclude it and its WAL file."""
    from cardcenter.vault import vault_ignore_problems, vault_path

    raw = str(vault) if vault else os.environ.get("BAKUGO_VAULT_DB")
    path = Path(raw).expanduser() if raw else repo_root / vault_path()
    problems = vault_ignore_problems(str(path), repo_root)
    if problems:
        raise UnsafeLocation(
            f"vault {path} holds private trainer data but "
            + "; ".join(problems)
            + ". Add the patterns or move the vault outside the repository."
        )
    return path.resolve()


def resolve_layout(root: Optional[os.PathLike] = None, repo_root: Path = REPO_ROOT,
                   vault: Optional[os.PathLike] = None) -> Layout:
    raw = root or os.environ.get("BAKUGO_PRIVATE_ROOT") or default_root()
    return Layout(assert_outside_repo(Path(raw), "BAKUGO_PRIVATE_ROOT", repo_root),
                  resolve_vault(vault, repo_root))


def check_secret_file(path: os.PathLike, what: str, repo_root: Path = REPO_ROOT) -> Path:
    """A credential file must exist, sit outside the repo, and (on POSIX) be
    unreadable by group/other. On Windows the ACL is the control and cannot be
    read portably from Python; the docs give the ``icacls`` command."""
    p = assert_outside_repo(Path(path), what, repo_root)
    if not p.is_file():
        raise FileNotFoundError(f"{what} not found at {p}")
    # BAKUGO_SECRET_PERMS=acl: the file is bind-mounted from a Windows host,
    # where the mode bits Docker reports are meaningless and the NTFS ACL is
    # the control.
    if os.name == "posix" and os.environ.get("BAKUGO_SECRET_PERMS") != "acl":
        mode = p.stat().st_mode
        if mode & (stat.S_IRWXG | stat.S_IRWXO):
            raise UnsafeLocation(
                f"{what} at {p} has mode {oct(mode & 0o777)}; "
                f"run: chmod 600 '{p}'"
            )
    return p
