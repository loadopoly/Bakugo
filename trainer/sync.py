"""Step 1 of the loop: mirror the Drive folder with rclone, read-only.

Why ``rclone copy`` and not ``rclone sync``: sync deletes local files that were
removed from Drive. A frozen test item that disappears breaks every later
comparison against the champion. Copy never deletes; the manifest marks files
that are no longer listed as ``present = false`` instead.

The remote must be configured with ``scope = drive.readonly``. The wrapper
reads the remote's config and refuses any other scope, so a token minted with
full ``drive`` scope is caught before it is used. The config file (which holds
the OAuth token) must be outside the repo and, on POSIX, mode 600.

This is the only step that needs network. It runs in the ``trainer-sync``
compose service; the ``trainer`` service that does everything else has
``network_mode: none``.
"""

from __future__ import annotations

import configparser
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from .paths import Layout, check_secret_file

READONLY_SCOPE = "drive.readonly"


class ScopeError(RuntimeError):
    pass


@dataclass(frozen=True)
class SyncConfig:
    remote: str            # rclone remote name, e.g. "bakugo-drive"
    folder: str            # path inside the remote, e.g. "Card Captures"
    config_file: Path      # rclone.conf (holds the token)
    rclone: str = "rclone"

    @classmethod
    def from_env(cls) -> "SyncConfig":
        missing = [k for k in ("BAKUGO_RCLONE_REMOTE", "BAKUGO_DRIVE_FOLDER", "RCLONE_CONFIG")
                   if not os.environ.get(k)]
        if missing:
            raise RuntimeError(f"sync needs {', '.join(missing)} set")
        return cls(
            remote=os.environ["BAKUGO_RCLONE_REMOTE"],
            folder=os.environ["BAKUGO_DRIVE_FOLDER"],
            config_file=Path(os.environ["RCLONE_CONFIG"]),
            rclone=os.environ.get("BAKUGO_RCLONE_BIN", "rclone"),
        )


def check_remote_scope(config_file: Path, remote: str) -> None:
    """Refuse unless ``[remote]`` is a drive remote with drive.readonly scope."""
    cp = configparser.ConfigParser(interpolation=None)
    # rclone may encrypt its config; an encrypted file cannot be inspected
    # here, so it is refused rather than trusted.
    text = Path(config_file).read_text(encoding="utf-8", errors="replace")
    if "RCLONE_ENCRYPT_V0" in text:
        raise ScopeError(
            "rclone config is encrypted; the scope cannot be verified. "
            "Use an unencrypted config protected by file permissions, or "
            "set BAKUGO_SKIP_SCOPE_CHECK=1 after verifying the scope yourself."
        )
    cp.read_string(text)
    if remote not in cp:
        raise ScopeError(f"remote [{remote}] not found in {config_file}")
    section = cp[remote]
    if section.get("type", "").strip() != "drive":
        raise ScopeError(f"remote [{remote}] is type {section.get('type')!r}, not 'drive'")
    scope = section.get("scope", "").strip()
    if scope != READONLY_SCOPE:
        raise ScopeError(
            f"remote [{remote}] has scope {scope or '(unset, which rclone treats as full drive)'}; "
            f"reconfigure with: rclone config update {remote} scope {READONLY_SCOPE} "
            "and re-authorise so the token is minted with that scope"
        )


def build_commands(cfg: SyncConfig, layout: Layout) -> tuple[list[str], list[str]]:
    src = f"{cfg.remote}:{cfg.folder}"
    common = ["--config", str(cfg.config_file), f"--drive-scope={READONLY_SCOPE}"]
    copy = [cfg.rclone, "copy", src, str(layout.mirror), *common,
            "--checksum", "--log-level", "INFO"]
    # lsjson gives the Drive file ID for each path, which the manifest keys on.
    listing = [cfg.rclone, "lsjson", src, *common, "--recursive", "--files-only", "--hash"]
    return copy, listing


def run_sync(cfg: SyncConfig, layout: Layout, runner=subprocess.run) -> dict:
    check_secret_file(cfg.config_file, "RCLONE_CONFIG")
    if os.environ.get("BAKUGO_SKIP_SCOPE_CHECK") != "1":
        check_remote_scope(cfg.config_file, cfg.remote)
    if runner is subprocess.run and shutil.which(cfg.rclone) is None:
        raise FileNotFoundError(f"rclone binary {cfg.rclone!r} not on PATH")
    layout.ensure()
    copy, listing = build_commands(cfg, layout)
    runner(copy, check=True)
    out = runner(listing, check=True, capture_output=True, text=True)
    entries = json.loads(out.stdout or "[]")
    tmp = layout.listing.with_suffix(".tmp")
    tmp.write_text(json.dumps(entries, indent=1), encoding="utf-8")
    tmp.replace(layout.listing)
    return {"listed": len(entries), "mirror": str(layout.mirror)}


def load_listing(layout: Layout) -> dict[str, dict]:
    """Map mirror-relative POSIX path -> lsjson entry (ID, Hashes, ModTime)."""
    if not layout.listing.exists():
        return {}
    entries: Sequence[dict] = json.loads(layout.listing.read_text(encoding="utf-8"))
    return {e["Path"].replace("\\", "/"): e for e in entries if "Path" in e}
