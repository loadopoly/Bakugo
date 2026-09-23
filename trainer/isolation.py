"""Runtime isolation for the trainer process.

* ``disable_side_channels`` turns off the QUIPU Observer mesh and the Supabase
  mirror for this process, so trainer outputs cannot leak through
  ``cardcenter.quipu_client`` or ``cardcenter.cloud``.
* ``assert_offline`` fails if an outbound TCP connection succeeds. The
  ``trainer`` compose service runs with ``network_mode: none``; this check
  catches a run that was started some other way.
* ``assert_no_realisation_credentials`` refuses to run while the process can
  see QUIPU's attestation key or realisation grant (Invariance #7): the agent
  may propose, it may not hold what would let it approve.
"""

from __future__ import annotations

import os
import socket
import sys
from pathlib import Path

SIDE_CHANNEL_VARS = (
    "CARDCENTER_QUIPU_URL", "QUIPU_URL",
    "CARDCENTER_SUPABASE_URL", "CARDCENTER_SUPABASE_ANON_KEY",
    "VITE_SUPABASE_URL", "VITE_SUPABASE_ANON_KEY",
    "SUPABASE_URL", "SUPABASE_ANON_KEY",
    "SUPABASE_SERVICE_KEY", "SUPABASE_SERVICE_ROLE_KEY", "SERVICE_ROLE_KEY",
)
REALISATION_VARS = ("QUIPU_ATTEST_KEY_FILE", "QUIPU_REALISE_GRANT_REF")
PROBES = (("1.1.1.1", 443), ("8.8.8.8", 53), ("140.82.112.3", 443))


class IsolationError(RuntimeError):
    pass


def disable_side_channels(env=os.environ) -> list[str]:
    removed = [k for k in SIDE_CHANNEL_VARS if k in env]
    for k in removed:
        env.pop(k, None)
    env["CARDCENTER_QUIPU_DISABLE"] = "1"
    return removed


def assert_offline(probes=PROBES, timeout: float = 1.5) -> None:
    for host, port in probes:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                pass
        except OSError:
            continue
        raise IsolationError(
            f"outbound connection to {host}:{port} succeeded; the trainer must run "
            "without network (docker compose --profile trainer run --rm trainer ...)"
        )


def _default_key_paths() -> list[Path]:
    paths = []
    if sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA")
        if base:
            paths.append(Path(base) / "QUIPU" / "attest.key")
    return paths


def assert_no_realisation_credentials(env=os.environ, key_paths=None) -> None:
    held = [k for k in REALISATION_VARS if env.get(k, "").strip()]
    if env.get("QUIPU_ACCEPT_SELF_ASSERTED", "0").strip() not in ("", "0"):
        held.append("QUIPU_ACCEPT_SELF_ASSERTED")
    for p in (key_paths if key_paths is not None else _default_key_paths()):
        try:
            with open(p, "rb"):
                held.append(str(p))
        except OSError:
            pass
    if held:
        raise IsolationError(
            "the trainer can see QUIPU realisation credentials "
            f"({', '.join(held)}). It writes proposals only; run it where those are "
            "absent (the trainer container) so it cannot approve its own changes."
        )


def enter_trainer_mode(require_offline: bool) -> dict:
    assert_no_realisation_credentials()
    removed = disable_side_channels()
    if require_offline:
        assert_offline()
    return {"side_channels_removed": removed, "offline_checked": require_offline}
