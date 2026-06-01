"""Gurobi environment setup, including Web License Service (WLS) credentials.

The MILP needs a licensed Gurobi environment. On machines with a local licence
file the default ``gp.Env()`` just works; for the cloud/academic WLS licence the
credentials ``WLSACCESSID``, ``WLSSECRET`` and ``LICENSEID`` must be supplied.

We read those from the process environment, falling back to a ``.env`` file in
the project root so a developer only has to drop the credentials in one place
(see ``.env`` -- it is git-ignored).
"""

from __future__ import annotations

import os
from pathlib import Path

import gurobipy as gp

_WLS_KEYS = ("WLSACCESSID", "WLSSECRET", "LICENSEID")


def _find_dotenv() -> Path | None:
    """Walk up from this file looking for a ``.env`` (project root)."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / ".env"
        if candidate.is_file():
            return candidate
    return None


def _read_dotenv(path: Path) -> dict[str, str]:
    """Parse a minimal ``KEY=value`` ``.env`` file (no interpolation, no export)."""
    values: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip("'\"")
    return values


def wls_credentials() -> dict[str, str] | None:
    """Return WLS credentials from the environment or ``.env``, or None if absent.

    Process environment wins over the ``.env`` file. Returns None unless all three
    keys are present, so callers can fall back to a local licence.
    """
    creds = {k: os.environ[k] for k in _WLS_KEYS if k in os.environ}
    if len(creds) < len(_WLS_KEYS):
        dotenv = _find_dotenv()
        if dotenv is not None:
            from_file = _read_dotenv(dotenv)
            for k in _WLS_KEYS:
                creds.setdefault(k, from_file.get(k, ""))
    if any(not creds.get(k) for k in _WLS_KEYS):
        return None
    return creds


def make_env(*, verbose: bool = False) -> gp.Env:
    """Build and start a Gurobi environment.

    Uses WLS credentials when available (env or ``.env``); otherwise starts a
    default environment that relies on a locally installed licence. The returned
    env is already started -- the caller owns it and should ``dispose()`` it (or
    use it as a context manager) when done.
    """
    creds = wls_credentials()
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 1 if verbose else 0)
    if creds is not None:
        env.setParam("WLSACCESSID", creds["WLSACCESSID"])
        env.setParam("WLSSECRET", creds["WLSSECRET"])
        env.setParam("LICENSEID", int(creds["LICENSEID"]))
    env.start()
    return env
