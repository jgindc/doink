"""Experiment registry: pre-registration enforcement and the trial ledger."""

from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

from swingbot.config import EXPERIMENTS_DIR, ROOT

REGISTRY = EXPERIMENTS_DIR / "registry"
RESULTS = EXPERIMENTS_DIR / "results"
TRIALS = EXPERIMENTS_DIR / "trials.csv"
TRIAL_FIELDS = ["logged_at", "exp_id", "config_id", "params", "oos_sharpe", "oos_cagr", "oos_max_dd",
                "dataset_version", "git_commit"]


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()


def load_entry(exp_id: str) -> tuple[Path, dict]:
    matches = sorted(REGISTRY.glob(f"{exp_id}-*.yaml"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected exactly one registry file for {exp_id}, found {matches}")
    return matches[0], yaml.safe_load(matches[0].read_text())


def assert_preregistered(path: Path) -> str:
    """Refuse to run unless the registry file is committed and unmodified. Returns the commit that registered it."""
    rel = str(path.relative_to(ROOT))
    if not _git("ls-files", rel):
        raise PermissionError(f"{rel} is not committed — pre-register (commit) before running.")
    if _git("status", "--porcelain", "--", rel):
        raise PermissionError(f"{rel} has uncommitted changes — commit the registration before running.")
    return _git("log", "-1", "--format=%h %cI", "--", rel)


def head_commit() -> str:
    dirty = "+dirty" if _git("status", "--porcelain", "--", "swingbot", "configs") else ""
    return _git("rev-parse", "--short", "HEAD") + dirty


def log_trials(rows: list[dict]) -> None:
    new = not TRIALS.exists()
    with TRIALS.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRIAL_FIELDS)
        if new:
            w.writeheader()
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for r in rows:
            r = {**r, "logged_at": stamp, "params": json.dumps(r.get("params", {}), sort_keys=True)}
            w.writerow({k: r.get(k, "") for k in TRIAL_FIELDS})


def results_dir(exp_id: str) -> Path:
    d = RESULTS / exp_id
    d.mkdir(parents=True, exist_ok=True)
    return d
