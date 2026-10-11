"""Config loading. Configs are plain YAML so every assumption is visible and diffable."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs"
DATA_DIR = ROOT / "data"
EXPERIMENTS_DIR = ROOT / "experiments"


def load_config(name: str = "crypto") -> dict[str, Any]:
    path = CONFIG_DIR / f"{name}.yaml"
    with path.open() as f:
        return yaml.safe_load(f)


def utc(t) -> "pd.Timestamp":
    """Timestamp in UTC (all bar indexes are tz-aware UTC)."""
    import pandas as pd

    t = pd.Timestamp(t)
    return t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")


def per_asset(mapping: dict[str, float], asset: str) -> float:
    """Look up a per-asset value with a 'default' fallback."""
    return float(mapping.get(asset, mapping["default"]))
