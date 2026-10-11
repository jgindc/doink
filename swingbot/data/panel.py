"""Daily price panel with a hard guard around the locked holdout period.

purpose="research": data stops the day before holdout_start. This is the default.
purpose="holdout" : full history; requires a written reason and appends to experiments/holdout_log.md.
purpose="live"    : full history, for paper/live signal generation only (never for evaluating strategies).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from swingbot.config import EXPERIMENTS_DIR
from swingbot.data import store

HOLDOUT_LOG = EXPERIMENTS_DIR / "holdout_log.md"


@dataclass
class Panel:
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    filled: pd.DataFrame  # True where a missing bar was forward-filled (no real trading data)

    @property
    def assets(self) -> list[str]:
        return list(self.close.columns)

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.close.index

    def truncate(self, end) -> "Panel":
        """Data up to and including `end`. Used by lookahead tests and by live signal generation."""
        return Panel(*(getattr(self, f).loc[:end] for f in ("open", "high", "low", "close", "volume", "filled")))

    def subset(self, assets: list[str]) -> "Panel":
        return Panel(*(getattr(self, f)[assets] for f in ("open", "high", "low", "close", "volume", "filled")))

    def eligible(self, min_history: int) -> pd.DataFrame:
        """True once an asset has at least `min_history` daily bars (inclusive of today)."""
        return self.close.notna().cumsum() >= min_history


def _fill_gaps(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    full = pd.date_range(df.index[0], df.index[-1], freq="1D")
    out = df.reindex(full)
    filled = out["close"].isna()
    out["close"] = out["close"].ffill()
    for c in ("open", "high", "low"):
        out[c] = out[c].fillna(out["close"])
    out["volume"] = out["volume"].fillna(0.0)
    return out, filled


def load_crypto_panel(cfg: dict, universe: list[str] | None = None, *, purpose: str = "research",
                      holdout_reason: str | None = None, source: str | None = None) -> Panel:
    universe = universe or cfg["data"]["universe"]
    source = source or cfg["data"]["source"]
    frames, fills = {}, {}
    for sym in universe:
        df, filled = _fill_gaps(store.read_bars(source, sym, "1d"))
        frames[sym] = df
        fills[sym] = filled
    idx = sorted(set().union(*(f.index for f in frames.values())))
    idx = pd.DatetimeIndex(idx, tz="UTC")
    fields = {c: pd.DataFrame({s: frames[s][c] for s in universe}).reindex(idx) for c in
              ("open", "high", "low", "close", "volume")}
    filled = pd.DataFrame({s: fills[s] for s in universe}).reindex(idx).fillna(False).astype(bool)
    panel = Panel(**fields, filled=filled)

    holdout_start = pd.Timestamp(cfg["splits"]["holdout_start"], tz="UTC")
    if purpose == "research":
        return panel.truncate(holdout_start - pd.Timedelta(days=1))
    if purpose == "holdout":
        if not holdout_reason:
            raise PermissionError("Opening the locked holdout requires a written reason (it is logged).")
        log_holdout_opening(holdout_reason)
        return panel
    if purpose == "live":
        return panel
    raise ValueError(f"unknown purpose {purpose!r}")


def log_holdout_opening(reason: str) -> None:
    HOLDOUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    if not HOLDOUT_LOG.exists():
        HOLDOUT_LOG.write_text("# Holdout openings\n\nEvery access to data on/after `holdout_start` is logged here.\n\n")
    with HOLDOUT_LOG.open("a") as f:
        f.write(f"- {datetime.now(timezone.utc).isoformat(timespec='seconds')} — {reason}\n")
