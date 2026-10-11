"""Benchmarks every result must beat. They run through the same simulator and costs as strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd

from swingbot.signals.indicators import sma


def month_starts(index: pd.DatetimeIndex) -> np.ndarray:
    m = index.tz_localize(None).to_period("M") if index.tz is not None else index.to_period("M")
    return np.r_[True, m[1:] != m[:-1]]


def buy_and_hold(panel, weights: dict[str, float], rebalance: str = "monthly") -> pd.DataFrame:
    """Fixed-weight portfolio, renormalized over assets that have a price that day.
    rebalance='monthly': act on the first day of each month and on the day a new asset first trades;
    other rows are NaN (hold). rebalance='never': act only when the asset set changes."""
    close = panel.close[list(weights)]
    w = pd.DataFrame({a: float(x) for a, x in weights.items()}, index=close.index).where(close.notna(), 0.0)
    w = w.div(w.sum(axis=1).replace(0, np.nan), axis=0)
    appear = (close.notna() & close.shift(1).isna()).any(axis=1).to_numpy()
    act = appear | (month_starts(close.index) if rebalance == "monthly" else False)
    w.loc[~act] = np.nan
    return w


def sma_trend(panel, asset: str = "BTC", n: int = 200) -> pd.DataFrame:
    """Long `asset` when close > SMA(n), else cash. The classic simple baseline."""
    c = panel.close[[asset]]
    s = sma(c, n)
    return (c > s).astype(float).where(s.notna(), 0.0)


def equal_weight(panel, eligible: pd.DataFrame) -> pd.DataFrame:
    """Equal weight across eligible assets, rebalanced monthly."""
    w = eligible.astype(float)
    w = w.div(w.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    w.loc[~month_starts(w.index)] = np.nan
    return w
