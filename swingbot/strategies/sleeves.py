"""Strategy sleeves. Each returns daily target weights (rows = decision dates, columns = assets).

Contract (enforced by tests/test_lookahead.py): row t uses only data with index <= t, and the
weights in each row are >= 0 and sum to <= 1 (long-only, no leverage).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from swingbot.signals.indicators import inverse_vol_weights, realized_vol, rsi, sma, total_return


# ------------------------------------------------------------------ 1. Trend (time-series momentum)
@dataclass(frozen=True)
class TrendParams:
    lookbacks: tuple = (20, 60, 120, 250)
    signal: str = "return_sign"     # "return_sign": past L-day return > 0; "above_sma": close > SMA(L)
    vol_lookback: int = 60


def trend_sleeve(panel, p: TrendParams, ppy: int, min_history: int) -> pd.DataFrame:
    """Each asset's inverse-vol slot is filled in proportion to how many lookbacks say 'uptrend'."""
    close = panel.close
    elig = panel.eligible(max(min_history, max(p.lookbacks) + 1))
    votes = []
    for L in p.lookbacks:
        v = total_return(close, L) > 0 if p.signal == "return_sign" else close > sma(close, L)
        votes.append(v.astype(float))
    score = sum(votes) / len(votes)
    ivw = inverse_vol_weights(realized_vol(close, p.vol_lookback, ppy), elig)
    return (score * ivw).where(elig, 0.0).fillna(0.0)


# ------------------------------------------------------------------ 2. Momentum rotation (cross-sectional)
@dataclass(frozen=True)
class RotationParams:
    lookbacks: tuple = (30, 60, 90)
    top_n: int = 2
    rebalance_days: int = 7
    vol_lookback: int = 60


def rotation_sleeve(panel, p: RotationParams, ppy: int, min_history: int) -> pd.DataFrame:
    """Rank by risk-adjusted momentum; hold the top N whose own momentum is positive; rest in cash."""
    close = panel.close
    elig = panel.eligible(max(min_history, max(p.lookbacks) + 1))
    vol = realized_vol(close, p.vol_lookback, ppy)
    dvol = vol / np.sqrt(ppy)
    score = sum(total_return(close, L) / (dvol * np.sqrt(L)) for L in p.lookbacks) / len(p.lookbacks)
    raw = sum(total_return(close, L) for L in p.lookbacks) / len(p.lookbacks)
    ok = elig & (raw > 0) & score.notna()
    ranks = score.where(ok).rank(axis=1, ascending=False, method="first")
    chosen = ranks <= p.top_n
    k = chosen.sum(axis=1)
    w = inverse_vol_weights(vol, chosen).mul(k / p.top_n, axis=0)
    days = (close.index - pd.Timestamp("1970-01-01", tz="UTC")).days
    is_rebal = np.asarray(days % p.rebalance_days == 0)
    w.loc[~is_rebal] = np.nan
    return w.ffill().fillna(0.0)


# ------------------------------------------------------------------ 3. Dip-buying inside uptrends
@dataclass(frozen=True)
class DipParams:
    rsi_len: int = 2
    entry: float = 10.0
    exit_sma: int = 5
    trend_sma: int = 200
    time_stop: int = 10
    vol_lookback: int = 60


def dip_sleeve(panel, p: DipParams, ppy: int, min_history: int) -> pd.DataFrame:
    """Buy when short-term RSI is oversold and price is above its long SMA; exit on close > SMA(exit) or time stop."""
    close = panel.close
    elig = panel.eligible(max(min_history, p.trend_sma))
    r = rsi(close, p.rsi_len).to_numpy()
    up = (close > sma(close, p.trend_sma)).to_numpy()
    bounce = (close > sma(close, p.exit_sma)).to_numpy()
    el = elig.to_numpy()
    N, n = close.shape
    inpos = np.zeros((N, n), bool)
    state = np.zeros(n, bool)
    held = np.zeros(n, int)
    for t in range(N):
        held[state] += 1
        exit_ = state & (bounce[t] | (held >= p.time_stop) | ~el[t])
        state &= ~exit_
        enter = ~state & el[t] & up[t] & (np.nan_to_num(r[t], nan=100.0) < p.entry)
        state |= enter
        held[enter] = 0
        inpos[t] = state
    ivw = inverse_vol_weights(realized_vol(close, p.vol_lookback, ppy), elig)
    return ivw.where(pd.DataFrame(inpos, index=close.index, columns=close.columns), 0.0)
