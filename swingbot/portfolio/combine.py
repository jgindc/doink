"""Combine sleeves into one book: fixed risk budgets -> per-asset caps -> volatility target (scale down only)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from swingbot.config import per_asset
from swingbot.signals.indicators import log_returns


def combine(sleeves: dict[str, pd.DataFrame], budgets: dict[str, float]) -> pd.DataFrame:
    cols = sorted(set().union(*(s.columns for s in sleeves.values())))
    out = None
    for name, w in sleeves.items():
        part = w.reindex(columns=cols).fillna(0.0) * budgets[name]
        out = part if out is None else out.add(part, fill_value=0.0)
    return out


def apply_caps(w: pd.DataFrame, max_weight: dict) -> pd.DataFrame:
    caps = pd.Series({a: per_asset(max_weight, a) for a in w.columns})
    return w.clip(upper=caps, axis=1)


def ex_ante_vol(w: pd.DataFrame, close: pd.DataFrame, lookback: int, ppy: int) -> pd.Series:
    """Portfolio vol implied by today's weights and the trailing `lookback`-day covariance (causal)."""
    R = log_returns(close[w.columns]).to_numpy()
    W = w.to_numpy()
    out = np.full(len(w), np.nan)
    for t in range(lookback, len(w)):
        active = W[t] > 0
        if not active.any():
            out[t] = 0.0
            continue
        win = R[t - lookback + 1:t + 1][:, active]
        win = win[~np.isnan(win).any(axis=1)]
        if len(win) < lookback // 2:
            continue
        cov = np.atleast_2d(np.cov(win, rowvar=False))
        out[t] = float(np.sqrt(max(W[t, active] @ cov @ W[t, active], 0.0) * ppy))
    return pd.Series(out, index=w.index)


def vol_target(w: pd.DataFrame, close: pd.DataFrame, target: float, lookback: int, ppy: int) -> pd.DataFrame:
    """Scale gross exposure down when ex-ante vol exceeds target. Never levers up."""
    vol = ex_ante_vol(w, close, lookback, ppy)
    scale = (target / vol).clip(upper=1.0).fillna(1.0).where(vol > 0, 1.0)
    return w.mul(scale, axis=0)


def build_book(sleeves: dict[str, pd.DataFrame], cfg: dict, close: pd.DataFrame) -> pd.DataFrame:
    p = cfg["portfolio"]
    w = combine(sleeves, p["sleeve_budget"])
    w = apply_caps(w, p["max_weight"])
    return vol_target(w, close, p["vol_target_annual"], p["vol_lookback_days"], cfg["annualization_days"])
