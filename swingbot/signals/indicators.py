"""Causal indicators. Every value at row t uses only rows <= t (tested in tests/test_lookahead.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(close: pd.DataFrame) -> pd.DataFrame:
    return np.log(close).diff()


def sma(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(n, min_periods=n).mean()


def total_return(close: pd.DataFrame, n: int) -> pd.DataFrame:
    return close / close.shift(n) - 1.0


def realized_vol(close: pd.DataFrame, n: int, periods_per_year: int) -> pd.DataFrame:
    """Annualized std of daily log returns over the last n days."""
    return log_returns(close).rolling(n, min_periods=max(10, n // 2)).std() * np.sqrt(periods_per_year)


def rsi(close: pd.DataFrame, n: int) -> pd.DataFrame:
    """Wilder's RSI (exponential smoothing with alpha = 1/n)."""
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs = up / dn
    out = 100 - 100 / (1 + rs)
    return out.where(dn > 0, 100.0).where(up.notna())


def atr(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, n: int) -> pd.DataFrame:
    prev = close.shift(1)
    tr = np.maximum(high - low, np.maximum((high - prev).abs(), (low - prev).abs()))
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def inverse_vol_weights(vol: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    """Inverse-volatility weights over assets where mask is True; rows sum to 1 (or 0 if none)."""
    iv = (1.0 / vol).where(mask & vol.gt(0))
    return iv.div(iv.sum(axis=1), axis=0).fillna(0.0)
