"""Random strategies for luck calibration. They use the same universe, sizing, simulator and costs as the
real sleeves but have no information, so anything they 'achieve' is luck (or market beta)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from swingbot.signals.indicators import inverse_vol_weights, realized_vol


def random_timing(panel, rng: np.random.Generator, ppy: int, min_history: int,
                  mean_hold: float, p_in: float) -> pd.DataFrame:
    """Each asset flips in/out of the market as a random two-state Markov chain with the given
    mean holding period and long-run time-in-market. Sizing = inverse vol (like the trend sleeve)."""
    elig = panel.eligible(min_history)
    N, n = elig.shape
    p_exit = 1.0 / mean_hold
    p_enter = min(p_exit * p_in / max(1 - p_in, 1e-9), 1.0)
    u = rng.random((N, n))
    state = np.zeros((N, n), bool)
    s = rng.random(n) < p_in
    for t in range(N):
        s = np.where(s, u[t] >= p_exit, u[t] < p_enter)
        state[t] = s
    state &= elig.to_numpy()
    ivw = inverse_vol_weights(realized_vol(panel.close, 60, ppy), elig)
    return ivw.where(pd.DataFrame(state, index=elig.index, columns=elig.columns), 0.0)


def random_rotation(panel, rng: np.random.Generator, ppy: int, min_history: int, top_n: int,
                    rebalance_days: int = 7) -> pd.DataFrame:
    """Every rebalance day hold `top_n` randomly chosen eligible assets (inverse-vol weighted)."""
    elig = panel.eligible(min_history)
    E = elig.to_numpy()
    scores = np.where(E, rng.random(E.shape), -1.0)
    order = np.argsort(-scores, axis=1)
    chosen = np.zeros_like(E)
    rows = np.arange(E.shape[0])[:, None]
    chosen[rows, order[:, :top_n]] = True
    chosen &= E
    chosen_df = pd.DataFrame(chosen, index=elig.index, columns=elig.columns)
    vol = realized_vol(panel.close, 60, ppy)
    w = inverse_vol_weights(vol, chosen_df).mul(chosen_df.sum(axis=1) / top_n, axis=0)
    days = (elig.index - pd.Timestamp("1970-01-01", tz="UTC")).days
    w.loc[~np.asarray(days % rebalance_days == 0)] = np.nan
    return w.ffill().fillna(0.0)
