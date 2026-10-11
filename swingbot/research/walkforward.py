"""Rolling walk-forward: pick the best configuration on a training window, apply it to the next unseen
window, step forward. The out-of-sample (OOS) pieces are spliced into one return series."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from swingbot.config import utc
from swingbot.research.overfit import sharpe_per_period


@dataclass
class WFResult:
    oos_returns: pd.Series
    windows: pd.DataFrame          # test_start, test_end, chosen config, its train Sharpe
    train_sharpes: pd.DataFrame    # one row per window, one column per config (annualized)


def walk_forward(returns: pd.DataFrame, weights: dict[str, pd.DataFrame] | None, cost_rates: dict[str, float],
                 *, start, end, train_days: int, test_days: int, step_days: int, ppy: int) -> WFResult:
    """returns: dates x configs (each column = one config simulated continuously over the whole period).
    Switching configs at a window boundary is charged sum(|w_new - w_old|) x one-way cost rate."""
    start, end = utc(start), utc(end)
    rows, tr_rows, pieces = [], [], []
    prev = None
    t = start + pd.Timedelta(days=train_days)
    while t <= end:
        t_end = min(t + pd.Timedelta(days=test_days - 1), end)
        train = returns.loc[t - pd.Timedelta(days=train_days): t - pd.Timedelta(days=1)]
        sr = {c: sharpe_per_period(train[c]) * np.sqrt(ppy) for c in returns.columns}
        best = max(sr, key=sr.get)
        piece = returns.loc[t:t_end, best].copy()
        if prev is not None and best != prev and weights is not None and len(piece):
            w_old = weights[prev].loc[:t - pd.Timedelta(days=1)].iloc[-1]
            w_new = weights[best].loc[:t].iloc[-1]
            rates = pd.Series(cost_rates).reindex(w_old.index).fillna(0.0)
            piece.iloc[0] -= float(((w_new - w_old).abs() * rates).sum())
        pieces.append(piece)
        rows.append({"test_start": t.date(), "test_end": t_end.date(), "chosen": best, "train_sharpe": sr[best]})
        tr_rows.append(pd.Series(sr, name=t.date()))
        prev = best
        t = t + pd.Timedelta(days=step_days)
    oos = pd.concat(pieces)
    oos = oos[~oos.index.duplicated(keep="first")]
    return WFResult(oos, pd.DataFrame(rows), pd.DataFrame(tr_rows))
