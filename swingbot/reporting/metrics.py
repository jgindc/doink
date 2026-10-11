"""Performance metrics. Sharpe/Sortino use a zero risk-free rate (idle cash earns 0 in the simulator too)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from swingbot.config import utc


def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1).min())


def perf(returns: pd.Series, ppy: int) -> dict:
    r = returns.dropna()
    if len(r) < 2:
        return {}
    eq = (1 + r).cumprod()
    years = len(r) / ppy
    cagr = eq.iloc[-1] ** (1 / years) - 1 if eq.iloc[-1] > 0 else -1.0
    vol = r.std() * np.sqrt(ppy)
    sharpe = r.mean() / r.std() * np.sqrt(ppy) if r.std() > 0 else 0.0
    downside = np.sqrt((np.minimum(r, 0) ** 2).mean())
    sortino = r.mean() / downside * np.sqrt(ppy) if downside > 0 else np.nan
    mdd = max_drawdown(pd.concat([pd.Series([1.0]), eq.reset_index(drop=True)]))
    return {"cagr": cagr, "vol": vol, "sharpe": sharpe, "sortino": sortino, "max_dd": mdd,
            "calmar": cagr / abs(mdd) if mdd < 0 else np.nan, "days": len(r),
            "total_return": eq.iloc[-1] - 1}


def sim_stats(res, ppy: int, start=None, end=None, tax: dict | None = None) -> dict:
    """Full metric set for a SimResult, optionally restricted to [start, end]."""
    start = utc(start) if start is not None else None
    end = utc(end) if end is not None else None
    sl = slice(start, end)
    r = res.returns.loc[sl]
    out = perf(r, ppy)
    years = len(r) / ppy
    w = res.weights.loc[sl]
    gross = w.sum(axis=1)
    out["turnover_x_per_year"] = res.turnover.loc[sl].sum() / years if years else np.nan
    out["cost_drag_per_year"] = res.costs.loc[sl].sum() / years if years else np.nan
    out["time_in_market"] = float((gross > 1e-6).mean())
    out["avg_exposure"] = float(gross.mean())
    ep = res.episodes
    if len(ep):
        s = start if start is not None else r.index[0]
        e = end if end is not None else r.index[-1]
        ep = ep[(ep["entry"] >= s) & (ep["entry"] <= e)]
    out["trades"] = int(len(ep))
    if len(ep):
        wins, losses = ep.loc[ep.contrib > 0, "contrib"].sum(), -ep.loc[ep.contrib <= 0, "contrib"].sum()
        out["win_rate"] = float((ep.contrib > 0).mean())
        out["profit_factor"] = float(wins / losses) if losses > 0 else np.inf
        out["avg_hold_days"] = float(ep["days"].mean())
    else:
        out.update(win_rate=np.nan, profit_factor=np.nan, avg_hold_days=np.nan)
    if tax is not None and start is None and end is None:  # tax needs the whole simulated path
        from swingbot.reporting.tax import after_tax
        at = after_tax(res, tax["short_term_rate"], tax["long_term_rate"], tax.get("long_term_days", 366))
        out["cagr_after_tax"] = at["cagr_after_tax"]
    return out


def calendar_returns(returns: pd.Series) -> pd.Series:
    return (1 + returns).groupby(returns.index.year).prod() - 1
