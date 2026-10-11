"""Simulated broker / backtest loop. The same order logic is meant to drive paper trading.

Timing (no lookahead):
  * targets.loc[t] is decided at the close of bar t, using data <= t.
  * It executes at the OPEN of bar t+1 (+ `delay` extra bars for the delay stress test).
  * A row of all-NaN in `targets` means "no decision today" (hold); NaN for one asset in a valid row means 0.

Accounting is shares-and-cash, long-only, no leverage. Costs (fee + slippage) are charged on traded notional.
Optional FIFO tax lots record realized short/long-term gains by calendar year.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from swingbot.config import per_asset, utc


@dataclass
class CostModel:
    fee_bps: float
    slippage_bps: dict
    multiplier: float = 1.0

    def rates(self, assets: list[str]) -> np.ndarray:
        """One-way cost per unit of notional, per asset."""
        return np.array([(self.fee_bps + per_asset(self.slippage_bps, a)) * self.multiplier / 1e4 for a in assets])

    @classmethod
    def from_config(cls, cfg: dict, multiplier: float = 1.0) -> "CostModel":
        c = cfg["costs"]
        return cls(fee_bps=c["fee_bps"], slippage_bps=c["slippage_bps"], multiplier=multiplier)


@dataclass
class SimResult:
    equity: pd.Series                 # marked at each close
    returns: pd.Series                # close-to-close portfolio returns
    weights: pd.DataFrame             # holdings at close (after the day's trades), fraction of equity
    turnover: pd.Series               # sum |traded notional| / equity at open, per day
    costs: pd.Series                  # costs paid / previous close equity, per day
    episodes: pd.DataFrame            # one row per position episode (flat -> long -> flat)
    realized: dict = field(default_factory=dict)  # {year: {"st": gain, "lt": gain}} in equity units
    open_lots: dict = field(default_factory=dict)  # {asset: [(entry_date, shares, basis_per_share)]}
    final_close: dict = field(default_factory=dict)
    cost_rates: dict = field(default_factory=dict)


def simulate(targets: pd.DataFrame, panel, costs: CostModel, *, start=None, end=None, delay: int = 0,
             band: float = 0.0, track_lots: bool = False, long_term_days: int = 366,
             initial: float = 1.0) -> SimResult:
    assets = list(targets.columns)
    idx = panel.index
    exec_targets = targets.reindex(idx)[assets].shift(1 + delay)
    start = utc(start) if start is not None else idx[0]
    end = utc(end) if end is not None else idx[-1]
    dates = idx[(idx >= start) & (idx <= end)]

    O = panel.open[assets].reindex(dates).to_numpy(float)
    C = panel.close[assets].reindex(dates).to_numpy(float)
    T = exec_targets.reindex(dates).to_numpy(float)
    rate = costs.rates(assets)
    n, N = len(assets), len(dates)

    shares = np.zeros(n)
    cash = float(initial)
    last_px = np.full(n, np.nan)
    prev_val = np.zeros(n)
    prev_eq = float(initial)

    eq_out = np.empty(N)
    w_out = np.zeros((N, n))
    to_out = np.zeros(N)
    cost_out = np.zeros(N)

    in_ep = np.zeros(n, bool)
    ep_start = np.zeros(n, int)
    ep_contrib = np.zeros(n)
    episodes: list[tuple] = []

    lots: list[deque] = [deque() for _ in range(n)]
    realized: dict = {}

    def _realize(a: int, q: float, px_net: float, i: int, exit_all: bool) -> None:
        y = dates[i].year
        bucket = realized.setdefault(y, {"st": 0.0, "lt": 0.0})
        lot_q = lots[a]
        remaining = q
        while lot_q and (remaining > 1e-15 or exit_all):
            d0, lq, basis = lot_q[0]
            take = lq if exit_all else min(lq, remaining)
            held = (dates[i] - dates[d0]).days
            bucket["lt" if held >= long_term_days else "st"] += take * (px_net - basis)
            remaining -= take
            if take >= lq - 1e-15:
                lot_q.popleft()
            else:
                lot_q[0] = (d0, lq - take, basis)

    for i in range(N):
        o, c = O[i], C[i]
        mark_open = np.where(np.isfinite(o), o, last_px)
        val_open = shares * np.nan_to_num(mark_open)
        eq_open = cash + val_open.sum()
        buy_cash = np.zeros(n)
        sell_proc = np.zeros(n)
        day_cost = 0.0
        traded = 0.0

        row = T[i]
        if not np.all(np.isnan(row)) and eq_open > 0:
            tgt = np.clip(np.nan_to_num(row), 0.0, None)
            s = tgt.sum()
            if s > 1.0:
                tgt = tgt / s
            tradable = np.isfinite(o) & (o > 0)
            diff = tgt - val_open / eq_open
            exit_all = (tgt <= 0) & (shares > 0)
            act = tradable & ((np.abs(diff) > band) | exit_all)
            for a in np.flatnonzero(act & (diff < 0)):
                q = shares[a] if exit_all[a] else min(shares[a], -diff[a] * eq_open / o[a])
                notional = q * o[a]
                c_ = notional * rate[a]
                cash += notional - c_
                sell_proc[a] = notional - c_
                day_cost += c_
                traded += notional
                if track_lots:
                    _realize(a, q, o[a] * (1 - rate[a]), i, bool(exit_all[a]))
                shares[a] = 0.0 if exit_all[a] else shares[a] - q
            buys = np.flatnonzero(act & (diff > 0))
            if len(buys):
                want = diff[buys] * eq_open
                need = float(np.sum(want * (1 + rate[buys])))
                if need > cash:
                    want = want * (max(cash, 0.0) / need)
                for a, notional in zip(buys, want):
                    if notional <= 0:
                        continue
                    q = notional / o[a]
                    shares[a] += q
                    cash -= notional * (1 + rate[a])
                    buy_cash[a] = notional * (1 + rate[a])
                    day_cost += notional * rate[a]
                    traded += notional
                    if track_lots:
                        lots[a].append((i, q, o[a] * (1 + rate[a])))

        # open new episodes
        new = (shares > 0) & ~in_ep
        in_ep |= new
        ep_start[new] = i
        ep_contrib[new] = 0.0

        last_px = np.where(np.isfinite(c), c, last_px)
        val_close = shares * np.nan_to_num(last_px)
        eq_close = cash + val_close.sum()
        pnl = val_close - prev_val - buy_cash + sell_proc
        ep_contrib[in_ep] += pnl[in_ep] / prev_eq if prev_eq > 0 else 0.0
        closed = in_ep & (shares <= 0)
        for a in np.flatnonzero(closed):
            episodes.append((assets[a], dates[ep_start[a]], dates[i], (dates[i] - dates[ep_start[a]]).days,
                             ep_contrib[a], True))
        in_ep &= ~closed

        eq_out[i] = eq_close
        w_out[i] = val_close / eq_close if eq_close > 0 else 0.0
        to_out[i] = traded / eq_open if eq_open > 0 else 0.0
        cost_out[i] = day_cost / prev_eq if prev_eq > 0 else 0.0
        prev_val, prev_eq = val_close, eq_close

    for a in np.flatnonzero(in_ep):  # still-open episodes at the end
        episodes.append((assets[a], dates[ep_start[a]], dates[-1], (dates[-1] - dates[ep_start[a]]).days,
                         ep_contrib[a], False))

    equity = pd.Series(eq_out, index=dates, name="equity")
    rets = equity.pct_change()
    rets.iloc[0] = equity.iloc[0] / initial - 1.0
    ep_df = pd.DataFrame(episodes, columns=["asset", "entry", "exit", "days", "contrib", "closed"])
    return SimResult(
        equity=equity, returns=rets,
        weights=pd.DataFrame(w_out, index=dates, columns=assets),
        turnover=pd.Series(to_out, index=dates), costs=pd.Series(cost_out, index=dates),
        episodes=ep_df, realized=realized,
        open_lots={assets[a]: [(dates[d], q, b) for d, q, b in lots[a]] for a in range(n) if lots[a]},
        final_close={assets[a]: float(last_px[a]) for a in range(n)},
        cost_rates={assets[a]: float(rate[a]) for a in range(n)},
    )
