"""Rough after-tax estimate for a U.S. taxable account. An ESTIMATE, not tax advice.

Model:
  * Realized gains from FIFO lots, split short-term / long-term by holding period.
  * Each calendar year: net ST and LT (a net loss in one offsets a gain in the other); remaining net losses
    carry forward with their character. The $3k ordinary-income offset is ignored (conservative).
  * Tax is paid out of the account at each year end (reduces compounding).
  * At the end of the test the position is assumed liquidated, so unrealized gains are taxed too. This makes
    buy-and-hold and active strategies comparable ("what you'd keep if you cashed out").
  * Wash-sale rules are ignored (not currently applied to crypto; they matter for ETFs later).
"""

from __future__ import annotations

import pandas as pd


def _net_tax(st: float, lt: float, st_rate: float, lt_rate: float) -> tuple[float, float, float]:
    """Return (tax, st_carry, lt_carry). Carries are <= 0."""
    if st >= 0 and lt >= 0:
        return st * st_rate + lt * lt_rate, 0.0, 0.0
    if st < 0 and lt < 0:
        return 0.0, st, lt
    total = st + lt
    if st < 0 <= lt:  # ST loss offsets LT gain
        return (total * lt_rate, 0.0, 0.0) if total >= 0 else (0.0, total, 0.0)
    return (total * st_rate, 0.0, 0.0) if total >= 0 else (0.0, 0.0, total)


def after_tax(res, st_rate: float, lt_rate: float, long_term_days: int = 366) -> dict:
    eq = res.equity
    initial = eq.iloc[0] / (1 + res.returns.iloc[0])
    ratio = 1.0
    cf_st = cf_lt = 0.0
    total_tax = 0.0
    ratios = pd.Series(1.0, index=eq.index)
    years = sorted(set(eq.index.year))
    for y in years[:-1]:  # complete years; the final (partial) year is settled at liquidation
        r = res.realized.get(y, {"st": 0.0, "lt": 0.0})
        tax, cf_st, cf_lt = _net_tax(r["st"] + cf_st, r["lt"] + cf_lt, st_rate, lt_rate)
        e_y = eq[eq.index.year == y].iloc[-1]
        if tax > 0 and e_y > 0:
            total_tax += tax * ratio
            ratio *= max(e_y - tax, 0.0) / e_y
        ratios[eq.index.year > y] = ratio

    # final year realized + liquidation of open lots at the last close (net of exit costs)
    last = eq.index[-1]
    r = res.realized.get(last.year, {"st": 0.0, "lt": 0.0})
    st, lt = r["st"], r["lt"]
    for asset, lots in res.open_lots.items():
        px = res.final_close[asset] * (1 - res.cost_rates.get(asset, 0.0))
        for d0, q, basis in lots:
            gain = q * (px - basis)
            if (last - d0).days >= long_term_days:
                lt += gain
            else:
                st += gain
    tax, _, _ = _net_tax(st + cf_st, lt + cf_lt, st_rate, lt_rate)
    final_pre = eq.iloc[-1]
    final_after = ratio * max(final_pre - tax, 0.0)
    total_tax += tax * ratio
    years_elapsed = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr_after = (final_after / initial) ** (1 / years_elapsed) - 1 if years_elapsed > 0 and final_after > 0 else float("nan")
    return {"final_after_tax": final_after, "cagr_after_tax": cagr_after, "taxes_paid": total_tax,
            "after_tax_equity": eq * ratios}
