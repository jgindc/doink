"""Markdown + PNG reports for experiment results. Plain tables; every number labeled."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

COLS = [("cagr", "CAGR", "pct"), ("vol", "Vol", "pct"), ("sharpe", "Sharpe", "f2"), ("sortino", "Sortino", "f2"),
        ("max_dd", "Max DD", "pct"), ("calmar", "Calmar", "f2"), ("trades", "Trades", "int"),
        ("win_rate", "Win %", "pct0"), ("profit_factor", "PF", "f2"), ("avg_hold_days", "Avg hold d", "f0"),
        ("time_in_market", "In mkt", "pct0"), ("avg_exposure", "Avg expo", "pct0"),
        ("turnover_x_per_year", "Turnover x/yr", "f1"), ("cost_drag_per_year", "Cost drag/yr", "pct"),
        ("cagr_after_tax", "CAGR after tax*", "pct")]


def fmt(v, kind: str) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v))):
        return "–"
    if isinstance(v, float) and math.isinf(v):
        return "∞"
    return {"pct": f"{v:.1%}", "pct0": f"{v:.0%}", "f2": f"{v:.2f}", "f1": f"{v:.1f}", "f0": f"{v:.0f}",
            "int": f"{int(v)}"}[kind]


def stats_table(rows: dict[str, dict], cols=COLS) -> str:
    present = [c for c in cols if any(c[0] in r for r in rows.values())]
    head = "| | " + " | ".join(c[1] for c in present) + " |"
    sep = "|---|" + "---|" * len(present)
    lines = [head, sep]
    for name, r in rows.items():
        lines.append(f"| {name} | " + " | ".join(fmt(r.get(c[0]), c[2]) for c in present) + " |")
    return "\n".join(lines)


def gates_table(gates: dict) -> str:
    lines = ["| Gate | Result | Value |", "|---|---|---|"]
    for g, (ok, val) in gates.items():
        lines.append(f"| {g} | {'PASS' if ok else 'FAIL'} | {val} |")
    return "\n".join(lines)


def df_md(df: pd.DataFrame, pct: bool = False, digits: int = 2) -> str:
    def f(v):
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            if isinstance(v, float) and math.isnan(v):
                return "–"
            return f"{v:.1%}" if pct else f"{v:.{digits}f}"
        return str(v)
    cols = [str(c) for c in df.columns]
    lines = ["| " + str(df.index.name or "") + " | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        lines.append(f"| {idx} | " + " | ".join(f(v) for v in row.to_numpy()) + " |")
    return "\n".join(lines)


def equity_plot(curves: dict[str, pd.Series], path: Path, title: str) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for name, eq in curves.items():
        eq = eq / eq.iloc[0]
        ax1.plot(eq.index, eq.values, label=name, lw=1.4 if "strategy" in name or "ensemble" in name else 1.0)
        ax2.plot(eq.index, (eq / eq.cummax() - 1).values, lw=1.0)
    ax1.set_yscale("log")
    ax1.set_ylabel("growth of $1 (log)")
    ax1.legend(fontsize=8)
    ax1.set_title(title)
    ax1.grid(alpha=0.3)
    ax2.set_ylabel("drawdown")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
