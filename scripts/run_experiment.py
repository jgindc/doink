"""Run one pre-registered experiment and write its report.

  uv run python scripts/run_experiment.py EXP-003

Refuses to run if the registry file is not committed or has local changes. Trials are logged once per
experiment in experiments/trials.csv (re-runs after a bug fix are noted in the results folder, not re-counted).
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from swingbot.data.store import dataset_version
from swingbot.reporting.report import df_md, equity_plot, gates_table, stats_table
from swingbot.research import phase1
from swingbot.research.registry import TRIALS, assert_preregistered, head_commit, load_entry, log_trials, results_dir

CAVEATS = """## Caveats (read before drawing conclusions)
- Backtest ≠ expected live result. Research data only (2016 → 2024-09-30); the 2024-10 → today holdout is still locked.
- The crypto universe was picked with hindsight (coins that exist and are liquid today) → survivorship bias, worst for
  rotation. G9 re-checks on BTC/ETH/LTC/BCH only.
- Costs: Alpaca tier-1 taker fee 25 bps + per-asset slippage from one spread snapshot. Real fills may differ.
- Sharpe uses a 0% risk-free rate; idle cash earns 0% in the simulator (real T-bill yields would help cash-heavy books).
- *After-tax figures are rough estimates (ST 35% / LT 20%, liquidation at the end), simulated from the OOS start.
- Statistical power is limited: ~5.75 years OOS of a highly volatile market dominated by a few bull/bear cycles."""


def header(entry: dict, reg: str) -> str:
    return (f"# {entry['id']} — {entry['title']}\n\n"
            f"Registered: `{reg}` · Run: {datetime.now(timezone.utc).isoformat(timespec='seconds')} · "
            f"Dataset: `{dataset_version()}` · Code: `{head_commit()}`\n\n"
            f"**Hypothesis.** {entry['hypothesis'].strip()}\n")


def verdict(gates: dict) -> str:
    n_ok = sum(ok for ok, _ in gates.values())
    status = "PASS — candidate for critic review" if n_ok == len(gates) else "FAIL"
    return f"## Verdict: {status} ({n_ok}/{len(gates)} gates)\n"


def report_benchmarks(entry, reg, out) -> str:
    r = phase1.run_benchmarks_experiment(entry)
    ctx_oos = entry["periods"]["oos"]
    equity_plot(r["equity"], out / "equity.png", "Crypto benchmarks, 2016 → 2024-09 (research data)")
    md = [header(entry, reg),
          f"## Full research period ({entry['periods']['full'][0]} → {entry['periods']['full'][1]})",
          stats_table(r["full"]),
          f"\n## Walk-forward OOS span ({ctx_oos[0]} → {ctx_oos[1]}) — the bar for every strategy",
          stats_table(r["oos"]),
          "\n## Calendar-year returns", df_md(r["calendar"], pct=True), "\n" + CAVEATS]
    return "\n\n".join(md)


def report_grid(entry, reg, out) -> tuple[str, list]:
    r = phase1.run_grid_experiment(entry)
    v = r["oos_stats"]
    rows = {"Strategy (walk-forward OOS)": v["base"], **r["bench"]["base"]}
    stress = {f"{k}": {"sharpe": v[k]["sharpe"], "cagr": v[k]["cagr"], "max_dd": v[k]["max_dd"],
                       "trades": v[k]["trades"]} for k in v}
    stress_bench = pd.DataFrame({k: {"strategy Sharpe": v[k]["sharpe"], "best benchmark Sharpe": r["best_bench"][k],
                                     "BTC B&H Sharpe": r["bench"][k]["BTC buy&hold"]["sharpe"]} for k in v}).T
    grid = pd.DataFrame({cid: {"OOS Sharpe": s["sharpe"], "OOS CAGR": s["cagr"], "OOS MaxDD": s["max_dd"]}
                         for cid, s in r["cfg_oos"].items()}).T
    grid.index.name = "config"
    wfw = r["wf_windows"].set_index("test_start")
    b = r["bootstrap"]
    at = r["after_tax_default"]
    equity_plot(r["equity"], out / "equity.png", f"{entry['id']} walk-forward OOS vs BTC buy-and-hold")
    md = [header(entry, reg), verdict(r["gates"]), gates_table(r["gates"]),
          "\n## Walk-forward out-of-sample vs benchmarks (same dates)", stats_table(rows),
          "\n## Stress and robustness variants (walk-forward OOS)", stats_table(stress),
          df_md(stress_bench),
          "\n## Every grid configuration over the OOS dates (continuous simulations)", df_md(grid),
          "\n## Walk-forward choices (train 3y → test 6m)", df_md(wfw),
          "\n## Overfitting checks",
          f"- Deflated Sharpe (vs zero, N={r['dsr']['n_trials']} trials): **{r['dsr']['dsr']:.3f}**\n"
          f"- Deflated Sharpe of *active* returns vs BTC buy-and-hold: **{r['dsr_active']['dsr']:.3f}**\n"
          f"- PBO (CSCV, 16 blocks, {r['pbo']['n_splits']} splits, from 2017): **{r['pbo']['pbo']:.2f}**",
          "\n## Bootstrap (paired stationary blocks, 2000 draws, vs BTC buy-and-hold)",
          f"- P(strategy Sharpe > BTC B&H Sharpe): **{b['p_sharpe_beats_benchmark']:.0%}**\n"
          f"- Sharpe 5/50/95%: {b['sharpe']['p5']:.2f} / {b['sharpe']['p50']:.2f} / {b['sharpe']['p95']:.2f}\n"
          f"- CAGR 5/50/95%: {b['cagr']['p5']:.1%} / {b['cagr']['p50']:.1%} / {b['cagr']['p95']:.1%}\n"
          f"- Max DD 5/50/95%: {b['max_dd']['p5']:.1%} / {b['max_dd']['p50']:.1%} / {b['max_dd']['p95']:.1%}",
          "\n## After-tax estimate (default config vs BTC buy-and-hold, simulated from OOS start)",
          df_md(pd.DataFrame(at).T, pct=True),
          "\n## Calendar-year returns (OOS)", df_md(r["calendar"], pct=True), "\n" + CAVEATS]
    return "\n\n".join(md), r["trial_rows"]


def sleeve_trial_sharpes_pp() -> list[float]:
    with TRIALS.open() as f:
        rows = [r for r in csv.DictReader(f) if r["exp_id"] in ("EXP-003", "EXP-004", "EXP-005")]
    if len({r["exp_id"] for r in rows}) < 3:
        raise RuntimeError("run EXP-003, EXP-004 and EXP-005 before EXP-006 (its DSR uses their trial Sharpes)")
    return [float(r["oos_sharpe"]) / np.sqrt(phase1.PPY) for r in rows]


def report_ensemble(entry, reg, out) -> tuple[str, list]:
    r = phase1.run_ensemble_experiment(entry, sleeve_trial_sharpes_pp())
    s = r["stats"]
    rows = {"Ensemble (OOS dates)": s["base"], **r["bench"]["base"]}
    stress = {k: s[k] for k in s}
    stress_bench = pd.DataFrame({k: {"ensemble Sharpe": s[k]["sharpe"], "best benchmark Sharpe": r["best_bench"][k],
                                     "BTC B&H Sharpe": r["bench"][k]["BTC buy&hold"]["sharpe"]}
                                 for k in r["best_bench"]}).T
    b = r["bootstrap"]
    equity_plot(r["equity"], out / "equity.png", "EXP-006 ensemble vs benchmarks (OOS dates)")
    md = [header(entry, reg), verdict(r["gates"]), gates_table(r["gates"]),
          "\n## OOS dates (2018-12-31 → 2024-09-30) vs benchmarks", stats_table(rows),
          "\n## Full research period 2016 → 2024-09 (no parameters were fit, so this is informative too)",
          stats_table({"Ensemble": r["full_stats"]}),
          "\n## Stress and robustness variants", stats_table(stress), df_md(stress_bench),
          "\n## Average OOS weights", df_md(r["avg_weights"].to_frame("avg weight"), pct=True),
          "\n## Overfitting checks",
          f"- Deflated Sharpe (vs zero, N={r['dsr']['n_trials']}): **{r['dsr']['dsr']:.3f}**\n"
          f"- Deflated Sharpe of active returns vs BTC buy-and-hold: **{r['dsr_active']['dsr']:.3f}**",
          "\n## Bootstrap vs BTC buy-and-hold (paired, 2000 draws)",
          f"- P(ensemble Sharpe > BTC B&H Sharpe): **{b['p_sharpe_beats_benchmark']:.0%}**\n"
          f"- Sharpe 5/50/95%: {b['sharpe']['p5']:.2f} / {b['sharpe']['p50']:.2f} / {b['sharpe']['p95']:.2f}\n"
          f"- CAGR 5/50/95%: {b['cagr']['p5']:.1%} / {b['cagr']['p50']:.1%} / {b['cagr']['p95']:.1%}\n"
          f"- Max DD 5/50/95%: {b['max_dd']['p5']:.1%} / {b['max_dd']['p50']:.1%} / {b['max_dd']['p95']:.1%}",
          "\n## After-tax estimate (simulated from OOS start)", df_md(pd.DataFrame(r["after_tax"]).T, pct=True),
          "\n## Calendar-year returns (OOS)", df_md(r["calendar"], pct=True), "\n" + CAVEATS]
    return "\n\n".join(md), r["trial_rows"]


def report_luck(entry, reg, out) -> str:
    r = phase1.run_luck_experiment(entry)
    s, g = r["singles"], r["groups"]
    s.to_csv(out / "singles.csv", index=False)
    g.to_csv(out / "groups.csv", index=False)
    rate = lambda df, cols: pd.DataFrame({c: df.groupby("kind")[c].mean() for c in cols}).assign(  # noqa: E731
        n=df.groupby("kind").size())
    tot_s = s[["G1", "G2", "G3", "all"]].mean().to_frame("all kinds").T
    tot_g = g[["G1", "G2", "G3", "G4", "G5", "G6", "all"]].mean().to_frame("all kinds").T
    md = [header(entry, reg),
          f"Benchmarks over OOS dates: best Sharpe = {r['best_bench']['base']:.2f}; "
          f"BTC buy-and-hold Sharpe = {r['bench']['BTC buy&hold']['sharpe']:.2f}, "
          f"max DD = {r['bench']['BTC buy&hold']['max_dd']:.1%}.",
          "\n## Single random strategies — share passing each gate (OOS dates)",
          df_md(rate(s, ["G1", "G2", "G3", "all"]), pct=True), df_md(tot_s, pct=True),
          f"\nRandom-strategy OOS Sharpe percentiles 50/90/99%: {s.sharpe.quantile(.5):.2f} / "
          f"{s.sharpe.quantile(.9):.2f} / {s.sharpe.quantile(.99):.2f}",
          "\n## Pseudo-experiments (groups of 9 random configs through the same walk-forward) — pass shares",
          df_md(rate(g, ["G1", "G2", "G3", "G4", "G5", "G6", "all"]), pct=True), df_md(tot_g, pct=True),
          "\nG7–G9 are not evaluated for random groups, so the joint pass share above is an upper bound on how often "
          "luck alone clears the full gate set."]
    return "\n\n".join(md)


def main() -> None:
    exp_id = sys.argv[1]
    path, entry = load_entry(exp_id)
    reg = assert_preregistered(path)
    out = results_dir(exp_id)
    kind = entry["kind"]
    trial_rows: list = []
    if kind == "benchmarks":
        md = report_benchmarks(entry, reg, out)
    elif kind == "grid":
        md, trial_rows = report_grid(entry, reg, out)
    elif kind == "fixed":
        md, trial_rows = report_ensemble(entry, reg, out)
    elif kind == "luck":
        md = report_luck(entry, reg, out)
    else:
        raise ValueError(kind)
    (out / "summary.md").write_text(md + "\n")
    already = TRIALS.exists() and any(r["exp_id"] == exp_id for r in csv.DictReader(TRIALS.open()))
    if trial_rows and not already:
        log_trials([{**r, "dataset_version": dataset_version(), "git_commit": head_commit()} for r in trial_rows])
    with (out / "runs.log").open("a") as f:
        f.write(json.dumps({"run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "registered": reg,
                            "code": head_commit(), "dataset": dataset_version(),
                            "trials_logged": bool(trial_rows and not already)}) + "\n")
    print(md)


if __name__ == "__main__":
    main()
