"""Phase 1 (crypto) experiment runners. Each function implements its registry entry as written.

Gate definitions live in docs/TESTING_PROTOCOL.md. All evaluation uses research data only (before the
locked holdout); the holdout is never touched here.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import product
from types import SimpleNamespace

import numpy as np
import pandas as pd

from swingbot.config import load_config, utc
from swingbot.data.panel import Panel, load_crypto_panel
from swingbot.execution.simulator import CostModel, SimResult, simulate
from swingbot.portfolio.combine import apply_caps, build_book, combine
from swingbot.reporting.metrics import calendar_returns, perf, sim_stats
from swingbot.reporting.tax import after_tax
from swingbot.research import luck
from swingbot.research.overfit import deflated_sharpe, paired_bootstrap, pbo_cscv, sharpe_per_period
from swingbot.research.walkforward import walk_forward
from swingbot.strategies import benchmarks as bm
from swingbot.strategies.sleeves import (DipParams, RotationParams, TrendParams, dip_sleeve, rotation_sleeve,
                                         trend_sleeve)

PPY = 365
MAIN_BENCH = ["BTC buy&hold", "50/50 BTC/ETH", "BTC 200d SMA"]
PBO_START = "2017-01-01"  # after indicator warm-up so every config is live
SLEEVES = {"trend": (TrendParams, trend_sleeve), "rotation": (RotationParams, rotation_sleeve),
           "dip": (DipParams, dip_sleeve)}


@dataclass
class Ctx:
    cfg: dict
    main: Panel
    legacy: Panel
    start: pd.Timestamp
    end: pd.Timestamp
    oos0: pd.Timestamp
    oos1: pd.Timestamp
    band: float
    minh: int
    wf: dict

    @property
    def variants(self) -> dict:
        """name -> (panel, cost multiplier, extra delay bars)"""
        return {"base": (self.main, 1.0, 0), "cost2x": (self.main, self.cfg["costs"]["stress_multiplier"], 0),
                "delay1": (self.main, 1.0, self.cfg["execution"]["delay_stress_bars"]), "legacy": (self.legacy, 1.0, 0)}


def make_ctx(entry: dict) -> Ctx:
    cfg = load_config("crypto")
    main = load_crypto_panel(cfg)  # purpose="research": stops before the locked holdout
    legacy = main.subset(cfg["data"]["legacy_universe"])
    oos = entry.get("oos") or entry.get("periods", {}).get("oos")
    sim = entry.get("simulation", {})
    return Ctx(cfg=cfg, main=main, legacy=legacy, start=utc(sim.get("start", cfg["splits"]["research_start"])),
               end=main.index[-1], oos0=utc(oos[0]), oos1=utc(oos[1]),
               band=float(sim.get("band", cfg["portfolio"]["no_trade_band"])),
               minh=cfg["data"]["min_history_days"], wf=cfg["splits"]["walk_forward"])


# ------------------------------------------------------------------ building blocks
def run_sim(ctx: Ctx, targets: pd.DataFrame, panel: Panel, mult: float = 1.0, delay: int = 0,
            band: float | None = None, track_lots: bool = False, start=None) -> SimResult:
    return simulate(targets, panel, CostModel.from_config(ctx.cfg, mult), start=start or ctx.start, end=ctx.end,
                    delay=delay, band=ctx.band if band is None else band, track_lots=track_lots,
                    long_term_days=ctx.cfg["tax"]["long_term_days"])


def sleeve_targets(name: str, params: dict, panel: Panel, ctx: Ctx) -> pd.DataFrame:
    cls, fn = SLEEVES[name]
    p = cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in params.items()})
    return fn(panel, p, PPY, ctx.minh)


def benchmark_targets(panel: Panel, minh: int) -> dict[str, pd.DataFrame]:
    return {"BTC buy&hold": bm.buy_and_hold(panel, {"BTC": 1.0}, rebalance="never"),
            "50/50 BTC/ETH": bm.buy_and_hold(panel, {"BTC": 0.5, "ETH": 0.5}, rebalance="monthly"),
            "BTC 200d SMA": bm.sma_trend(panel, "BTC", 200),
            "Equal-weight universe (info)": bm.equal_weight(panel, panel.eligible(minh))}


def run_benchmarks(ctx: Ctx, panel: Panel, mult: float = 1.0, delay: int = 0, track_lots: bool = False,
                   start=None) -> dict[str, SimResult]:
    return {k: run_sim(ctx, t, panel, mult, delay, band=0.0, track_lots=track_lots, start=start)
            for k, t in benchmark_targets(panel, ctx.minh).items()}


def grid_list(entry: dict) -> list[tuple[str, dict]]:
    keys = list(entry["grid"])
    out = []
    for combo in product(*(entry["grid"][k] for k in keys)):
        params = {**entry["default"], **dict(zip(keys, combo))}
        cid = "|".join(f"{k}={'-'.join(map(str, v)) if isinstance(v, list) else v}" for k, v in zip(keys, combo))
        out.append((cid, params))
    return out


def splice(wf, sims: dict[str, SimResult]) -> SimpleNamespace:
    """Stitch the chosen config's turnover/costs/weights/episodes per window, to compute stats on WF OOS."""
    parts = {"turnover": [], "costs": [], "weights": [], "episodes": []}
    for _, w in wf.windows.iterrows():
        s = sims[w["chosen"]]
        t0, t1 = utc(w["test_start"]), utc(w["test_end"])
        parts["turnover"].append(s.turnover.loc[t0:t1])
        parts["costs"].append(s.costs.loc[t0:t1])
        parts["weights"].append(s.weights.loc[t0:t1])
        ep = s.episodes
        parts["episodes"].append(ep[(ep["entry"] >= t0) & (ep["entry"] <= t1)])
    return SimpleNamespace(returns=wf.oos_returns, turnover=pd.concat(parts["turnover"]),
                           costs=pd.concat(parts["costs"]), weights=pd.concat(parts["weights"]),
                           episodes=pd.concat(parts["episodes"], ignore_index=True))


def run_wf(ctx: Ctx, sims: dict[str, SimResult], panel: Panel):
    rets = pd.DataFrame({k: s.returns for k, s in sims.items()})
    rates = next(iter(sims.values())).cost_rates
    return walk_forward(rets, {k: s.weights for k, s in sims.items()}, rates, start=ctx.start, end=ctx.end,
                        train_days=ctx.wf["train_days"], test_days=ctx.wf["test_days"],
                        step_days=ctx.wf["step_days"], ppy=PPY)


def oos_slice(ctx: Ctx, r: pd.Series) -> pd.Series:
    return r.loc[ctx.oos0:ctx.oos1]


def after_tax_compare(ctx: Ctx, targets: pd.DataFrame) -> dict:
    """Simulate from the OOS start with tax lots: strategy vs BTC buy-and-hold, pre- and after-tax CAGR."""
    tax = ctx.cfg["tax"]
    out = {}
    for name, tg, band in (("strategy", targets, None),
                           ("BTC buy&hold", bm.buy_and_hold(ctx.main, {"BTC": 1.0}, "never"), 0.0)):
        res = run_sim(ctx, tg, ctx.main, band=band, track_lots=True, start=ctx.oos0)  # ends at research end
        at = after_tax(res, tax["short_term_rate"], tax["long_term_rate"], tax["long_term_days"])
        out[name] = {"cagr_pre_tax": perf(res.returns, PPY)["cagr"], "cagr_after_tax": at["cagr_after_tax"]}
    return out


# ------------------------------------------------------------------ EXP-001 benchmarks
def run_benchmarks_experiment(entry: dict) -> dict:
    ctx = make_ctx(entry)
    sims = run_benchmarks(ctx, ctx.main)
    full = {k: sim_stats(s, PPY) for k, s in sims.items()}
    oos = {k: sim_stats(s, PPY, ctx.oos0, ctx.oos1) for k, s in sims.items()}
    cal = pd.DataFrame({k: calendar_returns(s.returns) for k, s in sims.items()})
    tax = ctx.cfg["tax"]
    at = {}
    for k, t in benchmark_targets(ctx.main, ctx.minh).items():
        r = run_sim(ctx, t, ctx.main, band=0.0, track_lots=True, start=ctx.oos0)
        at[k] = after_tax(r, tax["short_term_rate"], tax["long_term_rate"], tax["long_term_days"])["cagr_after_tax"]
    for k in oos:
        oos[k]["cagr_after_tax"] = at[k]
    return {"full": full, "oos": oos, "calendar": cal, "equity": {k: s.equity for k, s in sims.items()}}


# ------------------------------------------------------------------ EXP-003/004/005 sleeve grids
def run_grid_experiment(entry: dict) -> dict:
    ctx = make_ctx(entry)
    sleeve = entry["sleeve"]
    configs = grid_list(entry)
    sims, wfs, oos_stats, bench, best_bench = {}, {}, {}, {}, {}
    for v, (panel, mult, delay) in ctx.variants.items():
        sims[v] = {cid: run_sim(ctx, sleeve_targets(sleeve, d, panel, ctx), panel, mult, delay) for cid, d in configs}
        wfs[v] = run_wf(ctx, sims[v], panel)
        oos_stats[v] = sim_stats(splice(wfs[v], sims[v]), PPY)
        bench[v] = {k: sim_stats(s, PPY, ctx.oos0, ctx.oos1) for k, s in run_benchmarks(ctx, panel, mult, delay).items()}
        best_bench[v] = max(bench[v][k]["sharpe"] for k in MAIN_BENCH)

    base = sims["base"]
    cfg_oos = {cid: perf(oos_slice(ctx, s.returns), PPY) for cid, s in base.items()}
    cfg_sr_pp = [sharpe_per_period(oos_slice(ctx, s.returns)) for s in base.values()]
    wf_r = wfs["base"].oos_returns
    n_trials = entry.get("dsr", {}).get("n_trials", len(configs))
    dsr = deflated_sharpe(wf_r, n_trials, trial_sharpes=cfg_sr_pp)
    bh_oos = oos_slice(ctx, run_sim(ctx, bm.buy_and_hold(ctx.main, {"BTC": 1.0}, "never"), ctx.main, band=0.0).returns)
    active = (wf_r - bh_oos.reindex(wf_r.index)).dropna()
    dsr_active = deflated_sharpe(active, n_trials, trial_sharpes=cfg_sr_pp)
    pbo = pbo_cscv(pd.DataFrame({k: s.returns.loc[utc(PBO_START):] for k, s in base.items()}))
    btc = bench["base"]["BTC buy&hold"]
    s = oos_stats["base"]
    gates = {
        "G1 Sharpe > best benchmark": (s["sharpe"] > best_bench["base"], f"{s['sharpe']:.2f} vs {best_bench['base']:.2f}"),
        "G2 MaxDD <= 0.75x BTC B&H": (abs(s["max_dd"]) <= 0.75 * abs(btc["max_dd"]),
                                      f"{s['max_dd']:.1%} vs limit {-0.75 * abs(btc['max_dd']):.1%}"),
        "G3 >= 30 trades": (s["trades"] >= 30, f"{s['trades']}"),
        "G4 2x costs still > best benchmark": (oos_stats["cost2x"]["sharpe"] > best_bench["cost2x"],
                                               f"{oos_stats['cost2x']['sharpe']:.2f} vs {best_bench['cost2x']:.2f}"),
        "G5 +1 bar delay still > best benchmark": (oos_stats["delay1"]["sharpe"] > best_bench["delay1"],
                                                   f"{oos_stats['delay1']['sharpe']:.2f} vs {best_bench['delay1']:.2f}"),
        "G6 median config > BTC B&H": (np.median([c["sharpe"] for c in cfg_oos.values()]) > btc["sharpe"],
                                       f"{np.median([c['sharpe'] for c in cfg_oos.values()]):.2f} vs {btc['sharpe']:.2f}"),
        "G7 DSR >= 0.95": (dsr["dsr"] >= 0.95, f"{dsr['dsr']:.3f} (N={n_trials})"),
        "G8 PBO < 0.5": (pbo["pbo"] < 0.5, f"{pbo['pbo']:.2f}"),
        "G9 legacy universe > BTC B&H": (oos_stats["legacy"]["sharpe"] > bench["legacy"]["BTC buy&hold"]["sharpe"],
                                         f"{oos_stats['legacy']['sharpe']:.2f} vs {bench['legacy']['BTC buy&hold']['sharpe']:.2f}"),
    }
    default_tg = sleeve_targets(sleeve, entry["default"], ctx.main, ctx)
    return {
        "ctx": ctx, "configs": configs, "oos_stats": oos_stats, "bench": bench, "best_bench": best_bench,
        "cfg_oos": cfg_oos, "wf_windows": wfs["base"].windows, "train_sharpes": wfs["base"].train_sharpes,
        "dsr": dsr, "dsr_active": dsr_active, "pbo": pbo, "gates": gates,
        "bootstrap": paired_bootstrap(wf_r, bh_oos, PPY),
        "after_tax_default": after_tax_compare(ctx, default_tg),
        "calendar": pd.DataFrame({"strategy (WF)": calendar_returns(wf_r), "BTC buy&hold": calendar_returns(bh_oos)}),
        "equity": {"strategy (WF OOS)": (1 + wf_r).cumprod(), "BTC buy&hold": (1 + bh_oos).cumprod()},
        "trial_rows": [{"exp_id": entry["id"], "config_id": cid, "params": d,
                        "oos_sharpe": round(cfg_oos[cid]["sharpe"], 4), "oos_cagr": round(cfg_oos[cid]["cagr"], 4),
                        "oos_max_dd": round(cfg_oos[cid]["max_dd"], 4)} for cid, d in configs],
    }


# ------------------------------------------------------------------ EXP-006 ensemble
def ensemble_targets(ctx: Ctx, panel: Panel, sleeves_cfg: dict, vol_target: bool = True) -> pd.DataFrame:
    sl = {name: sleeve_targets(name, params, panel, ctx) for name, params in sleeves_cfg.items()}
    if vol_target:
        return build_book(sl, ctx.cfg, panel.close)
    p = ctx.cfg["portfolio"]
    return apply_caps(combine(sl, p["sleeve_budget"]), p["max_weight"])


def run_ensemble_experiment(entry: dict, sleeve_trial_sharpes_pp: list[float]) -> dict:
    ctx = make_ctx(entry)
    sc = entry["config"]["sleeves"]
    stats, bench, best_bench, sims = {}, {}, {}, {}
    for v, (panel, mult, delay) in ctx.variants.items():
        sims[v] = run_sim(ctx, ensemble_targets(ctx, panel, sc), panel, mult, delay)
        stats[v] = sim_stats(sims[v], PPY, ctx.oos0, ctx.oos1)
        bench[v] = {k: sim_stats(s, PPY, ctx.oos0, ctx.oos1) for k, s in run_benchmarks(ctx, panel, mult, delay).items()}
        best_bench[v] = max(bench[v][k]["sharpe"] for k in MAIN_BENCH)
    novt = run_sim(ctx, ensemble_targets(ctx, ctx.main, sc, vol_target=False), ctx.main)
    stats["no_vol_target (diagnostic)"] = sim_stats(novt, PPY, ctx.oos0, ctx.oos1)
    full_stats = sim_stats(sims["base"], PPY)
    bh_sim = run_sim(ctx, bm.buy_and_hold(ctx.main, {"BTC": 1.0}, "never"), ctx.main, band=0.0)
    bh_oos = oos_slice(ctx, bh_sim.returns)
    r = oos_slice(ctx, sims["base"].returns)
    n_trials = entry["dsr"]["n_trials"]
    dsr = deflated_sharpe(r, n_trials, trial_sharpes=sleeve_trial_sharpes_pp)
    dsr_active = deflated_sharpe((r - bh_oos).dropna(), n_trials, trial_sharpes=sleeve_trial_sharpes_pp)
    btc = bench["base"]["BTC buy&hold"]
    s = stats["base"]
    gates = {
        "G1 Sharpe > best benchmark": (s["sharpe"] > best_bench["base"], f"{s['sharpe']:.2f} vs {best_bench['base']:.2f}"),
        "G2 MaxDD <= 0.75x BTC B&H": (abs(s["max_dd"]) <= 0.75 * abs(btc["max_dd"]),
                                      f"{s['max_dd']:.1%} vs limit {-0.75 * abs(btc['max_dd']):.1%}"),
        "G3 >= 30 trades": (s["trades"] >= 30, f"{s['trades']}"),
        "G4 2x costs still > best benchmark": (stats["cost2x"]["sharpe"] > best_bench["cost2x"],
                                               f"{stats['cost2x']['sharpe']:.2f} vs {best_bench['cost2x']:.2f}"),
        "G5 +1 bar delay still > best benchmark": (stats["delay1"]["sharpe"] > best_bench["delay1"],
                                                   f"{stats['delay1']['sharpe']:.2f} vs {best_bench['delay1']:.2f}"),
        "G7 DSR >= 0.95": (dsr["dsr"] >= 0.95, f"{dsr['dsr']:.3f} (N={n_trials})"),
        "G9 legacy universe > BTC B&H": (stats["legacy"]["sharpe"] > bench["legacy"]["BTC buy&hold"]["sharpe"],
                                         f"{stats['legacy']['sharpe']:.2f} vs {bench['legacy']['BTC buy&hold']['sharpe']:.2f}"),
    }
    bench_eq = {k: oos_slice(ctx, x.returns) for k, x in run_benchmarks(ctx, ctx.main).items() if k in MAIN_BENCH}
    return {
        "ctx": ctx, "stats": stats, "full_stats": full_stats, "bench": bench, "best_bench": best_bench,
        "dsr": dsr, "dsr_active": dsr_active, "gates": gates,
        "bootstrap": paired_bootstrap(r, bh_oos, PPY),
        "after_tax": after_tax_compare(ctx, ensemble_targets(ctx, ctx.main, sc)),
        "calendar": pd.DataFrame({"ensemble": calendar_returns(r), "BTC buy&hold": calendar_returns(bh_oos)}),
        "equity": {"ensemble": (1 + r).cumprod(), **{k: (1 + x).cumprod() for k, x in bench_eq.items()}},
        "avg_weights": sims["base"].weights.loc[ctx.oos0:ctx.oos1].mean().sort_values(ascending=False),
        "trial_rows": [{"exp_id": entry["id"], "config_id": "ensemble_default", "params": sc,
                        "oos_sharpe": round(s["sharpe"], 4), "oos_cagr": round(s["cagr"], 4),
                        "oos_max_dd": round(s["max_dd"], 4)},
                       {"exp_id": entry["id"], "config_id": "ensemble_no_vol_target", "params": {"vol_target": False},
                        "oos_sharpe": round(stats["no_vol_target (diagnostic)"]["sharpe"], 4),
                        "oos_cagr": round(stats["no_vol_target (diagnostic)"]["cagr"], 4),
                        "oos_max_dd": round(stats["no_vol_target (diagnostic)"]["max_dd"], 4)}],
    }


# ------------------------------------------------------------------ EXP-002 luck calibration
_G: dict = {}


def _luck_init(entry: dict) -> None:
    ctx = make_ctx(entry)
    starts = []
    t = ctx.start + pd.Timedelta(days=ctx.wf["train_days"])
    while t <= ctx.end:
        starts.append(t - pd.Timedelta(days=1))
        t += pd.Timedelta(days=ctx.wf["step_days"])
    _G.update(ctx=ctx, boundary=pd.DatetimeIndex(starts))


def _luck_one(job: tuple) -> dict:
    kind, params, seed = job
    ctx: Ctx = _G["ctx"]
    rng = np.random.default_rng(seed)
    if kind == "timing":
        tg = luck.random_timing(ctx.main, rng, PPY, ctx.minh, params["mean_hold"], params["p_in"])
    else:
        tg = luck.random_rotation(ctx.main, rng, PPY, ctx.minh, params["top_n"])
    out = {"kind": kind, "params": params}
    for v in ("base", "cost2x", "delay1"):
        panel, mult, delay = ctx.variants[v]
        res = run_sim(ctx, tg, panel, mult, delay)
        out[v] = {"returns": res.returns, "weights": res.weights.reindex(_G["boundary"]),
                  "entries": res.episodes["entry"].to_numpy(), "rates": res.cost_rates}
    return out


def run_luck_experiment(entry: dict, workers: int = 4) -> dict:
    ctx = make_ctx(entry)
    gen = entry["generators"]
    rng = np.random.default_rng(entry["seed"])
    jobs = []
    for _ in range(gen["random_timing"]["count"]):
        lo, hi = gen["random_timing"]["time_in_market"]
        jobs.append(("timing", {"mean_hold": int(rng.choice(gen["random_timing"]["mean_hold_days"])),
                                "p_in": float(rng.uniform(lo, hi))}, int(rng.integers(2**31))))
    for _ in range(gen["random_rotation"]["count"]):
        jobs.append(("rotation", {"top_n": int(rng.choice(gen["random_rotation"]["top_n"]))},
                     int(rng.integers(2**31))))
    with ProcessPoolExecutor(max_workers=workers, initializer=_luck_init, initargs=(entry,)) as ex:
        results = list(ex.map(_luck_one, jobs, chunksize=8))

    bench = {v: {k: sim_stats(s, PPY, ctx.oos0, ctx.oos1) for k, s in run_benchmarks(ctx, ctx.main, m, d).items()}
             for v, (_, m, d) in ctx.variants.items() if v != "legacy"}
    best = {v: max(bench[v][k]["sharpe"] for k in MAIN_BENCH) for v in bench}
    btc = bench["base"]["BTC buy&hold"]

    def ntrades(entries, t0, t1):
        e = pd.DatetimeIndex(entries)
        return int(((e >= t0) & (e <= t1)).sum())

    singles = []
    for r in results:
        p = perf(oos_slice(ctx, r["base"]["returns"]), PPY)
        g1 = p["sharpe"] > best["base"]
        g2 = abs(p["max_dd"]) <= 0.75 * abs(btc["max_dd"])
        g3 = ntrades(r["base"]["entries"], ctx.oos0, ctx.oos1) >= 30
        singles.append({"kind": r["kind"], "sharpe": p["sharpe"], "max_dd": p["max_dd"], "G1": g1, "G2": g2, "G3": g3,
                        "all": g1 and g2 and g3})
    singles = pd.DataFrame(singles)

    groups = []
    gs = entry.get("group_size", 9)
    for gi in range(0, len(results) - gs + 1, gs):
        grp = results[gi:gi + gs]
        names = [f"r{gi + j}" for j in range(gs)]
        res_v = {}
        for v in ("base", "cost2x", "delay1"):
            rets = pd.DataFrame({n: g[v]["returns"] for n, g in zip(names, grp)})
            wts = {n: g[v]["weights"] for n, g in zip(names, grp)}
            wf = walk_forward(rets, wts, grp[0][v]["rates"], start=ctx.start, end=ctx.end,
                              train_days=ctx.wf["train_days"], test_days=ctx.wf["test_days"],
                              step_days=ctx.wf["step_days"], ppy=PPY)
            res_v[v] = wf
        p = perf(res_v["base"].oos_returns, PPY)
        trades = sum(ntrades(grp[names.index(w["chosen"])]["base"]["entries"], utc(w["test_start"]), utc(w["test_end"]))
                     for _, w in res_v["base"].windows.iterrows())
        med = np.median([perf(oos_slice(ctx, g["base"]["returns"]), PPY)["sharpe"] for g in grp])
        flags = {"G1": p["sharpe"] > best["base"], "G2": abs(p["max_dd"]) <= 0.75 * abs(btc["max_dd"]),
                 "G3": trades >= 30,
                 "G4": perf(res_v["cost2x"].oos_returns, PPY)["sharpe"] > best["cost2x"],
                 "G5": perf(res_v["delay1"].oos_returns, PPY)["sharpe"] > best["delay1"],
                 "G6": med > btc["sharpe"]}
        flags["all"] = all(flags.values())
        groups.append({"kind": grp[0]["kind"], "sharpe": p["sharpe"], "max_dd": p["max_dd"], **flags})
    groups = pd.DataFrame(groups)
    return {"ctx": ctx, "singles": singles, "groups": groups, "bench": bench["base"], "best_bench": best}
