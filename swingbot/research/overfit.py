"""Overfitting statistics.

* Probabilistic / Deflated Sharpe Ratio — Bailey & López de Prado (2012, 2014).
* Probability of Backtest Overfitting via CSCV — Bailey, Borwein, López de Prado & Zhu (2017).
* Stationary bootstrap — Politis & Romano (1994).
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

EULER_GAMMA = 0.5772156649015329


def sharpe_per_period(r) -> float:
    r = np.asarray(r, float)
    r = r[~np.isnan(r)]
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd > 0 else 0.0


def expected_max_sharpe(n_trials: float, var_sr: float) -> float:
    """Expected maximum of n_trials Sharpe ratios drawn with zero true mean and variance var_sr."""
    if n_trials <= 1:
        return 0.0
    z1 = stats.norm.ppf(1 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(var_sr) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def probabilistic_sharpe(r, sr_benchmark: float = 0.0) -> float:
    """P(true per-period Sharpe > sr_benchmark), adjusting for sample length, skew and fat tails."""
    r = np.asarray(r, float)
    r = r[~np.isnan(r)]
    T = len(r)
    sr = sharpe_per_period(r)
    skew = stats.skew(r)
    kurt = stats.kurtosis(r, fisher=False)
    denom = np.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr ** 2, 1e-12))
    return float(stats.norm.cdf((sr - sr_benchmark) * np.sqrt(T - 1) / denom))


def deflated_sharpe(r, n_trials: float, trial_sharpes=None, var_sr: float | None = None) -> dict:
    """DSR: probability the selected strategy's Sharpe beats the best-of-N-by-luck threshold.
    Sharpe values are per period (daily), not annualized."""
    if var_sr is None:
        ts = np.asarray(trial_sharpes, float) if trial_sharpes is not None else np.array([0.0])
        var_sr = float(np.var(ts, ddof=1)) if len(ts) > 1 else 0.0
    sr0 = expected_max_sharpe(n_trials, var_sr)
    return {"dsr": probabilistic_sharpe(r, sr0), "sr0_per_period": sr0, "n_trials": n_trials, "var_sr": var_sr,
            "sr_per_period": sharpe_per_period(r)}


def pbo_cscv(returns: pd.DataFrame, n_blocks: int = 16) -> dict:
    """Probability of Backtest Overfitting. `returns`: T x N (one column per configuration tried).
    PBO = share of IS/OOS splits where the IS-best configuration ranks at or below the OOS median."""
    M = returns.dropna(how="any").to_numpy()
    T, N = M.shape
    if N < 2:
        return {"pbo": float("nan"), "n_splits": 0}
    size = T // n_blocks
    M = M[: size * n_blocks].reshape(n_blocks, size, N)
    s1 = M.sum(axis=1)                # block sums
    s2 = (M ** 2).sum(axis=1)         # block sums of squares
    logits = []
    degr = []
    for is_blocks in combinations(range(n_blocks), n_blocks // 2):
        is_mask = np.zeros(n_blocks, bool)
        is_mask[list(is_blocks)] = True
        sr_is = _sr_from_sums(s1[is_mask].sum(0), s2[is_mask].sum(0), size * is_mask.sum())
        sr_oos = _sr_from_sums(s1[~is_mask].sum(0), s2[~is_mask].sum(0), size * (~is_mask).sum())
        best = int(np.argmax(sr_is))
        rank = stats.rankdata(sr_oos)[best]          # 1 = worst ... N = best
        w = rank / (N + 1)
        logits.append(np.log(w / (1 - w)))
        degr.append((sr_is[best], sr_oos[best]))
    logits = np.array(logits)
    d = np.array(degr)
    return {"pbo": float((logits <= 0).mean()), "n_splits": len(logits),
            "median_logit": float(np.median(logits)),
            "is_best_sr_mean": float(d[:, 0].mean()), "oos_of_is_best_sr_mean": float(d[:, 1].mean())}


def _sr_from_sums(s1, s2, n):
    mean = s1 / n
    var = (s2 - n * mean ** 2) / (n - 1)
    return np.where(var > 0, mean / np.sqrt(np.maximum(var, 1e-300)), 0.0)


def stationary_bootstrap_indices(T: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    idx = np.empty(T, int)
    idx[0] = rng.integers(T)
    p = 1.0 / mean_block
    jumps = rng.random(T) < p
    starts = rng.integers(T, size=T)
    for t in range(1, T):
        idx[t] = starts[t] if jumps[t] else (idx[t - 1] + 1) % T
    return idx


def paired_bootstrap(strategy: pd.Series, benchmark: pd.Series, ppy: int, n: int = 2000,
                     mean_block: float = 20, seed: int = 7) -> dict:
    """Bootstrap the strategy and benchmark on the SAME resampled days. Returns percentiles and
    P(strategy Sharpe > benchmark Sharpe)."""
    df = pd.concat([strategy, benchmark], axis=1).dropna().to_numpy()
    rng = np.random.default_rng(seed)
    T = len(df)
    sh_s, sh_b, cagr_s, mdd_s = [], [], [], []
    for _ in range(n):
        x = df[stationary_bootstrap_indices(T, mean_block, rng)]
        s, b = x[:, 0], x[:, 1]
        sh_s.append(s.mean() / s.std() * np.sqrt(ppy))
        sh_b.append(b.mean() / b.std() * np.sqrt(ppy))
        eq = np.cumprod(1 + s)
        cagr_s.append(eq[-1] ** (ppy / T) - 1)
        mdd_s.append((eq / np.maximum.accumulate(eq) - 1).min())
    sh_s, sh_b = np.array(sh_s), np.array(sh_b)
    q = lambda a: {f"p{p}": float(np.percentile(a, p)) for p in (5, 50, 95)}  # noqa: E731
    return {"p_sharpe_beats_benchmark": float((sh_s > sh_b).mean()), "sharpe": q(sh_s),
            "sharpe_diff": q(sh_s - sh_b), "cagr": q(cagr_s), "max_dd": q(mdd_s)}
