# Testing protocol (non-negotiable)

1. **Pre-register** in `experiments/registry/EXP-XXX-*.yaml` *and commit* before running: hypothesis, grid, data
   periods, pass/fail gates, default (a-priori) config. The runner refuses uncommitted/modified registry files.
2. **Splits** (crypto): research data 2015-07-20 → 2024-09-30 (simulations start 2016-01-01; earlier data = indicator
   warm-up). **Locked holdout: 2024-10-01 → today**, opened once per strategy version family, logged.
   Walk-forward inside research data: train 3y → test next 6m → step 6m. OOS ≈ 2018-12-31 → 2024-09-30.
3. **Robustness**: plateau across neighboring parameters; multiple assets and sub-periods; legacy-universe check
   (survivorship); 2× costs; +1 bar delay; stationary block bootstrap.
4. **Count every trial** in `experiments/trials.csv`. Report Deflated Sharpe (cumulative trials) and PBO (CSCV).
   Stress/robustness re-runs of an already-counted config are not new trials; any new config/variant looked at is.
5. **Luck calibration**: thousands of random strategies through the same pipeline; record how many "pass".
6. **Benchmarks**: crypto — BTC buy-and-hold, 50/50 BTC/ETH (monthly rebalanced), BTC 200-day SMA trend.
   ETFs — SPY, 60/40, SPY 200-day SMA. If a complex strategy can't beat the simple baseline, complexity isn't paying.
7. **Metrics**: CAGR, vol, Sharpe, Sortino, max DD, Calmar, turnover, cost drag, time in market, trades, win rate,
   profit factor, avg hold, estimated after-tax CAGR. Sharpe uses rf = 0 (idle cash earns 0 in the simulator).
8. **Minimum evidence**: ≥30 trades per sleeve (prefer 100+); paper trading ≥2–3 months within the backtest's range.
9. **Independent critic** (separate agent/session) reviews every promotion candidate for lookahead, survivorship, data
   errors, overfitting. The builder never grades its own work.
10. **Promotion gates**: idea → backtest → walk-forward → holdout → paper (2–3 months, beats benchmarks) → owner
    decides on tiny real money with hard limits → gradual scaling.
11. **Live paper kill rules** (set before launch): halt a sleeve and alert if equity falls below its expected band
    (bootstrap 5th percentile path) or drawdown exceeds 1.5× backtest max DD, or rolling performance shows the edge gone.

## Walk-forward pass gates (Phase 1 crypto, evaluated on the OOS period)
| Gate | Rule |
|---|---|
| G1 Sharpe | OOS Sharpe > best of the three benchmarks' Sharpe over the same dates |
| G2 Drawdown | OOS max DD ≤ 0.75 × BTC buy-and-hold max DD |
| G3 Trades | ≥ 30 closed trades (position episodes) in OOS |
| G4 Costs | at 2× costs, OOS Sharpe still > best benchmark Sharpe |
| G5 Delay | with +1 bar delay, OOS Sharpe still > best benchmark Sharpe |
| G6 Plateau | median OOS Sharpe across all grid configs > BTC buy-and-hold Sharpe (grids only) |
| G7 DSR | Deflated Sharpe ≥ 0.95 using the cumulative project trial count |
| G8 PBO | Probability of Backtest Overfitting < 0.5 (grids only) |
| G9 Survivorship | same rule on legacy universe (BTC, ETH, LTC, BCH) still beats BTC buy-and-hold Sharpe |
Informational: paired bootstrap P(Sharpe > BTC B&H), after-tax CAGR vs B&H after-tax, calendar-year table.
