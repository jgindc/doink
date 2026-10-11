# CLAUDE.md — read this first (you have no memory between sessions)

## Mission
Build the most profitable swing-trading system that **survives honest out-of-sample testing after all costs**.
Markets: U.S. ETFs (later) and crypto (now), one engine, one scoreboard. Holding period: days to weeks.
Paper money only. Real money is the owner's decision, never ours.
If honest testing finds no edge, say so plainly — that is a valid result (fallback: AI-assisted long-term investing research).

## Non-negotiables
- **Never**: deploy smart contracts or tutorial code, send funds, place real-money orders, use leverage/shorting/
  derivatives without explicit owner approval, commit secrets, or let any AI path override hard risk limits.
- Spot, long-only, no leverage. Paper mode is the default; live needs a config flag **and** human confirmation.
- API keys: trade-only, never withdrawal. Keys live in `.env` (gitignored).
- **Pre-register** every experiment in `experiments/registry/` and **commit it before running**.
  `scripts/run_experiment.py` refuses to run an uncommitted or modified registry entry.
- **Count every trial** (`experiments/trials.csv`). Report Deflated Sharpe and PBO.
- The final holdout (on/after `splits.holdout_start` in `configs/crypto.yaml`) is **locked**. Opening it is logged in
  `experiments/holdout_log.md`, once per strategy version family, only after walk-forward pass + critic review.
- The builder never grades its own promotion candidates — an independent critic agent/session reviews them.
- Never tweak rules to fit specific recent losing trades. Never present a backtest as an expected live result.
- Always report: trade count, costs, number of trials, caveats, and benchmarks (BTC B&H, 50/50 BTC/ETH, BTC 200-day SMA).

## Owner communication style
Simple, concise, easy to understand. Leave nothing important out, never overstate, don't trade quality for brevity.

## Repo map
- `swingbot/data/` — fetchers (Coinbase primary; Binance.US/Kraken/Alpaca cross-checks), parquet cache, validation,
  manifest, holdout-guarded `Panel` loader.
- `swingbot/signals/` — causal indicators. `swingbot/strategies/` — sleeves + benchmarks.
- `swingbot/portfolio/` — sleeve combination, caps, vol targeting; `limits.py` = hard limits.
- `swingbot/execution/simulator.py` — next-bar-open execution, costs, no-trade band, FIFO tax lots.
- `swingbot/reporting/` — metrics, after-tax estimate. `swingbot/research/` — walk-forward, DSR, PBO, bootstrap, luck.
- `swingbot/ledger/`, `swingbot/scheduler/` — Phase 2 (paper trading), not built yet.
- `experiments/registry/` (pre-registrations) · `experiments/results/` · `experiments/trials.csv` · `experiments/holdout_log.md`
- `docs/BLUEPRINT.md`, `docs/TESTING_PROTOCOL.md`, `docs/DECISIONS.md` (dated log), `docs/STATUS.md` (done/next),
  `docs/ENVIRONMENT.md` (tools, endpoints).

## Commands
```
uv sync                                         # install
uv run pytest                                   # tests (incl. lookahead-bias tests)
uv run python scripts/fetch_crypto.py --interval 1d   # update daily data (1h is slow, also writes 4h)
uv run python scripts/run_experiment.py EXP-002 # run a committed pre-registration
```

## Working rules
- Keep `docs/STATUS.md` and `docs/DECISIONS.md` current at the end of every session.
- Small commits. Simple, auditable code over clever code. Fixed seeds.
- Signals use completed bars only; orders fill at the next bar's open. Any new signal code gets a lookahead test.
