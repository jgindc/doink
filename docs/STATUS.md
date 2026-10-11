# Status

_Last updated: 2026-10-11 (session 1)_

## Done
- Environment inventory (`docs/ENVIRONMENT.md`).
- Repo structure, CLAUDE.md, blueprint, testing protocol, decision log.
- Crypto data pipeline: Coinbase daily (2015→), 1h and 4h; Binance.US / Kraken / Alpaca cross-checks; validation; manifest.
- Engine: holdout-guarded loader, causal indicators, next-open simulator with costs, no-trade band, FIFO tax lots,
  metrics, after-tax estimate, hard limits module.
- Research tools: walk-forward, Deflated Sharpe, PBO (CSCV), paired stationary bootstrap, random-strategy generators.
- Tests: accounting, next-bar timing, delay, tax netting, limits, holdout guard, DSR/PBO sanity, lookahead tests
  (truncation + wild future scrambling at 29 cut points; verified to catch a planted one-day peek).
- Pre-registered EXP-001 … EXP-006 (committed before running).

## In progress
- Running Phase 1 experiments.

## Next
- Phase 2 paper-trading harness (needs owner decisions: Alpaca paper keys, alert channel, where the scheduler runs).
