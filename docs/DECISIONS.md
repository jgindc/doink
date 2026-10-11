# Decision log (dated, newest last)

## Inherited from the kickoff briefing (before 2026-10-11)
- Swing (days–weeks), not day trading: fewer trades → less lost to fees/slippage/taxes; LLMs too slow for intraday.
- Stocks/ETFs + crypto on one engine; crypto first (no keys needed).
- Two lanes (rule bot vs AI-decides), both raced against buy-and-hold.
- Spot only, long-only, no leverage. Shorting/leverage/options/futures need explicit owner approval.
- Out of scope: day trading, HFT, prediction markets, DeFi/MEV/arbitrage bots, copying social-media bots.
- Paper only until results earn trust; real money is the owner's decision.

## 2026-10-11 (session 1)
- **Primary crypto history source = Coinbase Exchange** (BTC from 2015-07-20, ETH 2016-05, LTC 2016-08, BCH 2017-12),
  not Binance.US. Why: longer history (covers the 2018 bear market), and Binance.US USD pairs are unreliable
  (585 missing days; up to 18% price dislocation in Jul-2023 after it lost USD banking). Coinbase matches Kraken
  within a median 1.4 bps and Alpaca (the paper venue) within 1.8 bps on daily closes.
- **Paper venue = Alpaca** (crypto + stocks in one account). Alpaca crypto market data works without keys
  (history from 2021-01-01). Until keys arrive, the simulator runs on public prices.
- **Crypto universe** = BTC, ETH, LTC, BCH, LINK, SOL, ADA, AVAX, DOGE, DOT (all quoted on Alpaca today). XRP excluded
  for now: it was not tradable on U.S. venues Jan-2021 → Jul-2023 (data gap). Survivorship bias acknowledged; every
  result is re-checked on the legacy set BTC/ETH/LTC/BCH (large and listed by early 2018).
- **Cost model**: Alpaca tier-1 taker fee 25 bps/side (docs.alpaca.markets/docs/crypto-fees; Coinbase Advanced's lowest
  tier is ~60 bps taker, so Alpaca is cheaper for small accounts). Slippage from Alpaca spreads measured 2026-10-11
  (one Sunday snapshot; BTC/ETH ~2 bps, SOL 9, alts 16–61): BTC/ETH 5, SOL 10, LTC/BCH 35, other alts 25 bps/side.
  Stress at 2×. Idle cash earns 0% (conservative).
- **Execution timing**: decide on daily close (00:00 UTC bars), fill at next bar's open; delay stress = +1 bar.
- **Locked holdout = 2024-10-01 → present** (~2 years). Research data ends 2024-09-30. Enforced in code.
- **Walk-forward**: 3-year train, 6-month test, 6-month step; config chosen by train-window Sharpe; switching costs charged.
- **Sleeve defaults declared a priori from the literature** (not from data): trend lookbacks 20/60/120/250 with
  return-sign votes; rotation lookbacks 30/60/90, top 2, weekly; dip RSI(2) < 10, close > SMA200, exit close > SMA5 or
  10-day time stop. The ensemble uses only these defaults (1 trial), so its result is not selected from a grid.
- **Ensemble**: fixed 1/3 budgets per sleeve, per-asset caps (BTC/ETH 50%, others 20%), vol target 40%/yr (scale
  down only). Not tuned. Risk-parity sleeve budgets and a regime overlay are future hypotheses.
- **No ATR stops in Phase 1** (keeps parameter count low); tested later as a separate hypothesis.
- **Tax assumptions** (placeholder until owner gives their bracket): short-term 35%, long-term 20%, taxes paid from the
  account at each year end, liquidation at the end of the test. Wash sales ignored for crypto.
- **Sharpe uses rf = 0** (consistent with 0% cash yield in the simulator).
- **Daily bars committed to git** (≈1.2 MB, reproducibility); 1h/4h bars gitignored (≈30 MB, re-downloadable);
  `data/manifest.json` records row counts and content hashes for every file.
- **Python 3.13 + uv**; dependencies kept minimal (numpy, pandas, scipy, pyarrow, pyyaml, requests, matplotlib, pytest).
  Custom vectorized-signal + event-loop simulator instead of vectorbt (auditable, one code path for live).
