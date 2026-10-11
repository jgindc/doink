# Blueprint

## Goal
Best profit that holds up out-of-sample after costs (not best backtest). Benchmarks: buy-and-hold and simple trend.
Realistic bar: for ETFs, match/beat buy-and-hold with much smaller drawdowns. Crypto: bigger numbers, bigger swings.
A backtest showing 100%+/year is a red flag to investigate, not a win.

## Two lanes, raced against benchmarks
1. **Rule bot** — AI designs/tests rules; the rules trade.
2. **AI-decides bot** — an LLM reads a compact point-in-time snapshot each run and outputs strict JSON target weights
   (+ confidence, short rationale), inside the same hard limits. Judged on **live paper results only** (LLMs know
   historical outcomes, so backtesting them is contaminated). Every prompt/output logged.

## Universe
- Crypto (Phase 1): BTC, ETH, LTC, BCH, LINK, SOL, ADA, AVAX, DOGE, DOT — spot, all tradable on Alpaca.
  Chosen with hindsight → survivorship bias; every result is re-checked on the "legacy" set (BTC, ETH, LTC, BCH).
- ETFs (Phase 3): SPY QQQ IWM · XLK XLF XLE XLV XLI XLY XLP XLU XLB XLRE XLC · EFA EEM · TLT IEF SHY · GLD · (DBC).
  Total-return (split/dividend-adjusted) data.
- Single stocks: only later, only with survivorship-bias-free data.

## Sleeves (simple, evidence-based, combined as an ensemble)
| Sleeve | Rule | Evidence |
|---|---|---|
| Trend (TSMOM) | Per asset: fraction of lookbacks (20/60/120/250d) with positive return → fills that asset's inverse-vol slot; rest cash | Moskowitz-Ooi-Pedersen 2012; Liu-Tsyvinski 2021 (crypto) |
| Momentum rotation | Rank by risk-adjusted momentum (blend of lookbacks); hold top N with positive momentum; weekly | Jegadeesh-Titman 1993; Liu-Tsyvinski-Wu 2022 (crypto) |
| Dip-buy in uptrend | RSI(2) oversold while close > SMA(200); exit close > SMA(5) or time stop | Short-term reversal; weaker evidence in crypto |
| Earnings drift (stocks, later) | Drift after big earnings surprises; LLM reads reports | Bernard-Thomas 1989 |
| AI-decides | See above | Forward test only |

Ensemble: fixed sleeve budgets (1/3 each, a priori), per-asset caps, portfolio vol target (scale down only, never lever).

## Sizing & risk
- Inverse-volatility sizing inside sleeves. Portfolio vol target (crypto book 40%/yr; ETF book 10–15%/yr).
- Caps: per position, per sleeve, total crypto allocation. Regime overlay (cut exposure in vol spikes / market
  downtrend) is a separate hypothesis to test, not baked in.
- Kelly only as an upper-bound sanity check. Exits: trend reversal; time stop for dip-buys; ATR stops = future test.
- No-trade band (2% of equity) to cut turnover. Exits always execute.

## Hard limits (code, outside every strategy/AI path) — `swingbot/portfolio/limits.py`
Max daily loss 5%, max drawdown 25% → halt + alert owner. Max position 50%, max order 30% of equity, allowed symbols
only. Paper default; live requires config flag + human confirmation. No withdrawal permissions on keys.

## Costs & execution
Signals on bar close → fill at **next bar open**. Crypto: Alpaca tier-1 taker 25 bps + per-asset slippage
(BTC/ETH 5, SOL 10, LTC/BCH 35, others 25 bps). ETFs: commission-free + 2–5 bps. Stress: 2× costs and +1 bar delay.

## Architecture
`data/` · `signals/` · `strategies/` · `portfolio/` · `execution/` (simulator now; Alpaca paper adapter in Phase 2) ·
`ledger/` (SQLite, Phase 2) · `reporting/` · `scheduler/` (Phase 2) · `research/` · `experiments/`.
**One code path**: strategies produce target weights from data ≤ t; backtest and live both take the latest row and
send it through the same limits → broker (simulated or Alpaca paper).

## Phases
1. Crypto research (no keys): data, backtester, benchmarks, 3 sleeves, walk-forward, luck calibration. ← now
2. Paper-trading harness: Alpaca crypto paper (or simulated broker on live prices), scheduler, ledger, dashboard, alerts.
3. ETF book on Alpaca data through the same pipeline.
4. AI-decides lane in live paper, raced against rule bot and buy-and-hold.
5. Monthly improvement loop; owner decides on tiny real money.

## Taxes (owner is a U.S. taxpayer)
<1 year gains = ordinary income; >1 year = long-term rates. Minimize turnover, track lots, report an estimated
after-tax result (taxable account) next to pre-tax. ETFs may later run in an IRA/Roth; crypto likely taxable.
