# Environment inventory (2026-10-11, Claude Code cloud session)

## Machine
| Item | Value |
|---|---|
| OS | Ubuntu 24.04.5 LTS (container, **ephemeral** — anything not pushed to git is lost) |
| CPU / RAM / disk | 4 cores / 15 GB / ~30 GB free |
| Python / package mgr | Python 3.13, uv 0.11 (pip also present) |
| CLIs | git, gh (but GitHub access goes through the GitHub MCP tools), node 22, systemctl |
| Network | outbound HTTPS via an egress proxy; PyPI reachable |

## Scheduling options
| Option | Works? | Notes |
|---|---|---|
| cron/systemd in this container | technically, but useless | container is reclaimed when idle |
| Session cron (CronCreate) | only while this session lives | not durable |
| Claude Code Routines (cloud triggers) | yes, hourly minimum | each run is a fresh session; state must be committed to git |
| GitHub Actions scheduled workflow | yes (needs owner OK) | free for this public repo; ledger committed back to repo |
| Owner's always-on machine (e.g. the Raspberry Pi) | best | cron + `.env` with Alpaca paper keys; owner controls it |

## Data endpoints (reachability tested 2026-10-11)
| Endpoint | Status | Notes |
|---|---|---|
| Coinbase Exchange candles | ✅ | BTC daily from 2015-07-20; 300 bars/request; 1h/6h/1d (no 4h → built from 1h) |
| Binance.US klines | ✅ | from 2019-09; USD pairs have big gaps/dislocations (see DECISIONS) |
| Binance.com | ❌ 451 | geo-restricted |
| Kraken OHLC | ✅ | only last 720 bars via public API |
| Gemini candles | ✅ | also via Gemini MCP |
| Alpaca crypto data | ✅ without keys | history from 2021-01-01; latest quotes/spreads available |
| Alpaca stock data / paper trading API | 🔒 401 | needs owner's **paper** keys |
| Alpha Vantage | ✅ | demo key works on REST; MCP server connected |
| Yahoo Finance chart API | ✅ | fallback only |
| SEC EDGAR | ✅ | needs User-Agent header |
| FRED (T-bill yields) | ✅ | |
| Stooq | ❌ | connection reset |
| CoinGecko | ✅ | |

## MCP servers / plugins available to the agent
| Tool | Status | Use |
|---|---|---|
| Alpha Vantage MCP | ✅ connected | ETF/stock data, earnings, news sentiment (Phase 3–4) |
| Gemini MCP (read-only) | ✅ public data | crypto price cross-checks |
| TextToQuant | ✅ free tier, 5 runs lifetime, crypto, ≤3y | reserve for an independent check of a finalist |
| StockPortfolio.pro | ✅ | SEC filing data (earnings-drift sleeve later) |
| TradingCalc | ✅ | Deflated Sharpe / Kelly cross-checks of our own numbers |
| QuanterLab plugin | 🔒 needs owner to authorize in claude.ai connector settings | ETF walk-forward cross-checks |
| GitHub MCP | ✅ scoped to jgindc/doink | repo is **public** |
