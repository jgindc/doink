"""Download / update crypto bars into the local parquet cache and refresh the manifest.

Usage:
  uv run python scripts/fetch_crypto.py --interval 1d          # daily, all universe coins (Coinbase)
  uv run python scripts/fetch_crypto.py --interval 1h          # hourly (slow: ~300 requests per coin); also writes 4h
  uv run python scripts/fetch_crypto.py --crosscheck           # Binance.US + Kraken + Alpaca daily for BTC/ETH
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone

import pandas as pd

from swingbot.config import load_config
from swingbot.data import sources, store


def update_coinbase(symbol: str, interval: str, history_start: datetime) -> dict:
    product = f"{symbol}-USD"
    p = store.path_for("coinbase", symbol, interval)
    old = pd.read_parquet(p) if p.exists() else None
    if old is not None and len(old):
        start = old.index[-1].to_pydatetime() - timedelta(days=2)  # small overlap, then de-duplicate
    elif interval == "1d":
        start = history_start
    else:
        # begin hourly history at the coin's first daily bar to avoid thousands of empty requests
        daily = store.read_bars("coinbase", symbol, "1d")
        start = daily.index[0].to_pydatetime()
    new = sources.fetch_coinbase(product, interval, start)
    df = store.merge_bars(old, new)
    store.write_bars(df, "coinbase", symbol, interval)
    entries = {f"coinbase/{symbol}_{interval}": _entry(df, "coinbase", symbol, interval)}
    if interval == "1h":
        df4 = store.resample_ohlcv(df, "4h")
        store.write_bars(df4, "coinbase", symbol, "4h")
        entries[f"coinbase/{symbol}_4h"] = _entry(df4, "coinbase", symbol, "4h")
    return entries


def _entry(df: pd.DataFrame, source: str, symbol: str, interval: str) -> dict:
    rep = store.validate_bars(df, interval)
    return {"hash": store.file_hash(store.path_for(source, symbol, interval)),
            "rows": rep["rows"], "first": rep.get("first"), "last": rep.get("last"),
            "missing_bars": rep.get("missing_bars"), "duplicates": rep.get("duplicates"),
            "ohlc_inconsistent": rep.get("ohlc_inconsistent"), "extreme_moves": rep.get("extreme_moves"),
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def crosscheck() -> dict:
    """Fetch the same daily bars from other venues so the primary source can be checked."""
    entries = {}
    jobs = [
        ("binance_us", "BTC", lambda: sources.fetch_binance_us("BTCUSD", "1d", datetime(2019, 9, 1, tzinfo=timezone.utc))),
        ("binance_us", "ETH", lambda: sources.fetch_binance_us("ETHUSD", "1d", datetime(2019, 9, 1, tzinfo=timezone.utc))),
        ("kraken", "BTC", lambda: sources.fetch_kraken("XBTUSD")),
        ("kraken", "ETH", lambda: sources.fetch_kraken("ETHUSD")),
        ("alpaca", "BTC", lambda: sources.fetch_alpaca_crypto("BTC/USD", "1Day", datetime(2021, 1, 1, tzinfo=timezone.utc))),
        ("alpaca", "ETH", lambda: sources.fetch_alpaca_crypto("ETH/USD", "1Day", datetime(2021, 1, 1, tzinfo=timezone.utc))),
    ]
    for src, sym, fn in jobs:
        df = fn()
        store.write_bars(df, src, sym, "1d")
        entries[f"{src}/{sym}_1d"] = _entry(df, src, sym, "1d")
        print(f"{src:10s} {sym}: {len(df)} bars {df.index[0].date()} -> {df.index[-1].date()}")
    return entries


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", choices=["1d", "1h"])
    ap.add_argument("--symbols", nargs="*")
    ap.add_argument("--crosscheck", action="store_true")
    args = ap.parse_args()
    cfg = load_config("crypto")
    entries: dict = {}
    if args.interval:
        start = datetime.fromisoformat(cfg["data"]["history_start"]).replace(tzinfo=timezone.utc)
        for sym in args.symbols or cfg["data"]["universe"]:
            e = update_coinbase(sym, args.interval, start)
            entries.update(e)
            for k, v in e.items():
                print(f"{k:24s} rows={v['rows']:>7} {v['first'][:10]} -> {v['last'][:10]} "
                      f"missing={v['missing_bars']} dup={v['duplicates']} badOHLC={v['ohlc_inconsistent']} "
                      f"extreme={len(v['extreme_moves'])}")
            store.update_manifest(e)  # save progress per symbol
    if args.crosscheck:
        entries.update(crosscheck())
        store.update_manifest(entries)
    print(json.dumps({"dataset_version": store.dataset_version()}))


if __name__ == "__main__":
    main()
