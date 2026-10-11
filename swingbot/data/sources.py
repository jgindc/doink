"""Public market-data clients (no API keys needed).

All functions return a DataFrame indexed by bar-open time (UTC, tz-aware) with columns
open, high, low, close, volume. Only *completed* bars are returned.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

USER_AGENT = "swingbot-research/0.1 (paper-trading research)"
COLUMNS = ["open", "high", "low", "close", "volume"]


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def _get_json(session: requests.Session, url: str, params: dict | None = None, retries: int = 5):
    delay = 1.0
    for attempt in range(retries):
        try:
            r = session.get(url, params=params, timeout=30)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"{r.status_code} from {url}")
            r.raise_for_status()
            return r.json()
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError):
            if attempt == retries - 1:
                raise
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def _finalize(rows: list[list], now: datetime, bar: timedelta) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=COLUMNS, index=pd.DatetimeIndex([], tz="UTC", name="ts"))
    df = pd.DataFrame(rows, columns=["ts"] + COLUMNS)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.drop_duplicates("ts").set_index("ts").sort_index().astype(float)
    # drop the still-forming bar
    df = df[df.index + bar <= pd.Timestamp(now)]
    return df


# ---------------------------------------------------------------- Coinbase Exchange
COINBASE_URL = "https://api.exchange.coinbase.com/products/{product}/candles"
COINBASE_GRANULARITY = {"1h": 3600, "6h": 21600, "1d": 86400}


def fetch_coinbase(product: str, interval: str, start: datetime, end: datetime | None = None,
                   pause: float = 0.15) -> pd.DataFrame:
    """Page through Coinbase candles (max 300 per request). Row format is [time, low, high, open, close, volume]."""
    gran = COINBASE_GRANULARITY[interval]
    now = datetime.now(timezone.utc)
    end = end or now
    step = timedelta(seconds=gran * 300)
    s = _session()
    rows: list[list] = []
    t0 = start
    while t0 < end:
        t1 = min(t0 + step, end)
        data = _get_json(s, COINBASE_URL.format(product=product),
                         {"granularity": gran, "start": t0.isoformat(), "end": t1.isoformat()})
        if isinstance(data, dict):  # error payload
            raise RuntimeError(f"Coinbase error for {product}: {data}")
        for t, lo, hi, op, cl, vol in data:
            rows.append([pd.Timestamp(t, unit="s", tz="UTC"), op, hi, lo, cl, vol])
        t0 = t1
        time.sleep(pause)
    return _finalize(rows, now, timedelta(seconds=gran))


# ---------------------------------------------------------------- Binance.US (cross-check)
BINANCE_US_URL = "https://api.binance.us/api/v3/klines"
BINANCE_INTERVAL = {"1h": timedelta(hours=1), "4h": timedelta(hours=4), "1d": timedelta(days=1)}


def fetch_binance_us(symbol: str, interval: str, start: datetime, pause: float = 0.2) -> pd.DataFrame:
    """Page forward through Binance.US klines (max 1000 per request)."""
    now = datetime.now(timezone.utc)
    bar = BINANCE_INTERVAL[interval]
    s = _session()
    rows: list[list] = []
    start_ms = int(start.timestamp() * 1000)
    while True:
        data = _get_json(s, BINANCE_US_URL,
                         {"symbol": symbol, "interval": interval, "startTime": start_ms, "limit": 1000})
        if not data:
            break
        for k in data:
            rows.append([pd.Timestamp(k[0], unit="ms", tz="UTC"), k[1], k[2], k[3], k[4], k[5]])
        if len(data) < 1000:
            break
        start_ms = data[-1][0] + 1
        time.sleep(pause)
    return _finalize(rows, now, bar)


# ---------------------------------------------------------------- Kraken (cross-check, last 720 bars only)
KRAKEN_URL = "https://api.kraken.com/0/public/OHLC"


def fetch_kraken(pair: str, interval_minutes: int = 1440) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    data = _get_json(_session(), KRAKEN_URL, {"pair": pair, "interval": interval_minutes})
    if data.get("error"):
        raise RuntimeError(f"Kraken error: {data['error']}")
    key = next(k for k in data["result"] if k != "last")
    rows = [[pd.Timestamp(int(k[0]), unit="s", tz="UTC"), k[1], k[2], k[3], k[4], k[6]] for k in data["result"][key]]
    return _finalize(rows, now, timedelta(minutes=interval_minutes))


# ---------------------------------------------------------------- Alpaca crypto (paper venue; no key for data)
ALPACA_CRYPTO_URL = "https://data.alpaca.markets/v1beta3/crypto/us/bars"


def fetch_alpaca_crypto(symbol: str, timeframe: str, start: datetime, pause: float = 0.3) -> pd.DataFrame:
    """symbol like 'BTC/USD', timeframe like '1Day' or '1Hour'. History starts 2021-01-01."""
    now = datetime.now(timezone.utc)
    bar = {"1Day": timedelta(days=1), "1Hour": timedelta(hours=1), "4Hour": timedelta(hours=4)}[timeframe]
    s = _session()
    rows: list[list] = []
    params = {"symbols": symbol, "timeframe": timeframe, "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
              "limit": 10000}
    while True:
        data = _get_json(s, ALPACA_CRYPTO_URL, params)
        for b in data.get("bars", {}).get(symbol, []):
            rows.append([pd.Timestamp(b["t"]), b["o"], b["h"], b["l"], b["c"], b["v"]])
        token = data.get("next_page_token")
        if not token:
            break
        params["page_token"] = token
        time.sleep(pause)
    return _finalize(rows, now, bar)
