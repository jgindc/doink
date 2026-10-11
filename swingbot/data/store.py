"""Local parquet cache, validation and dataset manifest.

Layout: data/raw/<source>/<SYMBOL>_<interval>.parquet  (index = bar-open time, UTC)
Manifest: data/manifest.json records rows, date range and a content hash per file, so every
experiment result can name the exact dataset it used.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from swingbot.config import DATA_DIR

RAW = DATA_DIR / "raw"
MANIFEST = DATA_DIR / "manifest.json"
FREQ = {"1h": pd.Timedelta(hours=1), "4h": pd.Timedelta(hours=4), "1d": pd.Timedelta(days=1)}


def path_for(source: str, symbol: str, interval: str) -> Path:
    return RAW / source / f"{symbol}_{interval}.parquet"


def read_bars(source: str, symbol: str, interval: str) -> pd.DataFrame:
    return pd.read_parquet(path_for(source, symbol, interval))


def write_bars(df: pd.DataFrame, source: str, symbol: str, interval: str) -> Path:
    p = path_for(source, symbol, interval)
    p.parent.mkdir(parents=True, exist_ok=True)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.to_parquet(p)
    return p


def merge_bars(old: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    if old is None or old.empty:
        return new.sort_index()
    out = pd.concat([old, new])
    return out[~out.index.duplicated(keep="last")].sort_index()


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Aggregate bars (e.g. 1h -> 4h). Buckets are UTC-aligned; incomplete trailing bucket is dropped."""
    agg = df.resample(rule, origin="epoch", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    counts = df["close"].resample(rule, origin="epoch", label="left", closed="left").count()
    expected = int(pd.Timedelta(rule) / (df.index[1] - df.index[0])) if len(df) > 1 else 1
    agg = agg[counts > 0].dropna(subset=["close"])
    # drop the final bucket if it is not yet complete
    if len(agg) and counts.iloc[-1] < expected:
        agg = agg.iloc[:-1]
    return agg


def validate_bars(df: pd.DataFrame, interval: str) -> dict:
    """Data-quality report. Does not modify data."""
    rep: dict = {"rows": int(len(df))}
    if df.empty:
        return rep
    idx = df.index
    rep["first"] = str(idx[0])
    rep["last"] = str(idx[-1])
    rep["tz_utc"] = str(idx.tz) == "UTC"
    rep["monotonic"] = bool(idx.is_monotonic_increasing)
    rep["duplicates"] = int(idx.duplicated().sum())
    step = FREQ[interval]
    full = pd.date_range(idx[0], idx[-1], freq=step)
    rep["missing_bars"] = int(len(full.difference(idx)))
    rep["nonpositive_prices"] = int((df[["open", "high", "low", "close"]] <= 0).sum().sum())
    bad_ohlc = (df["low"] > df[["open", "close"]].min(axis=1) + 1e-12) | \
               (df["high"] < df[["open", "close"]].max(axis=1) - 1e-12)
    rep["ohlc_inconsistent"] = int(bad_ohlc.sum())
    r = np.log(df["close"]).diff().abs()
    thresh = {"1h": 0.15, "4h": 0.25, "1d": 0.40}[interval]
    big = r[r > thresh]
    rep["extreme_moves"] = [(str(t), round(float(v), 3)) for t, v in big.items()][:20]
    rep["zero_volume_bars"] = int((df["volume"] <= 0).sum())
    return rep


def file_hash(p: Path) -> str:
    """Hash of the bar values (not the file bytes), so it is stable across parquet writer versions."""
    df = pd.read_parquet(p)
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    return h.hexdigest()[:16]


def update_manifest(entries: dict[str, dict]) -> dict:
    man = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {"files": {}}
    man["files"].update(entries)
    man["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    h = hashlib.sha256(json.dumps({k: v.get("hash") for k, v in sorted(man["files"].items())}).encode())
    man["dataset_version"] = h.hexdigest()[:12]
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(man, indent=2, sort_keys=True))
    return man


def dataset_version() -> str:
    return json.loads(MANIFEST.read_text())["dataset_version"] if MANIFEST.exists() else "none"
