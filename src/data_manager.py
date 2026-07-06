from __future__ import annotations

import csv
import numpy as np
import pandas as pd
from pathlib import Path

# Fundamental columns that our strategy expects
FUNDAMENTAL_COLUMNS = [
    "roe",
    "roa",
    "roic_proxy",
    "revenue_growth",
    "eps",
    "debt_to_equity",
    "pe",
    "pb",
    "gross_margin",
    "operating_margin",
    "net_margin",
    "eva_like_score",
]


def read_universe(path: str | Path, benchmark_symbol: str) -> list[str]:
    """Reads stock universe tickers from CSV and filters out benchmark."""
    frame = pd.read_csv(path)
    symbols = frame["symbol"].dropna().astype(str).str.strip().tolist()
    symbols = [s for s in symbols if s and s != benchmark_symbol]
    return sorted(set(symbols))


def download_ohlcv(
    symbols: list[str],
    start: str,
    end: str | None,
    cache_dir: str | Path | None = None,
    batch_size: int = 80,
    progress_callback: callable | None = None,
) -> dict[str, pd.DataFrame]:
    """Downloads price data from yfinance, using local caching if available."""
    import yfinance as yf

    if not symbols:
        return {}

    data: dict[str, pd.DataFrame] = {}
    missing = []
    cache_path = Path(cache_dir) if cache_dir else None

    if cache_path:
        cache_path.mkdir(parents=True, exist_ok=True)
        for symbol in symbols:
            cached = _read_cached_ohlcv(cache_path, symbol, start, end)
            if cached is None:
                missing.append(symbol)
            else:
                data[symbol] = cached
    else:
        missing = symbols

    if missing:
        print(f"Downloading {len(missing)} tickers from yfinance; {len(data)} loaded from cache.", flush=True)

    # Download in batches to avoid overwhelming connections
    batches = [missing[i : i + batch_size] for i in range(0, len(missing), batch_size)]
    for i, batch in enumerate(batches, 1):
        if progress_callback:
            try:
                progress_callback(i, len(batches))
            except Exception:
                pass
        print(f"Downloading batch {i}/{len(batches)} ({len(batch)} symbols)...", flush=True)
        raw = yf.download(
            tickers=batch,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
            group_by="ticker",
            threads=True,
        )

        if len(batch) == 1:
            frame = _clean_ohlcv(raw)
            if not frame.empty:
                data[batch[0]] = frame
                if cache_path:
                    _write_cached_ohlcv(cache_path, batch[0], frame)
            continue

        available = set(raw.columns.get_level_values(0))
        for symbol in batch:
            if symbol not in available:
                continue
            frame = _clean_ohlcv(raw[symbol])
            if not frame.empty:
                data[symbol] = frame
                if cache_path:
                    _write_cached_ohlcv(cache_path, symbol, frame)

    return data


def _clean_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """Cleans columns, handles split adjustments via Adj Close, and parses index."""
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    df = frame.copy()
    if isinstance(df.columns, pd.MultiIndex):
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    
    # Adjust OHLC for corporate actions if Adj Close is present
    if "Adj Close" in df.columns:
        ratio = df["Adj Close"] / df["Close"]
        for col in ("Open", "High", "Low", "Close"):
            df[col] = df[col] * ratio
            
    needed = ["Open", "High", "Low", "Close", "Volume"]
    if not set(needed).issubset(df.columns):
        return pd.DataFrame(columns=needed)
        
    df = df[needed].dropna(subset=["Open", "Close"])
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def _cache_file(cache_dir: Path, symbol: str) -> Path:
    return cache_dir / f"{symbol.replace('.', '_')}.csv"


def _read_cached_ohlcv(cache_dir: Path, symbol: str, start: str, end: str | None) -> pd.DataFrame | None:
    path = _cache_file(cache_dir, symbol)
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
        frame.index = pd.to_datetime(frame.index).tz_localize(None).as_unit("ns")
        if frame.empty:
            return None
            
        start_ts = pd.Timestamp(start).as_unit("ns")
        end_ts = pd.Timestamp(end) if end else None

        # If we need the latest data and cache is very old, force re-download
        if end_ts is not None and frame.index.max() < end_ts - pd.Timedelta(days=10):
            return None
        
        if end_ts is None:
            return frame[frame.index >= start_ts]
        else:
            return frame[(frame.index >= start_ts) & (frame.index < end_ts)]
    except Exception:
        return None


def _write_cached_ohlcv(cache_dir: Path, symbol: str, frame: pd.DataFrame) -> None:
    output = frame.copy()
    output.index.name = "Date"
    output.to_csv(_cache_file(cache_dir, symbol))


def load_fundamentals(path: str | Path | None) -> pd.DataFrame:
    """Loads quarterly fundamentals file, parses dates, and cleans columns."""
    if not path or not Path(path).exists():
        return pd.DataFrame(columns=["symbol", "as_of_date", *FUNDAMENTAL_COLUMNS])
    df = pd.read_csv(path)
    df["symbol"] = df["symbol"].astype(str)
    df["as_of_date"] = pd.to_datetime(df["as_of_date"]).dt.tz_localize(None).dt.as_unit("ns")
    for col in FUNDAMENTAL_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values(["symbol", "as_of_date"])
