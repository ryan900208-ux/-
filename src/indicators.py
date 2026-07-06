from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Calculates Relative Strength Index using standard Wilder/rolling mean method."""
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Adds standard technical indicators to stock dataframe."""
    df = frame.copy()
    close = df["Close"]
    volume = df["Volume"]

    for w in (5, 20, 60, 120):
        df[f"ret{w}"] = close.pct_change(w)
        df[f"ma{w}"] = close.rolling(w).mean()

    df["ma20_slope"] = df["ma20"].diff(5) / df["ma20"].shift(5)
    df["rsi14"] = rsi(close, 14)
    df["volume_ma20"] = volume.rolling(20).mean()
    df["volume_ratio"] = volume / df["volume_ma20"]
    df["ma20_deviation"] = (close / df["ma20"]) - 1
    return df


def market_regime(benchmark: pd.DataFrame, config: dict) -> pd.Series:
    """Calculates daily market regime (bull, bear, neutral) based on benchmark index."""
    reg_cfg = config["market_regime"]
    df = add_indicators(benchmark)
    
    ma_slow_col = f"ma{reg_cfg['bear_close_below_ma']}"
    ma_fast_col = f"ma{reg_cfg['bear_ma_fast']}"
    ma_mid_col = f"ma{reg_cfg['bear_ma_slow']}"

    bear = (df["Close"] < df[ma_slow_col]) & (df[ma_fast_col] < df[ma_mid_col])
    bull = (df["Close"] >= df[ma_slow_col]) & (df[ma_fast_col] >= df[ma_mid_col])

    out = pd.Series("neutral", index=df.index, name="market_regime")
    out[bear] = "bear"
    out[bull] = "bull"
    return out
