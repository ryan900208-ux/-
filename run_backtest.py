from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

# Add the project root to PYTHONPATH
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from data_manager import download_ohlcv, load_fundamentals, read_universe
from indicators import add_indicators, market_regime
from strategy import add_strategy_scores, build_feature_panel
from backtester import run_backtest
from reporter import summarize


def main() -> None:
    parser = argparse.ArgumentParser(description="Quality Momentum Trading Backtester")
    parser.add_argument("--config", default="config.json", help="Path to config JSON file")
    args = parser.parse_args()

    config_path = ROOT / args.config
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    benchmark_symbol = config["benchmark_symbol"]
    universe_path = ROOT / config["universe_csv"]
    
    # 1. Load Stock Universe
    symbols = read_universe(universe_path, benchmark_symbol)
    all_symbols = sorted(set(symbols + [benchmark_symbol]))

    # 2. Download and Process Prices
    print(f"Loading/Downloading OHLCV for {len(all_symbols)} tickers...")
    raw_data = download_ohlcv(
        symbols=all_symbols,
        start=config["start"],
        end=config["end"],
        cache_dir=ROOT / config["price_cache_dir"],
        batch_size=config.get("yfinance_batch_size", 80),
    )

    if benchmark_symbol not in raw_data:
        print(f"Error: Benchmark {benchmark_symbol} price data is missing.")
        sys.exit(1)

    benchmark = raw_data.pop(benchmark_symbol)
    data = {sym: add_indicators(df) for sym, df in raw_data.items() if not df.empty}
    benchmark = add_indicators(benchmark)

    # 3. Calculate Market Regime
    regime = market_regime(benchmark, config)

    # 4. Load Fundamentals
    fundamentals_path = ROOT / config["fundamentals_csv"]
    fundamentals = load_fundamentals(fundamentals_path)

    # 5. Build Feature Panel and strategy scores
    print("Building feature panel and calculating strategy scores...")
    panel = build_feature_panel(data, benchmark, regime)
    if panel.empty:
        print("Error: Combined feature panel is empty. Check price data dates.")
        sys.exit(1)

    panel = add_strategy_scores(panel, fundamentals, config)

    # 6. Run Portfolio Backtest
    print("Running portfolio backtest...")
    equity, trades = run_backtest(panel, data, config)

    # 7. Generate Performance Summary
    summary = summarize(equity, trades)

    # 8. Save Outputs
    output_dir = ROOT / "outputs" / "backtest"
    output_dir.mkdir(parents=True, exist_ok=True)

    panel.to_csv(output_dir / "daily_features.csv", index=False)
    equity.to_csv(output_dir / "equity_curve.csv", index=False)
    trades.to_csv(output_dir / "trades.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)

    # 9. Output results to console
    print("\n" + "=" * 40)
    print("           BACKTEST PERFORMANCE")
    print("=" * 40)
    print(summary.to_string(index=False))
    print("=" * 40)
    print(f"Saved all backtest output files to: {output_dir}\n")


if __name__ == "__main__":
    main()
