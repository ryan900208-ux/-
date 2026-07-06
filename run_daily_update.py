from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

# Add src folder to PYTHONPATH
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from data_manager import download_ohlcv, load_fundamentals, read_universe
from indicators import add_indicators, market_regime
from strategy import add_strategy_scores, build_feature_panel, entry_candidates

# Path Configuration
STATE_PATH = ROOT / "docs" / "outputs" / "paper_trading" / "state.json"
TRADES_PATH = ROOT / "docs" / "outputs" / "paper_trading" / "trades.csv"
SNAPSHOTS_PATH = ROOT / "docs" / "outputs" / "paper_trading" / "daily_snapshots.csv"
SIGNALS_DIR = ROOT / "docs" / "outputs" / "paper_trading" / "signals"

INITIAL_CASH = 1000000.0
MAX_POSITIONS = 5
POSITION_WEIGHT = 0.20
STOP_LOSS = 0.12
COMMISSION_RATE = 0.001425
TAX_RATE = 0.003
SLIPPAGE_RATE = 0.001
MAX_HOLDING_DAYS = 252


@dataclass
class Position:
    symbol: str
    name: str
    shares: int
    entry_signal_date: str
    entry_date: str
    entry_price: float
    entry_cost: float
    holding_bars: int = 0


def _write_progress(progress: int, message: str, seconds_left: int) -> None:
    # BUG-01: use same path as api_app.py reads (docs/outputs/...)
    path = ROOT / "docs" / "outputs" / "paper_trading" / "progress.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({
                "status": "running",
                "progress": progress,
                "message": message,
                "seconds_left": seconds_left
            }, handle, ensure_ascii=False, indent=2)
    except Exception:
        pass


def main() -> None:
    _write_progress(5, "Initializing daily update...", 22)
    # 1. Create Output Directories
    SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / "docs" / "outputs" / "paper_trading").mkdir(parents=True, exist_ok=True)

    # 2. Load Config
    with open(ROOT / "config.json", "r", encoding="utf-8") as handle:
        config = json.load(handle)

    benchmark_symbol = config["benchmark_symbol"]
    universe_path = ROOT / config["universe_csv"]
    symbols = read_universe(universe_path, benchmark_symbol)
    all_symbols = sorted(set(symbols + [benchmark_symbol]))

    # 3. Load/Download latest price data (Force refresh to get today's close)
    _write_progress(10, f"Preparing to download {len(all_symbols)} tickers...", 20)
    print("Downloading latest stock prices...")
    
    def download_progress(current_batch: int, total_batches: int) -> None:
        pct = int(10 + (current_batch / total_batches) * 75)  # 15% to 85%
        seconds_left = max(2, int((total_batches - current_batch) * 3.2))
        _write_progress(
            pct, 
            f"Downloading stock prices (batch {current_batch}/{total_batches})...", 
            seconds_left
        )

    raw_data = download_ohlcv(
        symbols=all_symbols,
        start=config["start"],
        end=None,
        cache_dir=None,  # Force download to fetch today's data
        batch_size=config.get("yfinance_batch_size", 10),
        progress_callback=download_progress,
    )

    # Save downloaded data back to cache
    # Save downloaded data back to cache
    _write_progress(85, "Saving downloaded price data to local database...", 3)
    cache_path = ROOT / config["price_cache_dir"]
    cache_path.mkdir(parents=True, exist_ok=True)
    for sym, frame in raw_data.items():
        frame.index.name = "Date"
        frame.to_csv(cache_path / f"{sym.replace('.', '_')}.csv")

    if benchmark_symbol not in raw_data:
        print(f"Error: Benchmark {benchmark_symbol} price data is missing.")
        sys.exit(1)

    benchmark = raw_data.pop(benchmark_symbol)
    data = {sym: add_indicators(df) for sym, df in raw_data.items() if not df.empty}
    benchmark = add_indicators(benchmark)

    # 4. Strategy calculations
    _write_progress(90, "Calculating momentum technical indicators...", 2)
    regime = market_regime(benchmark, config)
    fundamentals = load_fundamentals(ROOT / config["fundamentals_csv"])

    panel = build_feature_panel(data, benchmark, regime)
    panel = add_strategy_scores(panel, fundamentals, config)

    latest_date = panel["date"].max()
    print(f"Latest trading day in price data: {latest_date.date()}")

    # 5. Extract Candidates
    day_candidates = entry_candidates(panel, latest_date, config)
    day_candidates.to_csv(SIGNALS_DIR / f"candidates_{latest_date.date()}.csv", index=False)

    state = _load_state()

    # Check for manual sell override input from environment (GitHub Actions input)
    import os
    manual_sell_env = os.environ.get("MANUAL_SELL")
    if manual_sell_env:
        manual_sell_symbols = [s.strip() for s in manual_sell_env.split(",") if s.strip()]
        for sym in manual_sell_symbols:
            pos = _find_position(state, sym)
            if pos and not _has_pending_order(state, sym, "sell"):
                state["pending_orders"].append({
                    "type": "sell",
                    "symbol": sym,
                    "name": pos["name"],
                    "signal_date": str(latest_date.date()),
                    "reason": "manual",
                    "status": "pending_next_open"
                })
                print(f"Added manual sell order for {sym} from GitHub Actions input.")

    # 7. Execute Pending Orders (using today's Open price, which is now in the cache)
    _write_progress(95, "Running portfolio rebalancing logic...", 1)
    state = _execute_pending_orders(state)

    # 8. Check exit conditions on active positions
    state, exit_signals = _apply_exits(state, panel, latest_date)

    # 9. Create new Buy orders if free slots exist
    state = _create_buy_orders(state, day_candidates, latest_date)

    # 10. Record snapshot and save state
    snapshot = _snapshot(state, panel, latest_date, len(day_candidates), len(exit_signals))
    _append_snapshot(snapshot)
    
    # Enrich positions with current close and pnl for serverless frontend
    for pos in state.get("positions", []):
        close = _get_last_close(pos["symbol"])
        if close:
            pos["close"] = close
            pos["market_value"] = pos["shares"] * close
            pos["unrealized_pnl"] = pos["market_value"] - pos["entry_cost"]
            pos["unrealized_return_pct"] = (close / pos["entry_price"] - 1) * 100
        else:
            pos["close"] = pos["entry_price"]
            pos["market_value"] = pos["entry_cost"]
            pos["unrealized_pnl"] = 0.0
            pos["unrealized_return_pct"] = 0.0

    _save_state(state)
    _generate_static_json_payloads(state, latest_date, snapshot, panel)

    # Delete progress file upon successful completion
    try:
        # BUG-01: cleanup matches the correct path now
        progress_path = ROOT / "docs" / "outputs" / "paper_trading" / "progress.json"
        if progress_path.exists():
            progress_path.unlink()
    except Exception:
        pass

    # 11. Print Summary Report
    print("\n" + "=" * 40)
    print("        DAILY PORTFOLIO SUMMARY")
    print("=" * 40)
    print(f"Date:             {latest_date.date()}")
    print(f"Market Regime:    {regime.iloc[-1]}")
    print(f"Total Equity:     {snapshot['equity']:.2f} TWD")
    print(f"Cash:             {snapshot['cash']:.2f} TWD")
    print(f"Open Positions:   {snapshot['positions']}")
    print(f"Pending Orders:   {snapshot['pending_orders']}")
    print("=" * 40)
    if state["positions"]:
        print("\nActive Positions:")
        for pos in state["positions"]:
            close = _get_last_close(pos["symbol"])
            mv = pos["shares"] * close if close else 0.0
            pnl = mv - pos["entry_cost"]
            print(f" - {pos['symbol']} ({pos['name']}): {pos['shares']} shares, MV: {mv:.2f}, PnL: {pnl:.2f}")
    if state["pending_orders"]:
        print("\nPending Orders (Executing next Open):")
        for order in state["pending_orders"]:
            print(f" - {order['type'].upper()} {order['symbol']} ({order.get('reason', 'signal')})")
    print("=" * 40 + "\n")


def _load_state() -> dict:
    if STATE_PATH.exists():
        with STATE_PATH.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return {
        "model_version": "Quality Momentum Optimized",
        "initial_cash": INITIAL_CASH,
        "cash": INITIAL_CASH,
        "start_date": None,
        "last_signal_date": None,
        "positions": [],
        "pending_orders": [],
    }


def _save_state(state: dict) -> None:
    # BUG-06: atomic write via temp file to prevent corruption on crash mid-write
    import tempfile
    tmp_path = STATE_PATH.with_suffix(".tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(state, handle, ensure_ascii=False, indent=2)
        import os
        os.replace(tmp_path, STATE_PATH)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def _execute_pending_orders(state: dict) -> dict:
    remaining_orders = []
    for order in state.get("pending_orders", []):
        open_price, trade_date = _get_next_open(order["symbol"], order["signal_date"])
        if open_price is None or pd.isna(open_price):
            # Price not available yet (open of t+1 not in cache), keep pending
            remaining_orders.append(order)
            continue

        if order["type"] == "sell":
            pos = _find_position(state, order["symbol"])
            if pos:
                # Sell execution
                proceeds = pos["shares"] * open_price * (1 - SLIPPAGE_RATE)
                fee = proceeds * COMMISSION_RATE
                tax = proceeds * TAX_RATE
                cash_in = proceeds - fee - tax
                state["cash"] += cash_in

                # Record trade
                pnl = cash_in - pos["entry_cost"]
                trade = {
                    "symbol": pos["symbol"],
                    "name": pos["name"],
                    "entry_signal_date": pos["entry_signal_date"],
                    "entry_date": pos["entry_date"],
                    "exit_signal_date": order["signal_date"],
                    "exit_date": trade_date,
                    "entry_price": pos["entry_price"],
                    "exit_price": open_price,
                    "shares": pos["shares"],
                    "pnl": pnl,
                    "return_pct": (open_price / pos["entry_price"]) - 1,
                    "holding_bars": pos["holding_bars"],
                    "exit_reason": order.get("reason", "sell_signal"),
                }
                _append_trade(trade)
                
                # Remove position
                state["positions"] = [p for p in state["positions"] if p["symbol"] != pos["symbol"]]
                print(f"Executed SELL order: {pos['symbol']} at {open_price:.2f} on {trade_date}")

        elif order["type"] == "buy":
            # BUG-08: guard against duplicate positions
            if _find_position(state, order["symbol"]):
                print(f"Skipped BUY order: {order['symbol']} (position already exists)")
                continue
            # Buy execution
            equity = _portfolio_equity(state)
            budget = min(state["cash"], equity * POSITION_WEIGHT)
            buy_price = open_price * (1 + SLIPPAGE_RATE)
            fee_adjusted = buy_price * (1 + COMMISSION_RATE)
            shares = int(budget // fee_adjusted)

            if shares <= 0:
                print(f"Skipped BUY order: {order['symbol']} (insufficient cash)")
                continue

            cost = shares * fee_adjusted
            state["cash"] -= cost
            new_pos = Position(
                symbol=order["symbol"],
                name=order["name"],
                shares=shares,
                entry_signal_date=order["signal_date"],
                entry_date=trade_date,
                entry_price=buy_price,
                entry_cost=cost,
            )
            state["positions"].append(asdict(new_pos))
            print(f"Executed BUY order: {order['symbol']} at {buy_price:.2f} on {trade_date}")

    state["pending_orders"] = remaining_orders
    return state


def _apply_exits(state: dict, panel: pd.DataFrame, signal_date: pd.Timestamp) -> tuple[dict, list[dict]]:
    exits = []
    latest_panel = panel[panel["date"] == signal_date].set_index("symbol")

    for pos in state.get("positions", []):
        if pos["symbol"] not in latest_panel.index:
            continue
        row = latest_panel.loc[pos["symbol"]]
        # BUG-09: guard against double-increment on same day (re-run protection)
        last_bar_date = pos.get("last_bar_date", "")
        current_date_str = str(signal_date.date())
        if last_bar_date != current_date_str:
            pos["holding_bars"] = int(pos.get("holding_bars", 0)) + 1
            pos["last_bar_date"] = current_date_str
        
        reason = _check_exit_reason(pos, row)
        if reason:
            signal = {
                "symbol": pos["symbol"],
                "name": pos["name"],
                "reason": reason,
                "signal_date": str(signal_date.date()),
            }
            exits.append(signal)
            
            # Put in pending sell orders if not already present
            if not _has_pending_order(state, pos["symbol"], "sell"):
                state["pending_orders"].append(
                    {
                        "type": "sell",
                        "symbol": pos["symbol"],
                        "name": pos["name"],
                        "signal_date": str(signal_date.date()),
                        "reason": reason,
                        "status": "pending_next_open",
                    }
                )
    return state, exits


def _check_exit_reason(pos: dict, row: pd.Series) -> str | None:
    if row.get("market_regime") == "bear":
        return "market_bear"
    close = row.get("Close")
    # BUG-10: guard against missing Close value — never trigger false stop-loss
    if close is None:
        return None
    close = float(close)
    if close <= 0:
        return None  # price data error; skip exit logic
    if close <= float(pos["entry_price"]) * (1 - STOP_LOSS):
        return "stop_loss"
    if int(pos.get("holding_bars", 0)) >= MAX_HOLDING_DAYS:
        return "max_holding_days"
    ma120 = row.get("ma120")
    if pd.notna(ma120) and close < float(ma120):
        return "below_ma120"
    return None


def _create_buy_orders(state: dict, candidates: pd.DataFrame, latest_date: pd.Timestamp) -> dict:
    latest_str = str(latest_date.date())
    if state.get("last_signal_date") == latest_str:
        return state

    held = {pos["symbol"] for pos in state.get("positions", [])}
    pending_buys = {order["symbol"] for order in state.get("pending_orders", []) if order["type"] == "buy"}
    slots = MAX_POSITIONS - len(held) - len(pending_buys)

    if slots > 0 and not candidates.empty:
        new_candidates = candidates[~candidates["symbol"].isin(held | pending_buys)].head(slots)
        for row in new_candidates.itertuples(index=False):
            state["pending_orders"].append(
                {
                    "type": "buy",
                    "symbol": row.symbol,
                    "name": getattr(row, "name", ""),
                    "signal_date": latest_str,
                    "status": "pending_next_open",
                }
            )
        state["last_signal_date"] = latest_str

    return state


def _portfolio_equity(state: dict) -> float:
    value = float(state["cash"])
    for pos in state.get("positions", []):
        close = _get_last_close(pos["symbol"])
        if close is not None:  # BUG-04: use 'is not None' to handle close=0.0 correctly
            value += pos["shares"] * close
    return value


def _snapshot(state: dict, panel: pd.DataFrame, latest_date: pd.Timestamp, candidates_count: int, exits_count: int) -> dict:
    equity = _portfolio_equity(state)
    holdings_mv = sum(pos["shares"] * (_get_last_close(pos["symbol"]) or 0.0) for pos in state["positions"])

    # Compute benchmark equity: rebase 0050.TW close to INITIAL_CASH
    # BUG-15: initialize as None (not empty string) for type consistency
    benchmark_equity = None
    try:
        bench_frame = _read_cached_prices("0050.TW")
        if not bench_frame.empty:
            # First snapshot date
            first_snap_date = pd.Timestamp("2026-06-01")
            try:
                if SNAPSHOTS_PATH.exists():
                    snapshots_df = pd.read_csv(SNAPSHOTS_PATH)
                    if not snapshots_df.empty:
                        first_snap_date = pd.Timestamp(snapshots_df.iloc[0]["date"])
            except Exception:
                pass
            
            # Find the close on or before the first snapshot date (baseline close)
            base_past = bench_frame[bench_frame.index.normalize() <= first_snap_date]
            first_close = float(base_past.iloc[-1]["Close"]) if not base_past.empty else float(bench_frame.iloc[0]["Close"])
            
            # Close on or before the snapshot date
            snap_ts = pd.Timestamp(latest_date.date())
            past = bench_frame[bench_frame.index.normalize() <= snap_ts]
            if not past.empty and first_close > 0:
                day_close = float(past.iloc[-1]["Close"])
                benchmark_equity = round(INITIAL_CASH * (day_close / first_close), 2)
    except Exception:
        pass

    return {
        "date": str(latest_date.date()),
        "cash": state["cash"],
        "market_value": holdings_mv,
        "equity": equity,
        "total_return_pct": (equity / INITIAL_CASH - 1) * 100,
        "positions": len(state["positions"]),
        "pending_orders": len(state["pending_orders"]),
        "candidate_rows": candidates_count,
        "exit_signals": exits_count,
        "benchmark": benchmark_equity,
    }


def _get_next_open(symbol: str, signal_date: str) -> tuple[float | None, str | None]:
    frame = _read_cached_prices(symbol)
    if frame.empty:
        return None, None
    signal_ts = pd.Timestamp(signal_date)
    future = frame[frame.index > signal_ts]
    if future.empty:
        return None, None
    row = future.iloc[0]
    return float(row["Open"]), str(future.index[0].date())


def _get_last_close(symbol: str) -> float | None:
    frame = _read_cached_prices(symbol)
    if frame.empty:
        return None
    return float(frame.iloc[-1]["Close"])


def _read_cached_prices(symbol: str) -> pd.DataFrame:
    path = ROOT / "work" / "price_cache" / f"{symbol.replace('.', '_')}.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    return frame


def _find_position(state: dict, symbol: str) -> dict | None:
    for pos in state.get("positions", []):
        if pos["symbol"] == symbol:
            return pos
    return None


def _has_pending_order(state: dict, symbol: str, order_type: str) -> bool:
    return any(o["symbol"] == symbol and o["type"] == order_type for o in state.get("pending_orders", []))


def _append_trade(trade: dict) -> None:
    trades_df = pd.read_csv(TRADES_PATH) if TRADES_PATH.exists() else pd.DataFrame(columns=[
        "symbol", "name", "entry_signal_date", "entry_date", "exit_signal_date", "exit_date",
        "entry_price", "exit_price", "shares", "pnl", "return_pct", "holding_bars", "exit_reason"
    ])
    trades_df = pd.concat([trades_df, pd.DataFrame([trade])], ignore_index=True)
    trades_df.to_csv(TRADES_PATH, index=False, encoding="utf-8-sig")


def _append_snapshot(row: dict) -> None:
    snapshots_df = pd.read_csv(SNAPSHOTS_PATH) if SNAPSHOTS_PATH.exists() else pd.DataFrame()
    if "date" in snapshots_df.columns:
        snapshots_df = snapshots_df[snapshots_df["date"] != row["date"]]
    snapshots_df = pd.concat([snapshots_df, pd.DataFrame([row])], ignore_index=True)
    # BUG-SORT: Sort daily snapshots by date to keep strict chronological order on charts
    if "date" in snapshots_df.columns:
        snapshots_df = snapshots_df.sort_values("date")
    snapshots_df.to_csv(SNAPSHOTS_PATH, index=False, encoding="utf-8-sig")


def _generate_static_json_payloads(state: dict, latest_date: pd.Timestamp, snapshot: dict, panel: pd.DataFrame) -> None:
    """Generates pre-compiled dashboard.json and holdings_history.json for static serverless hosting."""
    try:
        # 1. Compile dashboard.json
        snapshots = []
        if SNAPSHOTS_PATH.exists():
            snapshots_df = pd.read_csv(SNAPSHOTS_PATH)
            snapshots = snapshots_df.to_dict(orient="records")
            
        candidates = []
        candidates_path = SIGNALS_DIR / f"candidates_{latest_date.date()}.csv"
        if candidates_path.exists():
            candidates_df = pd.read_csv(candidates_path)
            candidates = candidates_df.head(10).to_dict(orient="records")
            
        trades = []
        if TRADES_PATH.exists():
            trades_df = pd.read_csv(TRADES_PATH)
            trades = trades_df.tail(30).to_dict(orient="records")
            
        payload = {
            "date": str(latest_date.date()),
            "market_regime": str(panel["market_regime"].iloc[-1]) if "market_regime" in panel.columns else "neutral",
            "cash": state["cash"],
            "equity": snapshot["equity"],
            "total_return_pct": snapshot["total_return_pct"],
            "positions": state["positions"] or [],
            "pending_orders": state["pending_orders"] or [],
            "candidates": candidates,
            "trades": trades,
            "snapshots": snapshots
        }
        
        dashboard_path = ROOT / "docs" / "outputs" / "paper_trading" / "dashboard.json"
        # BUG-12: sanitize NaN/Inf values from pandas before json.dump
        def _sanitize(obj):
            if isinstance(obj, float):
                import math
                if math.isnan(obj) or math.isinf(obj):
                    return None
                return obj
            if isinstance(obj, dict):
                return {k: _sanitize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_sanitize(v) for v in obj]
            return obj
        with open(dashboard_path, "w", encoding="utf-8") as handle:
            json.dump(_sanitize(payload), handle, ensure_ascii=False, indent=2)
        print(f"Pre-compiled static dashboard payload written to {dashboard_path.name}")
        
        # 2. Compile holdings_history.json
        positions = state.get("positions", [])
        if not positions:
            holdings_history = {"dates": [], "series": {}}
        else:
            union_dates = set()
            raw_series = {}
            six_months_ago = (latest_date - pd.Timedelta(days=180)).date().isoformat()
            
            for pos in positions:
                symbol = pos["symbol"]
                path = ROOT / "work" / "price_cache" / f"{symbol.replace('.', '_')}.csv"
                if path.exists():
                    df = pd.read_csv(path)
                    df["date_str"] = pd.to_datetime(df["Date"]).dt.date.astype(str)
                    df = df[df["date_str"] >= six_months_ago]
                    
                    prices = dict(zip(df["date_str"], df["Close"]))
                    raw_series[symbol] = prices
                    union_dates.update(df["date_str"])
                    
            sorted_dates = sorted(list(union_dates))
            normalized_series = {}
            for symbol, prices in raw_series.items():
                first_close = None
                for d in sorted_dates:
                    if d in prices:
                        first_close = float(prices[d])
                        break
                if not first_close:
                    first_close = 1.0
                    
                normalized_seq = []
                last_valid = 100.0
                for d in sorted_dates:
                    if d in prices:
                        val = (float(prices[d]) / first_close) * 100.0
                        normalized_seq.append(round(val, 2))
                        last_valid = val
                    else:
                        normalized_seq.append(round(last_valid, 2))
                normalized_series[symbol] = normalized_seq
                
            holdings_history = {
                "dates": sorted_dates,
                "series": normalized_series
            }
            
        history_path = ROOT / "docs" / "outputs" / "paper_trading" / "holdings_history.json"
        with open(history_path, "w", encoding="utf-8") as handle:
            json.dump(holdings_history, handle, ensure_ascii=False, indent=2)
        print(f"Pre-compiled static holdings history payload written to {history_path.name}")
        
    except Exception as e:
        print(f"Warning: Failed to compile static json payloads: {e}")


if __name__ == "__main__":
    main()
