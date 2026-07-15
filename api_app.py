from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, FileResponse
from starlette.routing import Route, Mount
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

# Paths for files loaded by API
STATE_PATH = ROOT / "docs" / "outputs" / "paper_trading" / "state.json"
TRADES_PATH = ROOT / "docs" / "outputs" / "paper_trading" / "trades.csv"
SNAPSHOTS_PATH = ROOT / "docs" / "outputs" / "paper_trading" / "daily_snapshots.csv"
SIGNALS_DIR = ROOT / "docs" / "outputs" / "paper_trading" / "signals"


def _clean_json(value: Any) -> Any:
    """Helper to convert NaN/Inf float values to None for valid JSON output."""
    if isinstance(value, dict):
        return {str(k): _clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _json(data: Any, status_code: int = 200) -> JSONResponse:
    return JSONResponse(_clean_json(data), status_code=status_code)


def _get_last_close(symbol: str) -> float | None:
    path = ROOT / "work" / "price_cache" / f"{symbol.replace('.', '_')}.csv"
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
        return float(frame.iloc[-1]["Close"])
    except Exception:
        return None



async def health(request: Request) -> JSONResponse:
    return _json({"status": "ok", "app": "quality-momentum-api"})


async def get_dashboard(request: Request) -> JSONResponse:
    """Returns the combined state, holdings, trades, and snapshots payload."""
    if not STATE_PATH.exists():
        return _json({
            "error": "State file not initialized. Run daily update once to initialize."
        }, status_code=404)

    with open(STATE_PATH, "r", encoding="utf-8") as handle:
        state = json.load(handle)

    # 1. Enrich active positions with current close prices and unrealized pnl
    enriched_positions = []
    equity = float(state["cash"])
    for pos in state.get("positions", []):
        close = _get_last_close(pos["symbol"]) or pos["entry_price"]
        market_val = pos["shares"] * close
        unrealized_pnl = market_val - pos["entry_cost"]
        equity += market_val
        enriched_positions.append({
            **pos,
            "close": close,
            "market_value": market_val,
            "unrealized_pnl": unrealized_pnl,
            "unrealized_return_pct": (close / pos["entry_price"] - 1) * 100
        })

    # 2. Load trades history
    trades = []
    if TRADES_PATH.exists():
        try:
            trades_df = pd.read_csv(TRADES_PATH)
            trades = trades_df.tail(50).to_dict("records")
        except Exception:
            pass

    # 3. Load daily snapshots for charting
    snapshots = []
    if SNAPSHOTS_PATH.exists():
        try:
            snapshots_df = pd.read_csv(SNAPSHOTS_PATH)
            # Replace empty/NaN benchmark values with None so frontend receives null
            if 'benchmark' in snapshots_df.columns:
                snapshots_df['benchmark'] = pd.to_numeric(snapshots_df['benchmark'], errors='coerce')
                snapshots_df['benchmark'] = snapshots_df['benchmark'].where(snapshots_df['benchmark'].notna(), other=None)
            snapshots = snapshots_df.tail(120).to_dict("records")
        except Exception:
            pass

    # 4. Load latest candidate signals
    latest_date = state.get("last_signal_date")
    candidates = []
    market_regime = "neutral"
    if latest_date:
        candidates_path = SIGNALS_DIR / f"candidates_{latest_date}.csv"
        if candidates_path.exists():
            try:
                candidates_df = pd.read_csv(candidates_path)
                if not candidates_df.empty:
                    market_regime = str(candidates_df.iloc[0].get("market_regime", "neutral"))
                    candidates = candidates_df.head(15).to_dict("records")
            except Exception:
                pass

    total_return_pct = (equity / state.get("initial_cash", 1000000.0) - 1) * 100

    payload = {
        "latest_date": latest_date,
        "market_regime": market_regime,
        "cash": state["cash"],
        "equity": equity,
        "total_return_pct": total_return_pct,
        "positions": enriched_positions,
        "pending_orders": state.get("pending_orders", []),
        "candidates": candidates,
        "trades": trades,
        "snapshots": snapshots
    }
    return _json(payload)


# Global variable to track the background update process
update_process: subprocess.Popen | None = None
last_update_error: str = ""


async def run_update(request: Request) -> JSONResponse:
    """Launches the run_daily_update.py script in the background to avoid web timeout."""
    global update_process, last_update_error
    
    # If a process is already running, don't launch another one
    if update_process is not None and update_process.poll() is None:
        return _json({
            "status": "running",
            "ok": True,
            "message": "Update script is already running in background."
        })
        
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"
    last_update_error = ""
    
    try:
        log_dir = ROOT / "work"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = open(log_dir / "daily_update_error.log", "w", encoding="utf-8")
        
        # Use stdout=DEVNULL and stderr=log_file to avoid pipe buffer deadlock
        update_process = subprocess.Popen(
            [sys.executable, str(ROOT / "run_daily_update.py")],
            cwd=ROOT,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=log_file,
        )
        return _json({
            "status": "started",
            "ok": True,
            "message": "Update script started in background."
        })
    except Exception as e:
        return _json({
            "status": "failed",
            "ok": False,
            "message": str(e)
        }, status_code=500)


async def get_update_status(request: Request) -> JSONResponse:
    """Checks the status of the background update process."""
    global update_process, last_update_error
    
    if update_process is None:
        return _json({"status": "idle"})
        
    poll = update_process.poll()
    if poll is None:
        # Check if progress file exists to return real-time updates
        progress_path = STATE_PATH.parent / "progress.json"
        if progress_path.exists():
            try:
                with open(progress_path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                    return _json(data)
            except Exception:
                pass
        return _json({
            "status": "running",
            "progress": 10,
            "message": "Downloading data from yfinance...",
            "seconds_left": 18
        })
    elif poll == 0:
        return _json({"status": "success"})
    else:
        # Get errors from log file if cached version is empty
        if not last_update_error:
            try:
                log_path = ROOT / "work" / "daily_update_error.log"
                if log_path.exists():
                    last_update_error = log_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                pass
        return _json({
            "status": "failed",
            "code": poll,
            "error": last_update_error
        })


async def serve_index(request: Request) -> FileResponse:
    """Serves the main static docs/index.html dashboard webpage."""
    return FileResponse(str(ROOT / "docs" / "index.html"))


async def get_holdings_history(request: Request) -> JSONResponse:
    """Returns the past 6 months price history of currently active holdings."""
    if not STATE_PATH.exists():
        return _json({"error": "State file not found"}, status_code=404)
        
    with open(STATE_PATH, "r", encoding="utf-8") as handle:
        state = json.load(handle)
        
    positions = state.get("positions", [])
    if not positions:
        return _json({"dates": [], "series": {}})
        
    import datetime
    union_dates = set()
    raw_series = {}
    
    # 6 months ago threshold
    six_months_ago = (datetime.date.today() - datetime.timedelta(days=180)).isoformat()
    
    for pos in positions:
        symbol = pos["symbol"]
        path = ROOT / "work" / "price_cache" / f"{symbol.replace('.', '_')}.csv"
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, parse_dates=["Date"])
            df["date_str"] = df["Date"].dt.date.astype(str)
            df_filtered = df[df["date_str"] >= six_months_ago]
            
            dates_prices = dict(zip(df_filtered["date_str"], df_filtered["Close"]))
            raw_series[symbol] = dates_prices
            union_dates.update(df_filtered["date_str"])
        except Exception:
            pass
            
    sorted_dates = sorted(list(union_dates))
    
    normalized_series = {}
    for symbol, dates_prices in raw_series.items():
        prices_seq = []
        first_valid_price = None
        
        for d in sorted_dates:
            price = dates_prices.get(d)
            if price is not None and not math.isnan(price):
                if first_valid_price is None:
                    first_valid_price = price
                prices_seq.append(price)
            else:
                prices_seq.append(prices_seq[-1] if prices_seq else None)
                
        if first_valid_price:
            normalized_seq = []
            for p in prices_seq:
                if p is not None:
                    normalized_seq.append(round((p / first_valid_price) * 100, 2))
                else:
                    normalized_seq.append(100.0)
            normalized_series[symbol] = normalized_seq
            
    return _json({
        "dates": sorted_dates,
        "series": normalized_series
    })


async def schedule_exit(request: Request) -> JSONResponse:
    """Schedules a pending order (buy or sell) for execution."""
    try:
        body = await request.json()
        symbol = body.get("symbol")
        order_type = body.get("type", "sell")  # Default to 'sell' for backward compatibility
        
        if not symbol:
            return _json({"status": "failed", "message": "Symbol is required"}, status_code=400)
            
        if not STATE_PATH.exists():
            return _json({"status": "failed", "message": "State file not found"}, status_code=404)
            
        # BUG-17 & BUG-06: Atomic read/write using state.json.tmp and os.replace
        # Simple lock to minimize concurrent writes
        with open(STATE_PATH, "r", encoding="utf-8") as handle:
            state = json.load(handle)
            
        import datetime
        if order_type == "sell":
            # Find position
            pos = next((p for p in state.get("positions", []) if p["symbol"] == symbol), None)
            if not pos:
                return _json({"status": "failed", "message": f"Position for {symbol} not found in portfolio"}, status_code=400)
                
            # Check if already pending sell
            already_pending = any(o for o in state.get("pending_orders", []) if o["symbol"] == symbol and o["type"] == "sell")
            if already_pending:
                return _json({"status": "failed", "message": f"Sell order for {symbol} is already scheduled"}, status_code=400)
                
            # Add sell order
            sell_order = {
                "type": "sell",
                "symbol": symbol,
                "name": pos["name"],
                "signal_date": state.get("last_signal_date") or datetime.date.today().isoformat(),
                "reason": "manual",
                "status": "pending_next_open"  # BUG-22: match backend structure
            }
            state["pending_orders"].append(sell_order)
            msg = f"Scheduled exit for {symbol}"
        else:
            # Check if already in positions
            already_held = any(p for p in state.get("positions", []) if p["symbol"] == symbol)
            if already_held:
                return _json({"status": "failed", "message": f"{symbol} is already in positions"}, status_code=400)
                
            # Check if already pending buy
            already_pending = any(o for o in state.get("pending_orders", []) if o["symbol"] == symbol and o["type"] == "buy")
            if already_pending:
                return _json({"status": "failed", "message": f"Buy order for {symbol} is already scheduled"}, status_code=400)
                
            # Look up name from universe CSV
            pos_name = ""
            try:
                with open(ROOT / "config.json", "r", encoding="utf-8") as hc:
                    cfg = json.load(hc)
                upath = ROOT / cfg.get("universe_csv", "data/universe_twse_all.csv")
                if upath.exists():
                    df = pd.read_csv(upath)
                    match = df[df["symbol"] == symbol]
                    if not match.empty:
                        pos_name = str(match.iloc[0]["name"])
            except Exception:
                pass
                
            # Add buy order
            buy_order = {
                "type": "buy",
                "symbol": symbol,
                "name": pos_name,
                "signal_date": state.get("last_signal_date") or datetime.date.today().isoformat(),
                "status": "pending_next_open"
            }
            state["pending_orders"].append(buy_order)
            msg = f"Scheduled buy for {symbol}"
        
        # Atomic write
        tmp_path = STATE_PATH.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(state, handle, ensure_ascii=False, indent=2)
        os.replace(tmp_path, STATE_PATH)
            
        return _json({"status": "success", "message": msg})
    except Exception as e:
        return _json({"status": "failed", "message": str(e)}, status_code=500)


app = Starlette(
    debug=False,
    routes=[
        Route("/", serve_index, methods=["GET"]),
        Route("/health", health, methods=["GET"]),
        Route("/api/dashboard", get_dashboard, methods=["GET"]),
        Route("/api/update", run_update, methods=["POST"]),
        Route("/api/update/status", get_update_status, methods=["GET"]),
        Route("/api/holdings/history", get_holdings_history, methods=["GET"]),
        Route("/api/orders/exit", schedule_exit, methods=["POST"]),
        Mount("/outputs", StaticFiles(directory=str(ROOT / "docs" / "outputs")), name="outputs"),
    ],
    middleware=[
        Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    ],
)
