from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
from strategy import entry_candidates


@dataclass
class Position:
    symbol: str
    shares: int
    entry_date: pd.Timestamp
    entry_price: float
    last_rs20_rank_pct: float
    rs20_weak_count: int = 0
    holding_bars: int = 0


def run_backtest(
    panel: pd.DataFrame,
    data: dict[str, pd.DataFrame],
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Runs a historical portfolio backtest with signal at t, execution at t+1 Open."""
    cash = float(config["initial_cash"])
    positions: dict[str, Position] = {}
    cooldown_until: dict[str, pd.Timestamp] = {}
    trades = []
    equity_rows = []

    dates = sorted(panel["date"].drop_duplicates())
    if not dates:
        return pd.DataFrame(), pd.DataFrame()

    for idx, date in enumerate(dates[:-1]):
        next_date = dates[idx + 1]
        day = panel[panel["date"] == date].set_index("symbol")

        # 1. Process Exits
        for symbol in list(positions):
            if symbol not in day.index:
                continue
            pos = positions[symbol]
            row = day.loc[symbol]

            pos.holding_bars += 1
            if row["rs20_rank_pct"] > pos.last_rs20_rank_pct:
                pos.rs20_weak_count += 1
            else:
                pos.rs20_weak_count = 0
            pos.last_rs20_rank_pct = float(row["rs20_rank_pct"])

            exit_reason = _exit_reason(pos, row, date, config)
            if exit_reason:
                open_price = _next_open(data[symbol], date, next_date)
                if pd.isna(open_price):
                    continue

                proceeds = pos.shares * open_price * (1 - config["slippage_rate"])
                fee = proceeds * config["commission_rate"]
                tax = proceeds * config["tax_rate"]
                cash_in = proceeds - fee - tax
                cash += cash_in

                pnl = cash_in - (pos.shares * pos.entry_price * (1 + config["commission_rate"]))
                trades.append(
                    {
                        "symbol": symbol,
                        "entry_date": pos.entry_date,
                        "exit_signal_date": date,
                        "exit_date": next_date,
                        "entry_price": pos.entry_price,
                        "exit_price": open_price,
                        "shares": pos.shares,
                        "pnl": pnl,
                        "return_pct": (open_price / pos.entry_price) - 1,
                        "holding_days": pos.holding_bars,
                        "exit_reason": exit_reason,
                    }
                )

                if exit_reason == "stop_loss":
                    cooldown_until[symbol] = date + pd.Timedelta(days=config["cooldown_days_after_stop"])
                    del positions[symbol]
                else:
                    del positions[symbol]

        # 2. Process Entries
        free_slots = config["max_positions"] - len(positions)
        if free_slots > 0:
            candidates = entry_candidates(panel, date, config)
            candidates = candidates[~candidates["symbol"].isin(set(positions.keys()))]
            candidates = candidates[
                candidates["symbol"].map(lambda sym: cooldown_until.get(sym, pd.Timestamp.min) <= date)
            ]

            for candidate in candidates.head(free_slots).itertuples(index=False):
                open_price = _next_open(data[candidate.symbol], date, next_date)
                if pd.isna(open_price):
                    continue

                max_gap = config["entry"].get("max_entry_gap")
                if max_gap is not None and (open_price / candidate.Close - 1) > max_gap:
                    continue

                sizing_base = config.get("position_sizing", "initial_cash")
                if sizing_base == "equity":
                    base_val = _portfolio_value(cash, positions, data, date)
                else:
                    base_val = config["initial_cash"]

                budget = min(cash, base_val * config["position_weight"])
                buy_price = open_price * (1 + config["slippage_rate"])
                fee_adjusted = buy_price * (1 + config["commission_rate"])
                shares = int(budget // fee_adjusted)

                if shares <= 0:
                    continue

                cost = shares * fee_adjusted
                cash -= cost
                positions[candidate.symbol] = Position(
                    symbol=candidate.symbol,
                    shares=shares,
                    entry_date=next_date,
                    entry_price=buy_price,
                    last_rs20_rank_pct=float(candidate.rs20_rank_pct),
                )

        # 3. Snapshot Daily Equity
        market_val = 0.0
        for symbol, pos in positions.items():
            close = _last_close(data[symbol], date)
            if not pd.isna(close):
                market_val += pos.shares * close
        equity_rows.append(
            {
                "date": date,
                "cash": cash,
                "market_value": market_val,
                "equity": cash + market_val,
            }
        )

    # 4. Snapshot Daily Equity for the very last date (BUG-36 fix)
    if dates:
        last_date = dates[-1]
        market_val = 0.0
        for symbol, pos in positions.items():
            close = _last_close(data[symbol], last_date)
            if not pd.isna(close):
                market_val += pos.shares * close
        equity_rows.append(
            {
                "date": last_date,
                "cash": cash,
                "market_value": market_val,
                "equity": cash + market_val,
            }
        )

    return pd.DataFrame(equity_rows), pd.DataFrame(trades)


def _portfolio_value(
    cash: float,
    positions: dict[str, Position],
    data: dict[str, pd.DataFrame],
    date: pd.Timestamp,
) -> float:
    market_val = 0.0
    for symbol, pos in positions.items():
        close = _last_close(data[symbol], date)
        if not pd.isna(close):
            market_val += pos.shares * close
    return cash + market_val


def _exit_reason(pos: Position, row: pd.Series, date: pd.Timestamp, config: dict) -> str | None:
    if row["market_regime"] == "bear":
        return "market_bear"
    if row["Close"] <= pos.entry_price * (1 - config["exit"]["stop_loss"]):
        return "stop_loss"
    if pos.holding_bars >= config["exit"]["max_holding_days"]:
        return "max_holding_days"
    trend_ma = config["exit"].get("trend_ma", 120)
    ma_col = f"ma{trend_ma}"
    if ma_col in row and row["Close"] < row[ma_col]:
        return f"below_ma{trend_ma}"
    if pos.rs20_weak_count >= config["exit"]["rs20_weak_days"]:
        return "rs20_weak"
    return None


def _next_open(frame: pd.DataFrame, signal_date: pd.Timestamp, next_date: pd.Timestamp) -> float:
    # BUG-37: Try using exact next_date, fallback to any date after signal_date
    future = frame[frame.index >= next_date]
    if future.empty:
        future = frame[frame.index > signal_date]
        if future.empty:
            return float("nan")
    return float(future.iloc[0]["Open"])


def _last_close(frame: pd.DataFrame, date: pd.Timestamp) -> float:
    past = frame[frame.index <= date]
    if past.empty:
        return float("nan")
    return float(past.iloc[-1]["Close"])

