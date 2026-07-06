from __future__ import annotations

import numpy as np
import pandas as pd


def _higher_better(value: float | None, low: float, high: float) -> float:
    if value is None or pd.isna(value):
        return np.nan
    if high == low:
        return np.nan
    return float(np.clip((value - low) / (high - low), 0, 1))


def _lower_better(value: float | None, high: float, low: float) -> float:
    if value is None or pd.isna(value):
        return np.nan
    if high == low:
        return np.nan
    return float(np.clip((high - value) / (high - low), 0, 1))


def _middle_better(value: float | None, low: float, target: float, high: float) -> float:
    if value is None or pd.isna(value) or value <= low or value >= high:
        return np.nan if value is None or pd.isna(value) else 0.0
    if target == low or high == target:
        return np.nan
    if value <= target:
        return float(np.clip((value - low) / (target - low), 0, 1))
    return float(np.clip((high - value) / (high - target), 0, 1))


def fundamental_score(row: pd.Series | None) -> float:
    """Calculates a fundamental score between 0 and 100 based on standard metrics."""
    if row is None:
        return 50.0
    if "eva_like_score" in row and not pd.isna(row.get("eva_like_score")):
        return float(row.get("eva_like_score"))

    points = {
        "roe": _higher_better(row.get("roe"), 0.05, 0.25),
        "revenue_growth": _higher_better(row.get("revenue_growth"), -0.1, 0.3),
        "eps": _higher_better(row.get("eps"), 0.0, 10.0),
        "debt_to_equity": _lower_better(row.get("debt_to_equity"), 2.0, 0.2),
        "pe": _middle_better(row.get("pe"), 5.0, 18.0, 45.0),
        "pb": _lower_better(row.get("pb"), 8.0, 1.0),
        "gross_margin": _higher_better(row.get("gross_margin"), 0.1, 0.5),
        "operating_margin": _higher_better(row.get("operating_margin"), 0.02, 0.25),
    }
    available = [val for val in points.values() if not np.isnan(val)]
    if not available:
        return 50.0
    return float(np.mean(available) * 100)


def passes_fundamental_filters(row: pd.Series | None, filters: dict) -> bool:
    """Checks if a stock passes hard fundamental criteria."""
    if row is None:
        return False
        
    def ge(v, thresh):
        if v is None or pd.isna(v): return False
        return float(v) >= float(thresh)
    
    def gt(v, thresh):
        if v is None or pd.isna(v): return False
        return float(v) > float(thresh)

    def le(v, thresh):
        if v is None or pd.isna(v): return False
        return float(v) <= float(thresh)

    if not pd.isna(row.get("eva_like_score", np.nan)):
        checks = [
            ge(row.get("eva_like_score"), filters.get("min_eva_like_score", filters["min_fundamental_score"])),
            ge(row.get("roe"), filters["min_roe"]),
            gt(row.get("eps"), filters["min_eps"]),
            le(row.get("debt_to_equity"), filters["max_debt_to_equity"]),
        ]
        return all(checks)

    checks = [
        ge(row.get("roe"), filters["min_roe"]),
        ge(row.get("revenue_growth"), filters["min_revenue_growth"]),
        gt(row.get("eps"), filters["min_eps"]),
        le(row.get("debt_to_equity"), filters["max_debt_to_equity"]),
        gt(row.get("pe"), filters["min_pe"]) and le(row.get("pe"), filters["max_pe"]),
        gt(row.get("pb"), filters["min_pb"]) and le(row.get("pb"), filters["max_pb"]),
    ]
    return all(checks)


def build_feature_panel(
    data: dict[str, pd.DataFrame],
    benchmark: pd.DataFrame,
    market_regime: pd.Series,
) -> pd.DataFrame:
    """Concatenates all symbol frames and merges benchmark returns and market regime."""
    frames = []
    for symbol, frame in data.items():
        if frame.empty:
            continue
        sym_frame = frame.copy()
        sym_frame["symbol"] = symbol
        sym_frame["date"] = sym_frame.index
        frames.append(sym_frame)

    if not frames:
        return pd.DataFrame()

    panel = pd.concat(frames, ignore_index=True)
    bench_features = pd.DataFrame(
        {
            "date": benchmark.index,
            "benchmark_ret20": benchmark["Close"].pct_change(20).to_numpy(),
            "benchmark_ret60": benchmark["Close"].pct_change(60).to_numpy(),
            "market_regime": market_regime.reindex(benchmark.index).to_numpy(),
        }
    )
    panel = panel.merge(bench_features, on="date", how="left")
    
    # Calculate global RS rank percentiles
    panel["rs20_rank_pct"] = panel.groupby("date")["ret20"].rank(ascending=False, pct=True)
    panel["rs60_rank_pct"] = panel.groupby("date")["ret60"].rank(ascending=False, pct=True)
    return panel.sort_values(["date", "symbol"], ignore_index=True)


def _technical_score(df: pd.DataFrame) -> pd.Series:
    """Calculates technical score (0-100) based on momentum, trend, RSI, and volume."""
    score = pd.Series(0.0, index=df.index)
    score += (1 - df["rs20_rank_pct"]).clip(0, 1) * 30
    score += (1 - df["rs60_rank_pct"]).clip(0, 1) * 25
    # BUG-27: Prevent division by zero if ma20 is zero
    ma20_safe = df["ma20"].replace(0, np.nan)
    score += ((df["Close"] / ma20_safe) - 1).fillna(0).clip(0, 0.12) / 0.12 * 10
    score += (df["ma20"] > df["ma60"]).astype(float) * 10
    score += (df["ma20_slope"] > 0).astype(float) * 10
    score += (1 - ((df["rsi14"] - 61).abs() / 9)).clip(0, 1) * 10
    score += (1 - ((df["volume_ratio"] - 1.6).abs() / 0.6)).clip(0, 1) * 5
    return score.clip(0, 100)



def add_strategy_scores(panel: pd.DataFrame, fundamentals: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Processes fundamentals, merges them, and computes technical, fundamental, and final weights."""
    df = panel.sort_values(["symbol", "date"]).copy()
    
    # BUG-25: Guard dt.tz_localize(None) against already tz-naive datetimes
    df_dt = pd.to_datetime(df["date"])
    if df_dt.dt.tz is not None:
        df_dt = df_dt.dt.tz_localize(None)
    df["date"] = df_dt.dt.as_unit("ns")

    use_fundamentals = config.get("use_fundamentals", True)
    if not use_fundamentals or fundamentals.empty:
        df["fundamental_score"] = 50.0
        df["fundamental_pass"] = True
    else:
        fund = fundamentals.copy().sort_values(["symbol", "as_of_date"])
        fund_dt = pd.to_datetime(fund["as_of_date"])
        if fund_dt.dt.tz is not None:
            fund_dt = fund_dt.dt.tz_localize(None)
        fund["as_of_date"] = fund_dt.dt.as_unit("ns")
        fund["fundamental_score"] = fund.apply(fundamental_score, axis=1)
        fund["fundamental_pass"] = fund.apply(
            lambda r: passes_fundamental_filters(r, config.get("fundamental_filters", {})),
            axis=1,
        )

        merged = []
        fund_symbols = set(fund["symbol"].dropna().astype(str))
        for symbol, group in df.groupby("symbol", sort=False):
            if symbol not in fund_symbols:
                out = group.copy()
                out["fundamental_score"] = 50.0
                out["fundamental_pass"] = False
                merged.append(out)
                continue
            fund_group = fund[fund["symbol"] == symbol]
            out = pd.merge_asof(
                group.sort_values("date"),
                fund_group.sort_values("as_of_date"),
                left_on="date",
                right_on="as_of_date",
                by="symbol",
                direction="backward",
                suffixes=("", "_fundamental"),
            )
            out["fundamental_score"] = out["fundamental_score"].fillna(50.0)
            out["fundamental_pass"] = out["fundamental_pass"].fillna(False).astype(bool)
            merged.append(out)
        df = pd.concat(merged, ignore_index=True)

    tech = _technical_score(df)
    weights = config["score_weights"] if use_fundamentals else {"technical": 1.0, "fundamental": 0.0}
    df["technical_score"] = tech
    df["score"] = df["technical_score"]
    df["final_score"] = weights["technical"] * df["technical_score"] + weights["fundamental"] * df["fundamental_score"]
    df["fundamental_rank"] = df.groupby("date")["fundamental_score"].rank(ascending=False, method="first")
    return df.sort_values(["date", "symbol"], ignore_index=True)


def _fundamental_universe_mask(day: pd.DataFrame, univ_filter: dict) -> pd.Series:
    mask = day["fundamental_score"] >= univ_filter.get("min_score", 0)
    top_n = univ_filter.get("top_n")
    if top_n is not None:
        mask = mask & (day["fundamental_rank"] <= top_n)
    return mask


def entry_candidates(panel: pd.DataFrame, date: pd.Timestamp, config: dict) -> pd.DataFrame:
    """Identifies stocks meeting the regime, technical, and fundamental entry rules for a date."""
    entry = config["entry"]
    filters = config.get("fundamental_filters", {})
    use_fundamentals = config.get("use_fundamentals", True)
    use_fundamental_filter = config.get("use_fundamental_filter", True)
    universe_filter = config.get("fundamental_universe_filter", {})
    allowed_regimes = set(entry.get("allowed_market_regimes", ["bull", "neutral"]))

    day = panel[panel["date"] == date].copy()
    if day.empty:
        return day

    rs20_rank = day["rs20_rank_pct"]
    rs60_rank = day["rs60_rank_pct"]
    rank_count = len(day)

    # Localize RS ranking inside the EVA Pool if configured
    if (
        use_fundamentals
        and universe_filter.get("enabled", False)
        and entry.get("rs_rank_scope") == "fundamental_universe"
    ):
        univ_mask = _fundamental_universe_mask(day, universe_filter)
        pool_count = int(univ_mask.sum())
        if pool_count > 0:
            rs20_rank = day["ret20"].where(univ_mask).rank(ascending=False, pct=True)
            rs60_rank = day["ret60"].where(univ_mask).rank(ascending=False, pct=True)
            rank_count = pool_count
        else:
            # BUG-26: Empty fundamental universe pool must yield no candidates
            return day.head(0)

    rs20_cutoff = max(entry["rs20_top_pct"], 1 / rank_count)
    rs60_cutoff = max(entry["rs60_top_pct"], 1 / rank_count)

    mask = (
        (day["market_regime"].isin(allowed_regimes))
        & (day["score"] >= entry["min_score"])
        & (rs20_rank <= rs20_cutoff)
        & (rs60_rank <= rs60_cutoff)
        & (day["Close"] > day["ma20"])
        & (day["ma20"] > day["ma60"])
        & (day["ma20_slope"] > 0)
        & (day["rsi14"].between(entry["rsi_min"], entry["rsi_max"]))
        & (day["ret20"] > day["benchmark_ret20"])
        & (day["ret60"] > day["benchmark_ret60"])
        & (day["ret5"] <= entry["ret5_max"])
        & (day["ret20"] <= entry["ret20_max"])
        & (day["volume_ratio"].between(entry["volume_ratio_min"], entry["volume_ratio_max"]))
        & (day["ma20_deviation"].abs() <= entry["ma20_deviation_max"])
    )

    if use_fundamentals and use_fundamental_filter:
        mask = mask & (day["fundamental_score"] >= filters["min_fundamental_score"]) & day["fundamental_pass"]
    if use_fundamentals and universe_filter.get("enabled", False):
        mask = mask & _fundamental_universe_mask(day, universe_filter)

    return day[mask].sort_values(["final_score", "score"], ascending=False)
