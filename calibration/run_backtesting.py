import sys
from pathlib import Path
import json
import pandas as pd
from typing import cast
import os
import time

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from graph.build_graph import build_graph
from state.trading_state import TradingState
from tools.tts_tools import precompute_indicators
from utils.trade_config import ATR_LOOKBACK, get_pair_config


# =========================
# BACKTEST CONSTANTS
# =========================
MARKET_MOVE_THRESHOLD = 0.0015
HORIZON_DAYS = 5
CURRENT_TEST_THRESHOLD = 0.20
ALLOWED_MONTHS = set(range(1, 13))


# =========================
# AUTO TIMEFRAME DETECTION
# =========================
def infer_candle_minutes(df: pd.DataFrame) -> int:
    if len(df) < 2:
        return 1
    delta = (df["timestamp"].iloc[1] - df["timestamp"].iloc[0]).total_seconds()
    return max(1, int(delta / 60))


# =========================
# TRADE SIMULATION
# =========================
def simulate_trade(
    direction: str,
    entry_price: float,
    future_window: pd.DataFrame,
    sl_distance: float,
    tp_distance: float,
):
    sl = entry_price - sl_distance if direction == "BUY" else entry_price + sl_distance
    tp = entry_price + tp_distance if direction == "BUY" else entry_price - tp_distance

    for _, candle in future_window.iterrows():

        if direction == "BUY":
            if candle["low"] <= sl:
                return sl, "SL"
            if candle["high"] >= tp:
                return tp, "TP"

        else:
            if candle["high"] >= sl:
                return sl, "SL"
            if candle["low"] <= tp:
                return tp, "TP"

    exit_price = future_window.iloc[-1]["close"] if not future_window.empty else entry_price
    return exit_price, "TIME"


# =========================
# STATE NORMALIZER
# =========================
def normalize_initial_state(state: dict):
    return {
        **state,
        "raw_article_count": 0,
        "ce_output": state.get("ce_output") or {},
        "backtest_mode": state.get("backtest_mode", False),
        "tts_output": state.get("tts_output") or {},
        "siv_output": state.get("siv_output") or {},
        "debug_log": state.get("debug_log") or [],
        "retry_count": state.get("retry_count", 0),
        "action": state.get("action", "NONE"),
        "atr": state.get("atr", 0.0),
    }


# =========================
# MAIN BACKTEST
# =========================
def run_backtest(target_pair: str, target_months: list, target_year: int):

    pair_cfg = get_pair_config(target_pair)
    sl_mult = float(pair_cfg.get("sl_mult", 2.0))
    rr_ratio = float(pair_cfg.get("rr_ratio", 1.75))

    project_root = Path(__file__).resolve().parents[1]
    ohlcv_path = project_root / "data" / "backtesting" / "forex_pairs" / f"{target_pair}.json"

    months_tag = "_".join(map(str, target_months))
    report_output_path = (
        project_root / "reports"
        / f"backtest_{target_pair}_{target_year}_{months_tag}.csv"
    )
    os.makedirs(project_root / "reports", exist_ok=True)

    with open(ohlcv_path, "r") as f:
        master_data = json.load(f)

    full_df = pd.DataFrame(master_data["data"])
    full_df.columns = [c.lower() for c in full_df.columns]
    full_df["timestamp"] = pd.to_datetime(full_df["timestamp"])
    full_df = full_df.sort_values("timestamp").reset_index(drop=True)

    # =========================
    # TIMEFRAME DETECTION
    # =========================
    CANDLE_MINUTES = infer_candle_minutes(full_df)
    print(f"[INFO] Detected timeframe: {CANDLE_MINUTES}-minute candles")

    precomputed = precompute_indicators(full_df)
    precomputed = precomputed.reset_index().set_index("timestamp")

    full_df["prev_close"] = full_df["close"].shift(1)

    full_df["tr"] = full_df.apply(
        lambda r: max(
            r["high"] - r["low"],
            abs(r["high"] - r["prev_close"]),
            abs(r["low"] - r["prev_close"])
        ),
        axis=1
    )

    full_df["atr"] = full_df["tr"].rolling(ATR_LOOKBACK).mean()

    future_windows = [
        full_df.iloc[i + 1: i + 1 + HORIZON_DAYS]
        for i in range(len(full_df))
    ]

    mask = (
        (full_df["timestamp"].dt.year == target_year) &
        (full_df["timestamp"].dt.month.isin(target_months))
    )
    calib_df = full_df[mask]

    app = build_graph()
    results = []

    print("\n--- STARTING BACKTEST ---")

    for idx in calib_df.index:

        start_time = time.perf_counter()

        row = full_df.iloc[idx]

        current_date = row["timestamp"].strftime("%m/%d/%Y")
        entry_price = float(row["close"])

        print("\n==============================")
        print(f"[DATE] {current_date}")
        print(f"[ENTRY] {entry_price}")

        future_window = future_windows[idx]

        # =========================
        # MARKET LABEL
        # =========================
        market_label = "NEUTRAL"

        if not future_window.empty:
            max_high = future_window["high"].max()
            min_low = future_window["low"].min()

            upside = (max_high - entry_price) / entry_price
            downside = (entry_price - min_low) / entry_price

            if upside > MARKET_MOVE_THRESHOLD and upside > downside:
                market_label = "BULLISH"
            elif downside > MARKET_MOVE_THRESHOLD:
                market_label = "BEARISH"

        print(f"[MARKET] {market_label}")

        atr = float(full_df.iloc[idx]["atr"])
        sl_distance = round(atr * sl_mult, 5)
        tp_distance = round(sl_distance * rr_ratio, 5)

        initial_state = cast(TradingState, normalize_initial_state({
            "target_date": current_date,
            "currency_pair": target_pair,
            "backtest_mode": True,
            "price": entry_price,
            "atr": atr,
            "precomputed_indicators": precomputed,
        }))

        try:
            final_output = app.invoke(initial_state)

            ce_data = final_output.get("ce_output", {})
            verdict = final_output.get("verdict", "HOLD").upper()
            weighted_score = final_output.get("weighted_score", 0.0)

            print(f"[DECISION] {verdict} | SCORE={weighted_score}")

            exit_reason = "HOLD_SKIP"

            if verdict in ("BUY", "SELL") and not future_window.empty:

                _, exit_reason = simulate_trade(
                    verdict,
                    entry_price,
                    future_window,
                    sl_distance,
                    tp_distance,
                )

            execution_time = round(time.perf_counter() - start_time, 6)

            print(f"[EXECUTION TIME] {execution_time} seconds")

            results.append({
                "Date": current_date,
                "Price": entry_price,
                "Sentiment": ce_data.get("sentiment"),
                "Final_Verdict": verdict,
                "Market_Reality": market_label,
                "Exit_Reason": exit_reason,
                "Execution_Time_Seconds": execution_time,
                "Weighted_Score": weighted_score,
            })

        except Exception as e:
            execution_time = round(time.perf_counter() - start_time, 6)
            print(f"[ERROR] {current_date}: {e}")

            results.append({
                "Date": current_date,
                "Price": entry_price,
                "Final_Verdict": "ERROR",
                "Execution_Time_Seconds": execution_time,
            })

    df = pd.DataFrame(results)
    df.to_csv(report_output_path, index=False)

    print("\nBACKTEST COMPLETE")
    print(report_output_path)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("pair", nargs="?", default="USDJPY")
    parser.add_argument("-m", "--months", default="8,9,10")
    parser.add_argument("-y", "--years", default="2018")

    args = parser.parse_args()
    months = [int(x) for x in args.months.split(",")]
    years = [int(y) for y in args.years.split(",")]

    for year in years:
        print(f"\nRUNNING BACKTEST: {args.pair.upper()} | YEAR={year} | MONTHS={months}")
        run_backtest(args.pair.upper(), months, year)