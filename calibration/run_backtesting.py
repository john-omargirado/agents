import sys
from pathlib import Path
import json
import pandas as pd
from typing import cast
import os

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from graph.build_graph import build_graph
from state.trading_state import TradingState
from utils.trade_config import ATR_LOOKBACK, get_pair_config


# =========================
# BACKTEST CONSTANTS
# =========================
MARKET_MOVE_THRESHOLD = 0.0015
HORIZON_DAYS = 5
CURRENT_TEST_THRESHOLD = 0.20

ALLOWED_MONTHS = set(range(1, 13))


# =========================
# ATR COMPUTATION
# =========================
def compute_atr(lookback_df: pd.DataFrame, period: int = 14) -> float:
    df = lookback_df.tail(period + 1).copy()
    if len(df) < 2:
        return 0.0
    df["prev_close"] = df["close"].shift(1)
    df["tr"] = df[["high", "low", "prev_close"]].apply(
        lambda r: max(
            r["high"] - r["low"],
            abs(r["high"] - r["prev_close"]),
            abs(r["low"] - r["prev_close"])
        ), axis=1
    )
    return float(df["tr"].tail(period).mean())


# =========================
# TRADE SIMULATION
# SL_Adaptive = P_entry +/- (k x ATR)
# TP = P_entry +/- (SL_distance x RR_ratio)
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

    for bars_held, (_, candle) in enumerate(future_window.iterrows(), start=1):
        if direction == "BUY":
            if candle["low"] <= sl:
                return sl, "SL", bars_held
            if candle["high"] >= tp:
                return tp, "TP", bars_held
        else:
            if candle["high"] >= sl:
                return sl, "SL", bars_held
            if candle["low"] <= tp:
                return tp, "TP", bars_held

    exit_price = future_window.iloc[-1]["close"] if not future_window.empty else entry_price
    return exit_price, "TIME", len(future_window)


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

    if not target_months:
        raise ValueError("target_months cannot be empty")

    if not all(m in ALLOWED_MONTHS for m in target_months):
        raise ValueError(f"Invalid month detected: {target_months}")

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

    if not ohlcv_path.exists():
        print(f"Missing data: {ohlcv_path}")
        return

    with open(ohlcv_path, "r") as f:
        master_data = json.load(f)

    full_df = pd.DataFrame(master_data["data"])
    full_df.columns = [c.lower() for c in full_df.columns]
    full_df["timestamp"] = pd.to_datetime(full_df["timestamp"])
    full_df = full_df.sort_values("timestamp").reset_index(drop=True)

    mask = (
        (full_df["timestamp"].dt.year == target_year) &
        (full_df["timestamp"].dt.month.isin(target_months))
    )
    calib_df = full_df[mask]

    if calib_df.empty:
        print("No data found for selection")
        return

    app = build_graph()
    results = []

    # =========================
    # ACCURACY TRACKING
    # =========================
    buy_correct = buy_total = 0
    sell_correct = sell_total = 0
    hold_correct = hold_total = 0
    tp_hits = sl_hits = time_exits = hold_skips = 0

    experiment_config = {
        "pair": target_pair,
        "year": target_year,
        "months": target_months,
        "horizon_days": HORIZON_DAYS,
        "market_threshold": MARKET_MOVE_THRESHOLD,
        "test_threshold": CURRENT_TEST_THRESHOLD,
        "sl_mult": sl_mult,
        "rr_ratio": rr_ratio,
    }

    print("\n--- STARTING BACKTEST ---")

    for idx in calib_df.index:

        row = full_df.iloc[idx]

        current_date = row["timestamp"].strftime("%m/%d/%Y")
        entry_price = float(row["close"])
        future_window = full_df.iloc[idx + 1: idx + 1 + HORIZON_DAYS]

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

        # ATR
        lookback_df = full_df.iloc[max(0, idx - ATR_LOOKBACK - 1): idx]
        atr = compute_atr(lookback_df, period=ATR_LOOKBACK)
        sl_distance = round(atr * sl_mult, 5)
        tp_distance = round(sl_distance * rr_ratio, 5)

        initial_state = cast(TradingState, normalize_initial_state({
            "target_date":           current_date,
            "currency_pair":         target_pair,
            "backtest_mode":         True,
            "price":                 entry_price,
            "calibration_threshold": CURRENT_TEST_THRESHOLD,
            "atr":                   atr,

            "ce_output": {
                "sentiment": "NEUTRAL", "raw_vibe": "NEUTRAL",
                "mean_score": 0.0, "sentiment_score": 0.0,
                "article_count": 0, "raw_article_count": 0,
                "confidence": "LOW", "error": None
            },
            "tts_output": {
                "total_score": 0.0, "ema_trend": "SIDEWAYS",
                "ema_score": 0.0, "rsi_value": 50.0, "rsi_score": 0.0,
                "bb_signal": "STABLE", "bb_score": 0.0, "breakout_score": 0.0,
                "price": entry_price, "ema_200_confidence": 0.0,
                "ema_200_reliable": False, "data_stale": False,
                "rows_available": 0, "tts_insufficient": True, "error": None
            },
            "siv_output": {
                "signal": "INCOHERENT", "conflict_type": "UNCLEAR",
                "price_deviation": 0.0, "issues": [],
                "tts_insufficient": True, "data_quality_ok": False,
                "explanation": "Not yet run"
            },

            "verdict": "HOLD", "verdict_reasoning": "",
            "weighted_score": 0.0, "risk_multiplier": 0.0,
        }))

        try:
            print(f"{current_date} | {market_label}")

            final_output = app.invoke(initial_state)

            ce_data = final_output.get("ce_output", {})
            tts_data = final_output.get("tts_output", {})
            siv_data = final_output.get("siv_output", {})
            verdict = final_output.get("verdict", "HOLD").upper()
            weighted_score = final_output.get("weighted_score", 0.0)

            exit_reason = "HOLD_SKIP"
            bars_held = 0

            if verdict in ("BUY", "SELL") and not future_window.empty and sl_distance > 0.0:
                _, exit_reason, bars_held = simulate_trade(
                    verdict,
                    entry_price,
                    future_window,
                    sl_distance,
                    tp_distance,
                )
                if exit_reason == "TP":
                    tp_hits += 1
                elif exit_reason == "SL":
                    sl_hits += 1
                else:
                    time_exits += 1
            else:
                hold_skips += 1

            # =========================
            # ACCURACY LOGIC
            # =========================
            correct = False

            if verdict == "BUY":
                buy_total += 1
                correct = market_label == "BULLISH"
                if correct:
                    buy_correct += 1

            elif verdict == "SELL":
                sell_total += 1
                correct = market_label == "BEARISH"
                if correct:
                    sell_correct += 1

            else:
                hold_total += 1
                correct = market_label == "NEUTRAL"
                if correct:
                    hold_correct += 1

            print(
                f"DEBUG: {current_date} | SIV={siv_data.get('signal')} | "
                f"VERDICT={verdict} | SCORE={weighted_score} | "
                f"SL={sl_distance} | TP={tp_distance} | EXIT={exit_reason}"
            )

            results.append({
                "Date": current_date,
                "Price": entry_price,

                "Sentiment": ce_data.get("sentiment"),
                "Articles": ce_data.get("article_count", 0),
                "CE_Explanation": ce_data.get("explanation", ""),

                "Tech_Score": tts_data.get("total_score", 0.0),
                "Weighted_Score": weighted_score,
                "TTS_Explanation": tts_data.get("explanation", ""),

                "SIV_Signal": siv_data.get("signal"),
                "SIV_Explanation": siv_data.get("explanation", ""),
                "Final_Verdict": verdict,
                "Verdict_Reasoning": final_output.get("verdict_reasoning", ""),

                "Market_Reality": market_label,
                "Correct": correct,

                "SL_Distance": sl_distance,
                "TP_Distance": tp_distance,
                "Exit_Reason": exit_reason,
                "Bars_Held": bars_held,

                "Experiment_Config": str(experiment_config)
            })

        except Exception as e:
            print(f"Failed on {current_date}: {e}")

    # =========================
    # ACCURACY HELPERS
    # =========================
    def acc(c, t):
        return round((c / t) * 100, 2) if t else 0.0

    total_correct = buy_correct + sell_correct + hold_correct
    total_trades = buy_total + sell_total + hold_total
    buysell_correct = buy_correct + sell_correct
    buysell_total = buy_total + sell_total

    # =========================
    # APPEND SUMMARY ROWS TO CSV
    # =========================
    df = pd.DataFrame(results)

    summary_rows = pd.DataFrame([
        {"Date": "---", "Final_Verdict": "ACCURACY SUMMARY"},
        {
            "Date": "BUY Accuracy",
            "Final_Verdict": f"{acc(buy_correct, buy_total)}%",
            "Correct": f"{buy_correct}/{buy_total}"
        },
        {
            "Date": "SELL Accuracy",
            "Final_Verdict": f"{acc(sell_correct, sell_total)}%",
            "Correct": f"{sell_correct}/{sell_total}"
        },
        {
            "Date": "BUY+SELL Accuracy",
            "Final_Verdict": f"{acc(buysell_correct, buysell_total)}%",
            "Correct": f"{buysell_correct}/{buysell_total}"
        },
        {
            "Date": "HOLD Accuracy",
            "Final_Verdict": f"{acc(hold_correct, hold_total)}%",
            "Correct": f"{hold_correct}/{hold_total}"
        },
        {
            "Date": "OVERALL Accuracy",
            "Final_Verdict": f"{acc(total_correct, total_trades)}%",
            "Correct": f"{total_correct}/{total_trades}"
        },
        {"Date": "---", "Final_Verdict": "SL/TP SUMMARY"},
        {"Date": "TP Hits",     "Correct": tp_hits},
        {"Date": "SL Hits",     "Correct": sl_hits},
        {"Date": "TIME Exits",  "Correct": time_exits},
        {"Date": "HOLD Skips",  "Correct": hold_skips},
    ])

    df = pd.concat([df, summary_rows], ignore_index=True)
    df.to_csv(report_output_path, index=False)

    # =========================
    # FINAL ACCURACY REPORT
    # =========================
    print("\n================ TRADE ACCURACY ================")
    print(f"BUY Accuracy     : {acc(buy_correct, buy_total)}% ({buy_correct}/{buy_total})")
    print(f"SELL Accuracy    : {acc(sell_correct, sell_total)}% ({sell_correct}/{sell_total})")
    print(f"BUY+SELL Accuracy: {acc(buysell_correct, buysell_total)}% ({buysell_correct}/{buysell_total})")
    print(f"HOLD Accuracy    : {acc(hold_correct, hold_total)}% ({hold_correct}/{hold_total})")
    print(f"OVERALL ACCURACY : {acc(total_correct, total_trades)}%")
    print(f"SL/TP Summary    : TP={tp_hits} | SL={sl_hits} | TIME={time_exits} | HOLD_SKIP={hold_skips}")
    print("=================================================\n")

    print("BACKTEST COMPLETE")
    print(report_output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("pair", nargs="?", default="USDJPY")
    parser.add_argument("-m", "--months", default="8,9,10")
    parser.add_argument("-y", "--year", type=int, default=2018)

    args = parser.parse_args()
    months = [int(x) for x in args.months.split(",")]

    run_backtest(args.pair.upper(), months, args.year)