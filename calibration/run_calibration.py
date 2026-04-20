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


# =========================
# CALIBRATION CONSTANTS
# =========================
MARKET_MOVE_THRESHOLD = 0.0015
HORIZON_DAYS = 5
CURRENT_TEST_THRESHOLD = 0.20

ALLOWED_MONTHS = set(range(1, 13))


# =========================
# STATE NORMALIZER
# =========================
def normalize_initial_state(state: dict):
    return {
        **state,
        "raw_article_count": 0,
        "ce_output": state.get("ce_output") or {},
        "tts_output": state.get("tts_output") or {},
        "siv_output": state.get("siv_output") or {},
        "debug_log": state.get("debug_log") or [],
        "retry_count": state.get("retry_count", 0),
        "action": state.get("action", "NONE"),
    }


# =========================
# MAIN CALIBRATION
# =========================
def run_calibration(target_pair: str, target_months: list, target_year: int):

    if not target_months:
        raise ValueError("target_months cannot be empty")

    if not all(m in ALLOWED_MONTHS for m in target_months):
        raise ValueError(f"Invalid month detected: {target_months}")

    project_root = Path(__file__).resolve().parents[1]
    ohlcv_path = project_root / "data" / "calibration" / "forex_pair" / f"{target_pair}.json"

    months_tag = "_".join(map(str, target_months))

    report_output_path = (
        project_root / "reports"
        / f"calibration_{target_pair}_{target_year}_{months_tag}.csv"
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

    experiment_config = {
        "pair": target_pair,
        "year": target_year,
        "months": target_months,
        "horizon_days": HORIZON_DAYS,
        "market_threshold": MARKET_MOVE_THRESHOLD,
        "test_threshold": CURRENT_TEST_THRESHOLD
    }

    print("\n--- STARTING CALIBRATION ---")

    for idx in calib_df.index:

        row = full_df.iloc[idx]

        current_date = row["timestamp"].strftime("%m/%d/%Y")
        current_price = float(row["close"])

        future_window = full_df.iloc[idx + 1: idx + 1 + HORIZON_DAYS]

        market_label = "NEUTRAL"

        if not future_window.empty:
            max_high = future_window["high"].max()
            min_low = future_window["low"].min()

            upside = (max_high - current_price) / current_price
            downside = (current_price - min_low) / current_price

            if upside > MARKET_MOVE_THRESHOLD and upside > downside:
                market_label = "BULLISH"
            elif downside > MARKET_MOVE_THRESHOLD:
                market_label = "BEARISH"

        initial_state = cast(TradingState, normalize_initial_state({
            "target_date": current_date,
            "currency_pair": target_pair,
            "price": current_price,
            "calibration_threshold": CURRENT_TEST_THRESHOLD,

            "ce_output": {
                "sentiment": "NEUTRAL",
                "raw_vibe": "NEUTRAL",
                "mean_score": 0.0,
                "sentiment_score": 0.0,
                "article_count": 0,
                "raw_article_count": 0,
                "confidence": "LOW",
                "error": None
            },

            "tts_output": {
                "total_score": 0.0,
                "ema_trend": "SIDEWAYS",
                "ema_score": 0.0,
                "rsi_value": 50.0,
                "rsi_score": 0.0,
                "bb_signal": "STABLE",
                "bb_score": 0.0,
                "breakout_score": 0.0,
                "price": current_price,
                "ema_200_confidence": 0.0,
                "ema_200_reliable": False,
                "data_stale": False,
                "rows_available": 0,
                "tts_insufficient": True,
                "error": None
            },

            "siv_output": {
                "signal": "INCOHERENT",
                "conflict_type": "UNCLEAR",
                "price_deviation": 0.0,
                "issues": [],
                "tts_insufficient": True,
                "data_quality_ok": False,
                "explanation": "Not yet run"
            },

            "verdict": "HOLD",
            "verdict_reasoning": "",
            "weighted_score": 0.0,
            "risk_multiplier": 0.0,
        }))

        try:
            print(f"{current_date} | {market_label}")

            final_output = app.invoke(initial_state)

            ce_data = final_output.get("ce_output", {})
            tts_data = final_output.get("tts_output", {})
            siv_data = final_output.get("siv_output", {})

            verdict = final_output.get("verdict", "HOLD").upper()
            weighted_score = final_output.get("weighted_score", 0.0)

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

            print(f"DEBUG: {current_date} | SIV={siv_data.get('signal')} | VERDICT={verdict} | SCORE={weighted_score}")

            results.append({
                "Date": current_date,
                "Price": current_price,

                "Sentiment": ce_data.get("sentiment"),
                "Articles": ce_data.get("article_count", 0),

                "Tech_Score": tts_data.get("total_score", 0.0),
                "Weighted_Score": weighted_score,

                "SIV_Signal": siv_data.get("signal"),
                "Final_Verdict": verdict,

                "Market_Reality": market_label,
                "Correct": correct,

                "Experiment_Config": str(experiment_config)
            })

        except Exception as e:
            print(f"Failed on {current_date}: {e}")

    df = pd.DataFrame(results)
    df.to_csv(report_output_path, index=False)

    # =========================
    # FINAL ACCURACY REPORT
    # =========================
    def acc(c, t):
        return round((c / t) * 100, 2) if t else 0.0

    print("\n================ TRADE ACCURACY ================")
    print(f"BUY Accuracy  : {acc(buy_correct, buy_total)}% ({buy_correct}/{buy_total})")
    print(f"SELL Accuracy : {acc(sell_correct, sell_total)}% ({sell_correct}/{sell_total})")
    print(f"HOLD Accuracy : {acc(hold_correct, hold_total)}% ({hold_correct}/{hold_total})")

    total_correct = buy_correct + sell_correct + hold_correct
    total_trades = buy_total + sell_total + hold_total

    print(f"OVERALL ACCURACY: {acc(total_correct, total_trades)}%")
    print("=================================================\n")

    print("CALIBRATION COMPLETE")
    print(report_output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("pair", nargs="?", default="USDJPY")
    parser.add_argument("-m", "--months", default="8,9,10")
    parser.add_argument("-y", "--year", type=int, default=2018)

    args = parser.parse_args()
    months = [int(x) for x in args.months.split(",")]

    run_calibration(args.pair.upper(), months, args.year)