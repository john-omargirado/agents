import sys
from pathlib import Path
import json
import pandas as pd
import os
import csv

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

ALLOWED_MONTHS = set(range(1, 13))  # safety guard (1-12)


def run_calibration(target_pair: str, target_months: list, target_year: int):

    # =========================
    # VALIDATE MONTH INPUT
    # =========================
    if not target_months:
        raise ValueError("target_months cannot be empty")

    if not all(m in ALLOWED_MONTHS for m in target_months):
        raise ValueError(f"Invalid month detected. Allowed range is 1-12. Got: {target_months}")

    # =========================
    # PATH SETUP
    # =========================
    project_root = Path(__file__).resolve().parents[1]
    ohlcv_path = project_root / "data" / "calibration" / "forex_pair" / f"{target_pair}.json"

    months_tag = "_".join([str(m) for m in target_months])

    report_output_path = (
        project_root
        / "reports"
        / f"calibration_{target_pair}_{target_year}_{months_tag}.csv"
    )

    os.makedirs(project_root / "reports", exist_ok=True)

    # =========================
    # LOAD DATA
    # =========================
    if not ohlcv_path.exists():
        print(f"Error: Could not find data at {ohlcv_path}")
        return

    with open(ohlcv_path, "r") as f:
        master_data = json.load(f)

    full_df = pd.DataFrame(master_data["data"])
    full_df.columns = [c.lower() for c in full_df.columns]
    full_df["timestamp"] = pd.to_datetime(full_df["timestamp"])
    full_df = full_df.sort_values(by="timestamp").reset_index(drop=True)

    # =========================
    # FILTER RANGE
    # =========================
    mask = (
        (full_df["timestamp"].dt.year == target_year) &
        (full_df["timestamp"].dt.month.isin(target_months))
    )

    calib_df = full_df[mask].copy()

    if calib_df.empty:
        print(f"No data found for months {target_months}/{target_year}")
        return

    calib_indices = calib_df.index.tolist()

    # =========================
    # EXPERIMENT METADATA
    # =========================
    experiment_config = {
        "pair": target_pair,
        "year": target_year,
        "months": target_months,
        "horizon_days": HORIZON_DAYS,
        "market_threshold": MARKET_MOVE_THRESHOLD,
        "test_threshold": CURRENT_TEST_THRESHOLD
    }

    print(f"\n--- STARTING CALIBRATION ---")
    print(f"PAIR: {target_pair}")
    print(f"YEAR: {target_year}")
    print(f"MONTHS: {target_months}")
    print(f"CONFIG: {experiment_config}")

    # =========================
    # GRAPH
    # =========================
    app = build_graph()
    results = []

    # =========================
    # MAIN LOOP
    # =========================
    for idx in calib_indices:

        row = full_df.iloc[idx]
        current_date = row["timestamp"].strftime("%m/%d/%Y")
        current_price = float(row["close"])

        future_window = full_df.iloc[idx + 1: idx + 1 + HORIZON_DAYS]
        market_label = "NEUTRAL"

        if not future_window.empty:
            max_high = pd.to_numeric(future_window["high"]).max()
            min_low = pd.to_numeric(future_window["low"]).min()

            upside = (max_high - current_price) / current_price
            downside = (current_price - min_low) / current_price

            if upside > MARKET_MOVE_THRESHOLD and upside > downside:
                market_label = "BULLISH"
            elif downside > MARKET_MOVE_THRESHOLD and downside > upside:
                market_label = "BEARISH"

        initial_state: TradingState = {
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
                "price": 0.0,
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
            "retry_count": 0,
            "action": "NONE",
            "debug_log": [
                f"Calibration Start: {current_date}",
                f"Experiment Config: {experiment_config}"
            ]
        }

        try:
            print(f"Processing: {current_date} | Reality: {market_label}")

            final_output = app.invoke(initial_state)

            ce_data  = final_output.get("ce_output", {})
            tts_data = final_output.get("tts_output", {})
            siv_data = final_output.get("siv_output", {})

            verdict           = final_output.get("verdict", "HOLD").upper()
            verdict_reasoning = final_output.get("verdict_reasoning", "")
            weighted_score    = final_output.get("weighted_score", 0.0)
            risk_multiplier   = final_output.get("risk_multiplier", 0.0)

            alignment = "neutral"

            if verdict == "BUY":
                alignment = "correct" if market_label == "BULLISH" else "incorrect"
            elif verdict == "SELL":
                alignment = "correct" if market_label == "BEARISH" else "incorrect"
            elif verdict == "HOLD":
                alignment = "correct" if market_label == "NEUTRAL" else "incorrect"

            results.append({
                "Date": current_date,
                "Price": current_price,

                "Sentiment": ce_data.get("sentiment", "N/A"),
                "CE_Confidence": ce_data.get("confidence", "N/A"),
                "Articles": ce_data.get("article_count", 0),

                "Tech_Decision": tts_data.get("decision", "HOLD"),
                "Tech_Score": tts_data.get("total_score", 0.0),

                "SIV_Signal": siv_data.get("signal", "N/A"),

                "Final_Verdict": verdict,
                "Weighted_Score": weighted_score,
                "Risk_Multiplier": risk_multiplier,
                "Verdict_Reasoning": verdict_reasoning,

                "Market_Reality": market_label,
                "Alignment": alignment,

                # IMPORTANT: experiment traceability
                "Experiment_Config": str(experiment_config)
            })

        except Exception as e:
            print(f"Failed on {current_date}: {e}")

    # =========================
    # EXPORT
    # =========================
    report_df = pd.DataFrame(results)
    report_df.to_csv(report_output_path, index=False)

    print("\n==============================")
    print(f"CALIBRATION COMPLETE: {target_pair}")
    print(f"Months Used: {target_months}")
    print(f"Saved: {report_output_path}")
    print("==============================")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run calibration for a currency pair")
    parser.add_argument("pair", nargs="?", default="USDJPY", help="Currency pair code (e.g., USDJPY)")
    parser.add_argument("-m", "--months", default="8,9,10", help="Comma-separated months (e.g., 8,9,10)")
    parser.add_argument("-y", "--year", type=int, default=2018, help="Year for calibration")

    args = parser.parse_args()
    try:
        months = [int(x) for x in args.months.split(",") if x.strip()]
    except Exception as e:
        raise SystemExit(f"Invalid months format: {e}")

    run_calibration(args.pair.upper(), months, args.year)