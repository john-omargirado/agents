import sys
from pathlib import Path
import json
import pandas as pd
import os

# Ensure project root is in path for internal imports
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from graph.build_graph import build_graph
from state.trading_state import TradingState

# --- CALIBRATION CONSTANTS ---
MARKET_MOVE_THRESHOLD = 0.0015
HORIZON_DAYS = 5
CURRENT_TEST_THRESHOLD = 0.20

def run_calibration(target_pair: str, target_months: list, target_year: int):
    # 1. PATHS
    project_root = Path(__file__).resolve().parents[1]
    ohlcv_path = project_root / "data" / "calibration" / "forex_pair" / f"{target_pair}.json"
    
    months_tag = "_".join([str(m) for m in target_months])
    report_output_path = project_root / "reports" / f"calibration_{target_pair}_{target_year}_{months_tag}.csv"
    os.makedirs(project_root / "reports", exist_ok=True)

    # 2. LOAD DATA
    if not ohlcv_path.exists():
        print(f"Error: Could not find data at {ohlcv_path}")
        return

    with open(ohlcv_path, "r") as f:
        master_data = json.load(f)

    full_df = pd.DataFrame(master_data["data"])
    full_df.columns = [c.lower() for c in full_df.columns]
    full_df["timestamp"] = pd.to_datetime(full_df["timestamp"])
    full_df = full_df.sort_values(by="timestamp").reset_index(drop=True)

    # 3. FILTER DATE RANGE
    mask = (
        (full_df["timestamp"].dt.year == target_year) &
        (full_df["timestamp"].dt.month.isin(target_months))
    )
    calib_df = full_df[mask].copy()

    if calib_df.empty:
        print(f"No data found for months {target_months}/{target_year}")
        return

    calib_indices = calib_df.index.tolist()

    # 4. INITIALIZE GRAPH
    app = build_graph()
    results = []

    print(f"--- STARTING CALIBRATION: {target_pair} | {target_year} ---")

    # 5. LOOP THROUGH HISTORICAL DATA
    for idx in calib_indices:
        row = full_df.iloc[idx]
        current_date = row["timestamp"].strftime("%m/%d/%Y")
        current_price = float(row["close"])

        # --- FUTURE MARKET LABEL (Ground Truth) ---
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

        # --- INITIAL STATE FOR AGENTS ---
        initial_state: TradingState = {
            "target_date": current_date,
            "currency_pair": target_pair,
            "price": current_price,
            "calibration_threshold": CURRENT_TEST_THRESHOLD,
            "ce_output": {},
            "tts_output": {},
            "siv_output": {},
            "verdict": "HOLD",
            "verdict_reasoning": "",
            "risk_multiplier": 0.0,
            "retry_count": 0,    # FIX: must be initialized so verdict_agent can read and increment it
            "action": "NONE",    # FIX: must be initialized so route_after_verdict doesn't get None
            "debug_log": [f"Calibration Start: {current_date}"]
        }

        try:
            print(f"Processing: {current_date} | Reality: {market_label}")
            final_output = app.invoke(initial_state)

            ce_data = final_output.get("ce_output", {})
            tts_data = final_output.get("tts_output", {})
            siv_data = final_output.get("siv_output", {})
            
            verdict = final_output.get("verdict", "HOLD").upper()
            verdict_reasoning = final_output.get("verdict_reasoning", "")
            risk_multiplier = final_output.get("risk_multiplier", 0.0)

            # --- ALIGNMENT CALCULATION ---
            alignment = "neutral"
            if verdict == "BUY":
                alignment = "correct" if market_label == "BULLISH" else "incorrect"
            elif verdict == "SELL":
                alignment = "correct" if market_label == "BEARISH" else "incorrect"
            elif verdict == "HOLD" and market_label == "NEUTRAL":
                alignment = "correct"

            results.append({
                "Date": current_date,
                "Price": current_price,
                "SIV_Signal": siv_data.get("integrity_signal", "N/A"),
                "SIV_Audit": siv_data.get("explanation", "").replace("\n", " "),
                "Sentiment": ce_data.get("overall_sentiment", "N/A"),
                "Articles": ce_data.get("articles_analyzed", 0),
                "Tech_Decision": tts_data.get("decision", "HOLD"),
                "Tech_Thought": tts_data.get("reasoning", ""),
                "Final_Verdict": verdict,
                "Risk_Multiplier": risk_multiplier,
                "Verdict_Reasoning": verdict_reasoning,
                "Market_Reality": market_label,
                "Alignment": alignment
            })

        except Exception as e:
            print(f"Failed on {current_date}: {e}")

    # 6. EXPORT TO CSV
    report_df = pd.DataFrame(results)
    report_df.to_csv(report_output_path, index=False)

    # 7. SUMMARY STATS
    active_trades = report_df[report_df["Final_Verdict"] != "HOLD"]
    accuracy = (active_trades["Alignment"] == "correct").mean() if not active_trades.empty else 0

    print("\n" + "=" * 40)
    print(f"CALIBRATION COMPLETE: {target_pair}")
    print(f"Report Saved: {report_output_path}")
    print(f"Total Active Trades: {len(active_trades)}")
    print(f"Accuracy: {accuracy:.2%}")
    print("=" * 40)

if __name__ == "__main__":
    run_calibration(
        target_pair="USDJPY",
        target_months=[4], # Starting with January for a quick test
        target_year=2018
    )