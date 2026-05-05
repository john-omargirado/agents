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
MARKET_MOVE_THRESHOLD = 0.0025
HORIZON_DAYS = 3
CURRENT_TEST_THRESHOLD = 0.05

VOLATILITY_PERCENTILE = 35
MIN_SIGNAL_CONFIDENCE = 0.05
MINIMUM_MOVE_FOR_TRADE = 0.0010
MIN_VERDICT_STRENGTH = 0.06

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
        "skip_llm": state.get("skip_llm", False),
    }


# =========================
# GOOD TRADING DAY DETECTOR
# =========================
def is_good_trading_day(df: pd.DataFrame, idx: int, atr_val: float, atr_percentile: float) -> bool:
    if atr_val < atr_percentile:
        return False

    future_window = df.iloc[idx + 1: idx + 1 + HORIZON_DAYS]
    if future_window.empty:
        return False

    current_price = float(df.iloc[idx]["close"])
    max_high = future_window["high"].max()
    min_low = future_window["low"].min()

    upside = (max_high - current_price) / current_price
    downside = (current_price - min_low) / current_price

    total_move = upside + downside
    return total_move > MINIMUM_MOVE_FOR_TRADE


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

    high_low   = full_df["high"] - full_df["low"]
    high_close = (full_df["high"] - full_df["close"].shift()).abs()
    low_close  = (full_df["low"]  - full_df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    full_df["atr14"] = tr.ewm(span=14, adjust=False).mean()

    mask = (
        (full_df["timestamp"].dt.year == target_year) &
        (full_df["timestamp"].dt.month.isin(target_months))
    )

    calib_df = full_df[mask]

    if calib_df.empty:
        print("No data found for selection")
        return

    atr_percentile_threshold = full_df["atr14"].quantile(VOLATILITY_PERCENTILE / 100)

    app = build_graph()
    results = []

    buy_correct = buy_total = 0
    sell_correct = sell_total = 0
    hold_correct = hold_total = 0

    good_buy_correct = good_buy_total = 0
    good_sell_correct = good_sell_total = 0
    good_hold_correct = good_hold_total = 0

    print("\n--- STARTING CALIBRATION ---")
    print(f"ATR Percentile Threshold: {atr_percentile_threshold:.4f}")

    for idx in calib_df.index:

        row = full_df.iloc[idx]
        atr_val = full_df.iloc[idx]["atr14"] if pd.notna(full_df.iloc[idx]["atr14"]) else 0.0
        atr_val = float(atr_val) if not isinstance(atr_val, (int, float)) else float(atr_val)

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
            "backtest_mode": False,
            "skip_llm": True,
            "atr": atr_val,
            "ce_output": {
                "ce_score": 0.0,
                "ce_confidence": 0.0
            },
            "tts_output": {
                "tts_score": 0.0,
                "price": current_price
            },
            "siv_output": {
                "signal": "COHERENT",
                "risk_penalty": 0.0,
                "issues": [],
                "score_multiplier": 1.0
            },
            "verdict": "HOLD",
            "weighted_score": 0.0,
            "risk_multiplier": 0.0,
        }))

        try:
            print(f"{current_date} | {market_label}")

            final_output = app.invoke(initial_state)

            ce_data = final_output.get("ce_output", {})
            tts_data = final_output.get("tts_output", {})
            siv_data = final_output.get("siv_output", {})

            action      = final_output.get("action", "NONE")
            raw_verdict = final_output.get("verdict", "HOLD").upper()
            verdict     = raw_verdict
            final_score = final_output.get("weighted_score", 0.0)

            # ✅ FIX 1: `continue` is now correctly inside the if block
            if action == "SKIP":
                results.append({
                    "Date": current_date,
                    "Price": current_price,
                    "ATR": atr_val,
                    "CE_Score": ce_data.get("ce_score", 0.0),
                    "TTS_Score": tts_data.get("tts_score", 0.0),
                    "Final_Score": 0.0,
                    "SIV_Signal": siv_data.get("signal"),
                    "Final_Verdict": "SKIP",
                    "Market_Reality": market_label,
                    "Correct": None,
                })
                continue  # ← now inside the if block

            # =========================
            # ACCURACY LOGIC
            # =========================
            correct = False

            ce_raw = float(ce_data.get("ce_score", 0.0))
            tts_raw = float(tts_data.get("tts_score", 0.0))
            ce_score = abs(ce_raw)
            tts_score = abs(tts_raw)
            combined_confidence = (ce_score + tts_score) / 2.0

            is_strong_verdict = abs(final_score) > MIN_VERDICT_STRENGTH

            signals_agree = (ce_raw * tts_raw) > 0.0
            trend_ok = True
            siv_ok = True

            trade_allowed = (
                verdict in ["BUY", "SELL"]
                and is_strong_verdict
                and (combined_confidence >= MIN_SIGNAL_CONFIDENCE)
                and trend_ok
                and siv_ok
            )
            if verdict in ["BUY", "SELL"] and not trade_allowed:
                verdict = "HOLD"

            # ✅ FIX 3: separate flag so good_hold is not gated by trade_allowed
            is_active_trade = trade_allowed and verdict in ["BUY", "SELL"]

            if verdict == "BUY":
                buy_total += 1
                correct = market_label == "BULLISH"
                if correct:
                    buy_correct += 1
                if is_active_trade:
                    good_buy_total += 1
                    if correct:
                        good_buy_correct += 1

            elif verdict == "SELL":
                sell_total += 1
                correct = market_label == "BEARISH"
                if correct:
                    sell_correct += 1
                if is_active_trade:
                    good_sell_total += 1
                    if correct:
                        good_sell_correct += 1

            else:  # HOLD
                hold_total += 1
                correct = market_label == "NEUTRAL"
                if correct:
                    hold_correct += 1
                good_hold_total += 1  # all HOLDs count as "good" (not overridden)
                if correct:
                    good_hold_correct += 1

            print(
                f"DEBUG: {current_date} | "
                f"ATR={atr_val:.4f} | "
                f"STRONG={is_strong_verdict} | "
                f"ALLOWED={trade_allowed} | "
                f"AGREE={signals_agree} | "
                f"TREND_OK={trend_ok} | "
                f"SIV_OK={siv_ok} | "
                f"CONF={combined_confidence:.2f} | "
                f"SCORE={final_score:.4f} | "
                f"SIV={siv_data.get('signal')} | "
                f"RAW={raw_verdict} | "
                f"VERDICT={verdict}"
            )

            results.append({
                "Date": current_date,
                "Price": current_price,
                "Strong_Verdict": is_strong_verdict,
                "Trade_Allowed": trade_allowed,
                "Raw_Verdict": raw_verdict,
                "Signals_Agree": signals_agree,
                "Trend_OK": trend_ok,
                "ATR": atr_val,
                "Signal_Confidence": combined_confidence,
                "CE_Score": ce_data.get("ce_score", 0.0),
                "CE_Confidence": ce_data.get("ce_confidence", 0.0),
                "TTS_Score": tts_data.get("tts_score", 0.0),
                "Final_Score": final_score,
                "SIV_Signal": siv_data.get("signal"),
                "Risk_Penalty": siv_data.get("risk_penalty", 0.0),
                "Final_Verdict": verdict,
                "Market_Reality": market_label,
                "Correct": correct,
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

    print("\n" + "="*60)
    print("OVERALL ACCURACY (ALL DAYS)")
    print("="*60)
    print(f"BUY Accuracy  : {acc(buy_correct, buy_total)}% ({buy_correct}/{buy_total})")
    print(f"SELL Accuracy : {acc(sell_correct, sell_total)}% ({sell_correct}/{sell_total})")
    print(f"HOLD Accuracy : {acc(hold_correct, hold_total)}% ({hold_correct}/{hold_total})")

    all_directional_correct = buy_correct + sell_correct
    all_directional_trades = buy_total + sell_total
    print(f"BUY+SELL Accuracy (Directional): {acc(all_directional_correct, all_directional_trades)}% ({all_directional_correct}/{all_directional_trades})")

    total_correct = buy_correct + sell_correct + hold_correct
    total_trades = buy_total + sell_total + hold_total
    print(f"OVERALL ACCURACY: {acc(total_correct, total_trades)}%")

    # ✅ FIX 2: Added missing section title
    print("\n" + "="*60)
    print("ACCURACY ON QUALITY-GATED TRADES ONLY")
    print("="*60)
    print(f"BUY Accuracy  : {acc(good_buy_correct, good_buy_total)}% ({good_buy_correct}/{good_buy_total})")
    print(f"SELL Accuracy : {acc(good_sell_correct, good_sell_total)}% ({good_sell_correct}/{good_sell_total})")
    print(f"HOLD Accuracy : {acc(good_hold_correct, good_hold_total)}% ({good_hold_correct}/{good_hold_total})")

    good_directional_correct = good_buy_correct + good_sell_correct
    good_directional_trades = good_buy_total + good_sell_total
    print(f"GOOD DAYS BUY+SELL Accuracy (Directional): {acc(good_directional_correct, good_directional_trades)}% ({good_directional_correct}/{good_directional_trades})")

    good_total_correct = good_buy_correct + good_sell_correct + good_hold_correct
    good_total_trades = good_buy_total + good_sell_total + good_hold_total
    print(f"GOOD DAYS OVERALL ACCURACY: {acc(good_total_correct, good_total_trades)}%")
    print("="*60 + "\n")

    skipped = len(df[df["Final_Verdict"] == "SKIP"])
    traded  = len(df[df["Final_Verdict"] != "SKIP"])
    print(f"Total days    : {len(df)}")
    print(f"Skipped       : {skipped} ({round(skipped/len(df)*100,1)}% filtered by quality gate)")
    print(f"Traded        : {traded}")

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