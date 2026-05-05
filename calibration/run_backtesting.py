import sys
from pathlib import Path
import json
import pandas as pd
from typing import cast
import time
import os
from copy import deepcopy


current_dir  = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from graph.build_graph import build_graph
from state.trading_state import TradingState
from tools.tts_tools import precompute_indicators
from utils.trade_config import ATR_LOOKBACK, get_pair_config
from explanation_pipeline import run_explanation_pipeline


# =========================
# DEBUG MODE
# =========================
DEBUG_BACKTEST = True


def log(stage: str, start: float):
    if DEBUG_BACKTEST:
        print(f"[BT TIMER] {stage}: {time.perf_counter() - start:.4f}s")


# =========================
# CONSTANTS
# Must match calibration settings for consistent signal evaluation
# =========================
MARKET_MOVE_THRESHOLD = 0.0025   # aligned with calibration (was 0.0015)
HORIZON_DAYS          = 3        # aligned with calibration (was 5)

# Verdict threshold — must match CURRENT_TEST_THRESHOLD in run_calibration.py
VERDICT_THRESHOLD = 0.05


# =========================
# TRADE SIMULATION
# =========================
def simulate_trade(
    direction: str,
    entry_price: float,
    future_window: pd.DataFrame,
    sl_distance: float,
    tp_distance: float,
) -> tuple[float, str]:

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

    exit_price = float(future_window.iloc[-1]["close"]) if len(future_window) > 0 else entry_price
    return exit_price, "TIME"


# =========================
# STATE NORMALIZER
# =========================
def normalize_initial_state(state: dict) -> dict:
    return {
        **state,
        "backtest_mode": True,
        "skip_llm":      True,
        "debug_log":     [],
        "tts_output":    {},
        "ce_output":     {},
        # Provide a safe placeholder — SIV agent overwrites this on every run.
        # score_multiplier must be present so verdict_agent never KeyErrors.
        "siv_output": {
            "signal":           "COHERENT",
            "risk_penalty":     0.0,
            "issues":           [],
            "score_multiplier": 1.0,
        },
    }


# =========================
# MAIN BACKTEST
# =========================
def run_backtest(target_pair: str, target_months: list, target_year: int):

    total_start = time.perf_counter()

    pair_cfg = get_pair_config(target_pair)
    sl_mult  = float(pair_cfg.get("sl_mult", 2.0))
    rr_ratio = float(pair_cfg.get("rr_ratio", 1.75))

    project_root = Path(__file__).resolve().parents[1]
    ohlcv_path   = (
        project_root / "data" / "backtesting" / "forex_pairs" / f"{target_pair}.json"
    )

    with open(ohlcv_path, "r") as f:
        master_data = json.load(f)

    full_df = pd.DataFrame(master_data["data"])
    full_df.columns = [c.lower() for c in full_df.columns]
    full_df["timestamp"] = pd.to_datetime(full_df["timestamp"])
    full_df = full_df.sort_values("timestamp").reset_index(drop=True)

    # =========================
    # PRECOMPUTE INDICATORS
    # =========================
    precomputed = precompute_indicators(full_df).reset_index().set_index("timestamp")

    # ATR
    full_df["prev_close"] = full_df["close"].shift(1)
    full_df["tr"] = full_df.apply(
        lambda r: max(
            r["high"] - r["low"],
            abs(r["high"] - r["prev_close"]),
            abs(r["low"]  - r["prev_close"]),
        ),
        axis=1,
    )
    full_df["atr"] = full_df["tr"].rolling(ATR_LOOKBACK).mean()

    # =========================
    # PRE-BUILD FUTURE WINDOWS
    # =========================
    future_windows = [
        full_df.iloc[i + 1 : i + 1 + HORIZON_DAYS]
        for i in range(len(full_df))
    ]

    # =========================
    # FILTER TO TARGET PERIOD
    # =========================
    mask = (
        (full_df["timestamp"].dt.year  == target_year) &
        (full_df["timestamp"].dt.month.isin(target_months))
    )
    calib_df = full_df[mask]

    app     = build_graph()
    results = []
    raw_outputs = []   # saved for explanation pipeline

    # =========================
    # ACCURACY COUNTERS
    # =========================
    buy_correct  = buy_total  = 0
    sell_correct = sell_total = 0
    hold_correct = hold_total = 0
    skip_total   = 0                # quality-gate skips (action == "SKIP")

    tp_hits    = 0
    sl_hits    = 0
    time_exits = 0
    hold_skips = 0   # days with HOLD verdict — no simulation run

    print("\n--- STARTING BACKTEST ---")
    print(f"Pair={target_pair} | Year={target_year} | Months={target_months}")
    print(f"Threshold={VERDICT_THRESHOLD} | ATR_lookback={ATR_LOOKBACK}")
    print(f"Market threshold={MARKET_MOVE_THRESHOLD} | Horizon={HORIZON_DAYS}d")

    # =========================
    # MAIN LOOP
    # =========================
    for idx in calib_df.index:

        loop_start = time.perf_counter()

        row          = full_df.iloc[idx]
        current_date = row["timestamp"].strftime("%m/%d/%Y")
        entry_price  = float(row["close"])
        fw           = future_windows[idx]
        atr_val      = float(full_df.iloc[idx]["atr"]) if pd.notna(full_df.iloc[idx]["atr"]) else 0.0

        # =========================
        # MARKET LABEL
        # =========================
        market_label = "NEUTRAL"
        if not fw.empty:
            max_high  = fw["high"].max()
            min_low   = fw["low"].min()
            upside    = (max_high   - entry_price) / entry_price
            downside  = (entry_price - min_low)    / entry_price

            if upside > MARKET_MOVE_THRESHOLD and upside > downside:
                market_label = "BULLISH"
            elif downside > MARKET_MOVE_THRESHOLD:
                market_label = "BEARISH"

        # =========================
        # BUILD STATE
        # =========================
        state = cast(TradingState, normalize_initial_state({
            "target_date":            current_date,
            "currency_pair":          target_pair,
            "price":                  entry_price,
            "atr":                    atr_val,
            "calibration_threshold":  VERDICT_THRESHOLD,   # ← must match calibration
            "precomputed_indicators": precomputed,
        }))

        state = deepcopy(state)   # isolate state per iteration

        # =========================
        # GRAPH EXECUTION
        # =========================
        try:
            final_output = app.invoke(state)
        except Exception as e:
            print(f"[BACKTEST ERROR] {current_date}: {e}")
            continue

        ce  = final_output.get("ce_output",  {})
        tts = final_output.get("tts_output", {})
        siv = final_output.get("siv_output", {})

        action            = final_output.get("action", "NONE")
        verdict           = final_output.get("verdict", "HOLD").upper()
        weighted_score    = final_output.get("weighted_score", 0.0)
        verdict_reasoning = final_output.get("verdict_reasoning", "")

        exec_time = time.perf_counter() - loop_start

        # =========================
        # QUALITY GATE SKIP
        # verdict_agent returns action="SKIP" when signals are too weak.
        # Record the day for analysis but exclude from accuracy counters.
        # =========================
        if action == "SKIP":
            skip_total += 1
            print(f"SKIP: {current_date} | {verdict_reasoning}")

            results.append({
                "Date":             current_date,
                "Price":            entry_price,
                "ATR":              atr_val,
                "Final_Verdict":    "SKIP",
                "Market_Reality":   market_label,
                "Correct":          None,
                "Weighted_Score":   0.0,
                "Exit_Reason":      "QUALITY_GATE_SKIP",
                "CE_Article_Count": ce.get("article_count", 0),
                "CE_Sentiment":     ce.get("sentiment", "NEUTRAL"),
                "CE_Score":         ce.get("ce_score", 0.0),
                "CE_Confidence":    ce.get("ce_confidence", 0.0),
                "TTS_Score":        tts.get("total_score", 0.0),
                "TTS_Decision":     tts.get("decision", "HOLD"),
                "TTS_Regime":       tts.get("regime", "UNKNOWN"),
                "SIV_Signal":       siv.get("signal", "UNKNOWN"),
                "SIV_Multiplier":   siv.get("score_multiplier", 1.0),
                "Verdict_Reasoning": verdict_reasoning,
                "CE_Explanation":   "skipped",
                "TTS_Explanation":  "skipped",
                "SIV_Explanation":  "skipped",
                "Execution_Time_Seconds": round(exec_time, 4),
            })
            continue   # do NOT count in accuracy

        # =========================
        # SIMULATION
        # Only runs for BUY/SELL — HOLD has no trade to simulate
        # =========================
        sim_reason = "HOLD_SKIP"

        if verdict in ("BUY", "SELL") and not fw.empty:
            sl_dist = atr_val * sl_mult
            tp_dist = sl_dist * rr_ratio
            _, sim_reason = simulate_trade(verdict, entry_price, fw, sl_dist, tp_dist)

            if sim_reason == "TP":
                tp_hits += 1
            elif sim_reason == "SL":
                sl_hits += 1
            else:
                time_exits += 1
        else:
            hold_skips += 1   # HOLD verdict — simulation not run

        # =========================
        # ACCURACY
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

        else:   # HOLD
            hold_total += 1
            correct = market_label == "NEUTRAL"
            if correct:
                hold_correct += 1

        print(
            f"DEBUG: {current_date} | "
            f"SIV={siv.get('signal')} | "
            f"VERDICT={verdict} | "
            f"SCORE={weighted_score:.4f} | "
            f"MARKET={market_label} | "
            f"CORRECT={correct} | "
            f"EXIT={sim_reason} | "
            f"t={exec_time:.2f}s"
        )

        # =========================
        # RESULT ROW
        # =========================
        results.append({
            "Date":           current_date,
            "Price":          entry_price,
            "ATR":            atr_val,
            "Final_Verdict":  verdict,
            "Market_Reality": market_label,
            "Correct":        correct,
            "Weighted_Score": weighted_score,

            # CE
            "CE_Article_Count": ce.get("article_count", 0),
            "CE_Sentiment":     ce.get("sentiment", "NEUTRAL"),
            "CE_Score":         ce.get("ce_score", 0.0),
            "CE_Confidence":    ce.get("ce_confidence", 0.0),
            "CE_Explanation":   "pending_explanation",

            # TTS
            "TTS_Score":       tts.get("total_score", 0.0),
            "TTS_Decision":    tts.get("decision", "HOLD"),
            "TTS_Regime":      tts.get("regime", "UNKNOWN"),
            "TTS_Explanation": "pending_explanation",

            # SIV
            "SIV_Signal":      siv.get("signal", "UNKNOWN"),
            "SIV_Multiplier":  siv.get("score_multiplier", 1.0),
            "SIV_Explanation": "pending_explanation",

            # Trade
            "Verdict_Reasoning":       verdict_reasoning,
            "Exit_Reason":             sim_reason,
            "Execution_Time_Seconds":  round(exec_time, 4),
        })

        # =========================
        # RAW OUTPUT (for explanation pipeline)
        # =========================
        raw_outputs.append({
            "date":           current_date,
            "currency_pair":  target_pair,
            "atr":            atr_val,
            "ce_output":      ce,
            "tts_output":     tts,
            "siv_output":     siv,
            "verdict":        verdict,
            "weighted_score": weighted_score,
            "market_reality": market_label,
        })

    # =========================
    # ACCURACY HELPERS
    # =========================
    def acc(c: int, t: int) -> float:
        return round((c / t) * 100, 2) if t else 0.0

    buysell_correct = buy_correct  + sell_correct
    buysell_total   = buy_total    + sell_total
    total_correct   = buy_correct  + sell_correct + hold_correct
    total_trades    = buy_total    + sell_total   + hold_total

    # =========================
    # SAVE OUTPUTS
    # =========================
    os.makedirs(project_root / "reports", exist_ok=True)
    os.makedirs(project_root / "reports" / "raw", exist_ok=True)

    df = pd.DataFrame(results)

    summary_rows = pd.DataFrame([
        {"Date": "---",               "Final_Verdict": "ACCURACY SUMMARY"},
        {"Date": "BUY Accuracy",      "Final_Verdict": f"{acc(buy_correct,  buy_total)}%",         "Market_Reality": f"{buy_correct}/{buy_total}"},
        {"Date": "SELL Accuracy",     "Final_Verdict": f"{acc(sell_correct, sell_total)}%",        "Market_Reality": f"{sell_correct}/{sell_total}"},
        {"Date": "BUY+SELL Accuracy", "Final_Verdict": f"{acc(buysell_correct, buysell_total)}%",  "Market_Reality": f"{buysell_correct}/{buysell_total}"},
        {"Date": "HOLD Accuracy",     "Final_Verdict": f"{acc(hold_correct, hold_total)}%",        "Market_Reality": f"{hold_correct}/{hold_total}"},
        {"Date": "OVERALL Accuracy",  "Final_Verdict": f"{acc(total_correct, total_trades)}%",     "Market_Reality": f"{total_correct}/{total_trades}"},
        {"Date": "Skipped (gate)",    "Final_Verdict": str(skip_total),                            "Market_Reality": "quality gate"},
        {"Date": "---",               "Final_Verdict": "SL/TP SUMMARY"},
        {"Date": "TP Hits",           "Market_Reality": tp_hits},
        {"Date": "SL Hits",           "Market_Reality": sl_hits},
        {"Date": "TIME Exits",        "Market_Reality": time_exits},
        {"Date": "HOLD Skips",        "Market_Reality": hold_skips},
    ])

    df_with_summary = pd.concat([df, summary_rows], ignore_index=True)
    csv_out = project_root / "reports" / f"backtest_{target_pair}_{target_year}.csv"
    df_with_summary.to_csv(csv_out, index=False)

    raw_out = project_root / "reports" / "raw" / f"backtest_{target_pair}_{target_year}_raw.json"
    with open(raw_out, "w") as f:
        json.dump(raw_outputs, f, indent=2)

    # =========================
    # PRINT ACCURACY REPORT
    # =========================
    print("\n================ BACKTEST ACCURACY ================")
    print(f"BUY Accuracy      : {acc(buy_correct,  buy_total)}%  ({buy_correct}/{buy_total})")
    print(f"SELL Accuracy     : {acc(sell_correct, sell_total)}%  ({sell_correct}/{sell_total})")
    print(f"BUY+SELL Accuracy : {acc(buysell_correct, buysell_total)}%  ({buysell_correct}/{buysell_total})")
    print(f"HOLD Accuracy     : {acc(hold_correct, hold_total)}%  ({hold_correct}/{hold_total})")
    print(f"OVERALL ACCURACY  : {acc(total_correct, total_trades)}%")
    print(f"Quality Gate Skip : {skip_total} days")
    print(f"SL/TP             : TP={tp_hits} | SL={sl_hits} | TIME={time_exits} | HOLD={hold_skips}")
    print("====================================================\n")
    print(f"TOTAL BACKTEST TIME : {round(time.perf_counter() - total_start, 2)}s")
    print(f"CSV  : {csv_out}")
    print(f"JSON : {raw_out}")

    # =========================
    # AUTO-RUN EXPLANATION PIPELINE
    # =========================
    print("\n--- AUTO-STARTING EXPLANATION PIPELINE ---")
    


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("pair",     nargs="?", default="USDJPY")
    parser.add_argument("-m", "--months", default="8,9,10")
    parser.add_argument("-y", "--years",  default="2018")

    args = parser.parse_args()
    months = [int(x) for x in args.months.split(",")]

    raw_years = args.years.strip()
    if "-" in raw_years and "," not in raw_years:
        start, end = raw_years.split("-")
        years = list(range(int(start), int(end) + 1))
    else:
        years = [int(y) for y in raw_years.split(",")]

    # =========================
    # SEQUENTIAL YEAR QUEUE
    # Explanation pipeline finishes before next year starts
    # =========================
    total_years = len(years)
    for i, year in enumerate(years, 1):
        print(f"\n{'='*50}")
        print(f"QUEUE [{i}/{total_years}]: {args.pair.upper()} | YEAR={year} | MONTHS={months}")
        print(f"{'='*50}")

        try:
            run_backtest(args.pair.upper(), months, year)
        except Exception as e:
            print(f"[QUEUE ERROR] Year {year} failed: {e}")
            print("[QUEUE] Skipping to next year...\n")
            continue

        if i < total_years:
            print(f"\n[QUEUE] Year {year} complete. Starting {years[i]} in 5s...")
            time.sleep(5)

    print(f"\n[QUEUE] All {total_years} years complete.")