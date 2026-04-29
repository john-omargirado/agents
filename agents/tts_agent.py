import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import requests
import time
import os
from datetime import datetime
from typing import Optional

from state.trading_state import TradingState
from tools.tts_tools import calculate_technical_indicators
from utils.credentials import get_do_model_key


URL = "https://inference.do-ai.run/v1/chat/completions"

_ohlcv_cache: dict = {}
_indicator_cache: dict = {}


# =========================
# SIMPLE LOGGER
# =========================
def log(stage: str, start: float):
    elapsed = (time.perf_counter() - start) * 1000
    print(f"[TTS TIMER] {stage}: {elapsed:.2f} ms")


def _load_ohlcv(file_path: Path):
    from tools.tts_tools import precompute_indicators

    key = str(file_path)

    if key in _ohlcv_cache:
        print("[TTS CACHE] OHLCV HIT")
        return _ohlcv_cache[key].copy(), _indicator_cache[key].copy()

    print("[TTS CACHE] OHLCV MISS - LOADING FILE")

    t0 = time.perf_counter()

    with open(file_path, "r") as f:
        raw_data = json.load(f)

    df = pd.DataFrame(raw_data["data"])
    df.columns = [c.lower() for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    _ohlcv_cache[key] = df
    _indicator_cache[key] = precompute_indicators(df)

    log("OHLCV LOAD + PRECOMPUTE", t0)

    return _ohlcv_cache[key].copy(), _indicator_cache[key].copy()


# =========================
# EXPLANATION (UNCHANGED)
# =========================
def call_tts_explanation(tts_data: dict, state: Optional[TradingState] = None) -> str:
    key = get_do_model_key()
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    prompt = f"""
You are a technical analysis explanation engine.

Explain briefly:
- Why signals produced this score
- Why decision is {tts_data.get('decision')}

INPUT:
{json.dumps(tts_data)}
"""

    data = {
        "model": "alibaba-qwen3-32b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 700,
        "temperature": 0.2
    }

    for attempt in range(3):
        try:
            t0 = time.perf_counter()
            resp = requests.post(URL, headers=headers, json=data, timeout=90)
            log("LLM REQUEST", t0)

            result = resp.json()
            message = result.get("choices", [{}])[0].get("message", {})
            content = message.get("content") or message.get("reasoning_content")

            return str(content).strip() if content else "explanation_unavailable"

        except Exception as e:
            print(f"[TTS ERROR] {e}")
            time.sleep(5 * (attempt + 1))

    return "explanation_unavailable"


# =========================
# MAIN AGENT WITH FULL TIMING
# =========================
def tts_agent(state: TradingState):

    total_start = time.perf_counter()

    state["debug_log"].append("TTS agent: starting analysis")

    pair = state.get("currency_pair", "USDJPY").upper()
    target_date = state.get("target_date")
    backtest_mode = state.get("backtest_mode", False)

    project_root = Path(__file__).resolve().parents[1]

    # ✅ FIX: define file_path BEFORE usage
    if backtest_mode:
        file_path = project_root / "data" / "backtesting" / "forex_pairs" / f"{pair}.json"
    else:
        file_path = project_root / "data" / "calibration" / "forex_pair" / f"{pair}.json"

    # =========================
    # LOAD DATA
    # =========================
    t0 = time.perf_counter()
    full_df, precomputed = _load_ohlcv(file_path)
    log("DATA LOAD", t0)

    # =========================
    # INDICATORS
    # =========================
    t1 = time.perf_counter()
    tech = calculate_technical_indicators(full_df, target_date, precomputed)
    log("INDICATORS", t1)

    if not tech:
        return {"tts_output": {"decision": "HOLD"}}

# =========================
    # FEATURE ENGINEERING
    # =========================
    t2 = time.perf_counter()

    ema_vote = 1 if tech["trend"] == "BULLISH" else -1 if tech["trend"] == "BEARISH" else 0
    ema_score = ema_vote * tech["trend_strength"]

    rsi_score = (tech["rsi"] - 50) / 50

    if tech["bb_signal"] == "OVERSOLD":
        bb_score = tech["bb_strength"]
    elif tech["bb_signal"] == "OVERBOUGHT":
        bb_score = -tech["bb_strength"]
    else:
        bb_score = 0.0

    price = tech["price"]
    macd_score = tech.get("macd_score", 0.0)

    log("FEATURE ENGINEERING", t2)

    # =========================
    # SCORING
    # =========================
    t3 = time.perf_counter()

    total_score = (
        0.35 * ema_score +
        0.25 * rsi_score +
        0.20 * bb_score +
        0.20 * macd_score
    )

    total_score = max(-1.0, min(total_score, 1.0))

    log("SCORING", t3)

    # =========================
    # DECISION
    # =========================
    if total_score > 0.15:
        decision = "BUY"
    elif total_score < -0.15:
        decision = "SELL"
    else:
        decision = "HOLD"

    # =========================
    # OUTPUT
    # =========================
    tts_result = {
        "decision": decision,
        "total_score": round(total_score, 4),
        "price": price,
        "ema_trend": tech["trend"],
        "rsi": tech["rsi"],
        "bb_signal": tech["bb_signal"],
        "ema_200_confidence": tech["ema_200_confidence"],
        "ema_200_reliable": tech["ema_200_reliable"],
        "macd_hist": tech.get("macd_hist", 0.0),
        "data_stale": tech["data_stale"],
        "explanation": "skipped_backtest" if backtest_mode else "pending"
    }
    # After building tts_result, add:
    if not backtest_mode:
        tts_result["explanation"] = call_tts_explanation(tts_result, state=state)
        print(f"\n[TTS EXPLANATION]\n{tts_result['explanation']}")
    else:
        tts_result["explanation"] = "skipped_backtest"

    log("TOTAL TTS PIPELINE", total_start)

    return {"tts_output": tts_result}