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

    attempt = 0
    while True:
        attempt += 1
        try:
            t0 = time.perf_counter()
            resp = requests.post(URL, headers=headers, json=data, timeout=90)
            log("LLM REQUEST", t0)

            if resp.status_code == 429:
                print(f"[TTS] Rate limited on attempt {attempt}. Waiting 15s...")
                time.sleep(15)
                continue

            if resp.status_code != 200:
                print(f"[TTS ERROR] HTTP {resp.status_code} on attempt {attempt} — retrying in 10s...")
                time.sleep(10)
                continue

            result = resp.json()
            message = result.get("choices", [{}])[0].get("message", {})
            content = message.get("content") or message.get("reasoning_content")

            if not content:
                print(f"[TTS ERROR] Empty content on attempt {attempt} — retrying in 10s...")
                time.sleep(10)
                continue

            return str(content).strip()

        except Exception as e:
            print(f"[TTS ERROR] Attempt {attempt}: {e} — retrying in 10s...")
            time.sleep(10)
            continue

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
    # FEATURE ENGINEERING (OLD LOGIC RESTORED, NO MACD)
    # =========================
    t2 = time.perf_counter()

    # EMA signal
    ema_vote = 1 if tech["trend"] == "BULLISH" else -1 if tech["trend"] == "BEARISH" else 0
    ema_strength = tech["trend_strength"]
    ema_score = ema_vote * min(ema_strength, 1.0) if ema_vote else 0.0

    # RSI signal (old style centered scoring)
    rsi_score = max(-1.0, min((tech["rsi"] - 50) / 50, 1.0))

    # Bollinger Bands signal
    if tech["bb_signal"] == "OVERSOLD":
        bb_score = min(tech["bb_strength"], 1.0)
    elif tech["bb_signal"] == "OVERBOUGHT":
        bb_score = -min(tech["bb_strength"], 1.0)
    else:
        bb_score = 0.0

    # Breakout logic (RESTORED)
    filtered_df = full_df[full_df["timestamp"] <= pd.to_datetime(target_date)]
    recent_high = filtered_df["high"].tail(20).max()
    recent_low = filtered_df["low"].tail(20).min()
    price = tech["price"]

    if price > recent_high:
        breakout_score = 1.0
    elif price < recent_low:
        breakout_score = -1.0
    else:
        breakout_score = 0.0

    log("FEATURE ENGINEERING", t2)

    # =========================
    # SCORING (OLD SYSTEM RESTORED)
    # =========================
    t3 = time.perf_counter()

    weights = {
        "ema": 0.3,
        "rsi": 0.3,
        "bb": 0.2,
        "breakout": 0.2
    }

    total_score = (
        weights["ema"] * ema_score +
        weights["rsi"] * rsi_score +
        weights["bb"] * bb_score +
        weights["breakout"] * breakout_score
    )

    # conflict penalty (OLD BEHAVIOR)
    if ema_score * rsi_score < 0:
        total_score *= 0.7

    total_score = max(-1.0, min(total_score, 1.0))

    log("SCORING", t3)


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