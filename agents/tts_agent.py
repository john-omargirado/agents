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

    pair = state.get("currency_pair", "USDJPY").upper()
    target_date = state.get("target_date")
    backtest_mode = state.get("backtest_mode", False)
    skip_llm = state.get("skip_llm", False)

    project_root = Path(__file__).resolve().parents[1]

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

    atr_val = state.get("atr", 0.0)

    # =========================
    # REGIME DETECTION
    # =========================
    adx_proxy = tech.get("adx_proxy", 0.0)

    if adx_proxy > 0.4:
        regime = "TRENDING"
        trend_weight, mr_weight = 0.75, 0.25
    elif adx_proxy < 0.15:
        regime = "RANGING"
        trend_weight, mr_weight = 0.30, 0.70
    else:
        regime = "TRANSITIONAL"
        trend_weight, mr_weight = 0.55, 0.45

    # =========================
    # SIGNALS
    # =========================

    rsi = tech["rsi"]
    macd_score = tech.get("macd_direction_score", 0.0)
    is_macd_cross = abs(macd_score) >= 0.6

    if rsi > 60:
        rsi_score = -1.0 * min((rsi - 60) / 40, 1.0)   # overbought → SELL
    elif rsi < 40:
        rsi_score = 1.0 * min((40 - rsi) / 40, 1.0)    # oversold  → BUY
    else:
        rsi_score = 0.0

    if tech["bb_signal"] == "OVERSOLD":
        bb_score = min(tech["bb_strength"], 1.0)              # below lower band → BUY
    elif tech["bb_signal"] == "OVERBOUGHT":
        bb_score = -1.0 * min(tech["bb_strength"], 1.0)      # above upper band → SELL
    else:
        bb_score = 0.0

    ema_context = tech["trend"]

    # ── Breakout score ───────────────────────────────────────────────────────
    # BREAKOUT_UP   → positive score (momentum BUY)
    # BREAKOUT_DOWN → negative score (momentum SELL)
    # NONE          → 0.0 (no breakout, ignored)
    #
    # Regime weighting:
    #   TRENDING     → breakout is a strong confirmation signal  (weight 0.25)
    #   TRANSITIONAL → moderate confirmation                     (weight 0.15)
    #   RANGING      → breakout is less reliable / fakeout-prone (weight 0.05)
    breakout_signal   = tech.get("breakout_signal", "NONE")
    breakout_strength = tech.get("breakout_strength", 0.0)

    if breakout_signal == "BREAKOUT_UP":
        breakout_raw_score = breakout_strength          # +ve → BUY momentum
    elif breakout_signal == "BREAKOUT_DOWN":
        breakout_raw_score = -breakout_strength         # -ve → SELL momentum
    else:
        breakout_raw_score = 0.0

    breakout_weight = {"TRENDING": 0.25, "TRANSITIONAL": 0.15, "RANGING": 0.05}[regime]

    print(
        f"[TTS DEBUG] breakout_signal={breakout_signal} | "
        f"raw_score={breakout_raw_score:.4f} | weight={breakout_weight} | regime={regime}"
    )
    # ────────────────────────────────────────────────────────────────────────

    # =========================
    # SCORING
    # =========================
    # Weights are redistributed to make room for breakout_weight.
    # The leftover (1.0 - breakout_weight) is split among existing signals
    # using the same proportions as before so existing behaviour is preserved
    # when there is no breakout (breakout_raw_score == 0).

    remaining = 1.0 - breakout_weight

    if is_macd_cross:
        total_score = (
            macd_score          * (0.50 * remaining) +
            rsi_score           * (0.30 * remaining) +
            bb_score            * (0.20 * remaining) +
            breakout_raw_score  * breakout_weight
        )
    else:
        total_score = (
            rsi_score           * (0.45 * remaining) +
            bb_score            * (0.25 * remaining) +
            macd_score          * (0.30 * remaining) +
            breakout_raw_score  * breakout_weight
        )

    total_score = max(-1.0, min(total_score, 1.0))

    # =========================
    # DECISION
    # =========================
    if total_score > 0.15:
        decision = "BUY"
    elif total_score < -0.15:
        decision = "SELL"
    else:
        decision = "HOLD"

    tts_result = {
        "decision":             decision,
        "tts_score":            total_score,
        "total_score":          round(total_score, 4),
        "price":                tech["price"],
        "ema_trend":            tech["trend"],
        "rsi":                  rsi,
        "bb_signal":            tech["bb_signal"],
        "macd_direction_score": macd_score,
        "is_macd_cross":        is_macd_cross,
        "regime":               regime,
        "ema_200_confidence":   tech["ema_200_confidence"],
        "ema_200_reliable":     tech["ema_200_reliable"],
        "data_stale":           tech["data_stale"],
        # ── breakout keys ──────────────────────────────
        "breakout_signal":      breakout_signal,
        "breakout_strength":    round(breakout_strength, 4),
        "breakout_high":        tech.get("breakout_high"),
        "breakout_low":         tech.get("breakout_low"),
        # ───────────────────────────────────────────────
        "explanation":          "pending"
    }

    if not backtest_mode and not skip_llm:
        tts_result["explanation"] = call_tts_explanation(tts_result, state=state)
        print(f"\n[TTS EXPLANATION]\n{tts_result['explanation']}")
    else:
        tts_result["explanation"] = "skipped"

    log("TOTAL TTS PIPELINE", total_start)

    return {"tts_output": tts_result}