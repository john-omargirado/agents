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
from tools.tts_tools import calculate_technical_indicators, precompute_indicators
from utils.credentials import get_do_model_key

URL = "https://inference.do-ai.run/v1/chat/completions"

_ohlcv_cache: dict = {}
_indicator_cache: dict = {}


# =========================
# DATE NORMALIZER
# =========================
def normalize_date(date_str: Optional[str]) -> Optional[str]:
    if not date_str:
        return None

    date_str = str(date_str).strip()

    for fmt in ("%m-%d-%Y", "%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue

    print(f"[TTS WARNING] Invalid date format: {date_str}")
    return None


# =========================
# LOGGER
# =========================
def log(stage: str, start: float):
    elapsed = (time.perf_counter() - start) * 1000
    print(f"[TTS TIMER] {stage}: {elapsed:.2f} ms")


# =========================
# LOAD OHLCV
# =========================
def _load_ohlcv(file_path: Path):
    key = str(file_path)

    if key in _ohlcv_cache:
        return _ohlcv_cache[key].copy(), _indicator_cache[key].copy()

    with open(file_path, "r") as f:
        raw_data = json.load(f)

    df = pd.DataFrame(raw_data["data"])
    df.columns = [c.lower() for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    _ohlcv_cache[key] = df
    _indicator_cache[key] = precompute_indicators(df)

    return _ohlcv_cache[key].copy(), _indicator_cache[key].copy()


# =========================
# EXPLANATION
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

    while True:
        try:
            resp = requests.post(URL, headers=headers, json=data, timeout=90)

            if resp.status_code == 429:
                time.sleep(15)
                continue

            if resp.status_code != 200:
                time.sleep(10)
                continue

            result = resp.json()
            message = result.get("choices", [{}])[0].get("message", {})
            content = message.get("content") or message.get("reasoning_content")

            if not content:
                time.sleep(10)
                continue

            return str(content).strip()

        except Exception:
            time.sleep(10)
            continue

    return "explanation_unavailable"


# =========================
# MAIN AGENT
# =========================
def tts_agent(state: TradingState):

    pair = state.get("currency_pair", "USDJPY").upper()
    target_date = state.get("target_date")
    backtest_mode = state.get("backtest_mode", False)
    live_mode = state.get("live_mode", False)
    skip_llm = state.get("skip_llm", False)

    BASE_DIR = Path(__file__).resolve().parent.parent
    pair_safe = pair.replace("/", "").upper().strip()

    file_path = BASE_DIR / "data" / "backtesting" / "forex_pairs" / f"{pair_safe}.json"

    full_df, precomputed = _load_ohlcv(file_path)

    # normalize date
    normalized_date = normalize_date(target_date)

    if not normalized_date:
        target_date = str(full_df.iloc[-1]["timestamp"].date())
    else:
        target_date = normalized_date

    # =========================
    # INDICATORS
    # =========================
    tech = calculate_technical_indicators(
        full_df,
        target_date,
        precomputed,
        live_mode=False
    )

    if not tech:
        return {"tts_output": {"decision": "HOLD"}}

    # =========================
    # ATR FIX (IMPORTANT)
    # =========================
    atr = tech.get("atr", None)

    if atr is None or atr <= 0:
        # fallback safety (prevents verdict crash)
        high = tech.get("breakout_high", 0)
        low = tech.get("breakout_low", 0)
        atr = abs(high - low) if high and low else 0.0

    # =========================
    # REGIME
    # =========================
    adx_proxy = tech.get("adx_proxy", 0.0)

    if adx_proxy > 0.4:
        regime = "TRENDING"
    elif adx_proxy < 0.15:
        regime = "RANGING"
    else:
        regime = "TRANSITIONAL"

    # =========================
    # SIGNALS
    # =========================
    rsi = tech["rsi"]
    macd_score = tech.get("macd_direction_score", 0.0)
    is_macd_cross = abs(macd_score) >= 0.6

    if rsi > 60:
        rsi_score = -1.0 * min((rsi - 60) / 40, 1.0)
    elif rsi < 40:
        rsi_score = 1.0 * min((40 - rsi) / 40, 1.0)
    else:
        rsi_score = 0.0

    bb_score = 0.0
    if tech["bb_signal"] == "OVERSOLD":
        bb_score = min(tech["bb_strength"], 1.0)
    elif tech["bb_signal"] == "OVERBOUGHT":
        bb_score = -min(tech["bb_strength"], 1.0)

    breakout_signal = tech.get("breakout_signal", "NONE")
    breakout_strength = tech.get("breakout_strength", 0.0)

    breakout_raw_score = (
        breakout_strength if breakout_signal == "BREAKOUT_UP"
        else -breakout_strength if breakout_signal == "BREAKOUT_DOWN"
        else 0.0
    )

    breakout_weight = {
        "TRENDING": 0.25,
        "TRANSITIONAL": 0.15,
        "RANGING": 0.05
    }[regime]

    remaining = 1.0 - breakout_weight

    if is_macd_cross:
        total_score = (
            macd_score * (0.50 * remaining) +
            rsi_score * (0.30 * remaining) +
            bb_score * (0.20 * remaining) +
            breakout_raw_score * breakout_weight
        )
    else:
        total_score = (
            rsi_score * (0.45 * remaining) +
            bb_score * (0.25 * remaining) +
            macd_score * (0.30 * remaining) +
            breakout_raw_score * breakout_weight
        )

    total_score = max(-1.0, min(total_score, 1.0))

    decision = (
        "BUY" if total_score > 0.15
        else "SELL" if total_score < -0.15
        else "HOLD"
    )

    # =========================
    # OUTPUT
    # =========================
    tts_result = {
        "decision": decision,
        "atr": float(atr),
        "tts_score": total_score,
        "total_score": round(total_score, 4),
        "price": tech["price"],
        "ema_trend": tech["trend"],
        "rsi": rsi,
        "bb_signal": tech["bb_signal"],
        "macd_direction_score": macd_score,
        "is_macd_cross": is_macd_cross,
        "regime": regime,
        "ema_200_confidence": tech["ema_200_confidence"],
        "ema_200_reliable": tech["ema_200_reliable"],
        "data_stale": tech["data_stale"],
        "breakout_signal": breakout_signal,
        "breakout_strength": round(breakout_strength, 4),
        "breakout_high": tech.get("breakout_high"),
        "breakout_low": tech.get("breakout_low"),
        "explanation": "pending"
    }

    if not skip_llm:
        tts_result["explanation"] = call_tts_explanation(tts_result, state=state)
    else:
        tts_result["explanation"] = "skipped"

    return {
        "tts_output": tts_result,
        "atr": float(atr),        # ← populate TradingState.atr from real ATR
        "price": tech["price"],   # ← populate TradingState.price too
    }