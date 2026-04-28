import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import requests
import time
import os
from datetime import datetime
from typing import Optional, List

from state.trading_state import TradingState
from tools.tts_tools import calculate_technical_indicators
from utils.credentials import get_do_model_key


URL = "https://inference.do-ai.run/v1/chat/completions"

_ohlcv_cache: dict = {}
_indicator_cache: dict = {}

def _load_ohlcv(file_path: Path):
    from tools.tts_tools import precompute_indicators

    key = str(file_path)

    if key not in _ohlcv_cache:
        with open(file_path, "r") as f:
            raw_data = json.load(f)

        df = pd.DataFrame(raw_data["data"])
        df.columns = [c.lower() for c in df.columns]
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        df = df.sort_values("timestamp").reset_index(drop=True)

        _ohlcv_cache[key] = df
        _indicator_cache[key] = precompute_indicators(df)

    return _ohlcv_cache[key].copy(), _indicator_cache[key].copy()


def call_tts_explanation(tts_data: dict, state: Optional[TradingState] = None) -> str:
    key = get_do_model_key()
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    prompt = f"""
You are a technical analysis explanation engine for the Traditional Trading Strategies (TTS) Agent.

Explain briefly:
- Why the EMA/RSI/BB/breakout signals produced this score
- What the dominant signal was
- Any conflicting indicators
- Why the final decision is {tts_data.get('decision')}
- Note EMA 200 confidence level and whether it affects reliability

Be concise and factual. No structured format.

INPUT:
{json.dumps(tts_data)}
"""
    data = {
        "model": "alibaba-qwen3-32b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
        "temperature": 0.2
    }

    max_retries = 3
    backoff = 5

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(URL, headers=headers, json=data, timeout=90)
            print(f"[TTS EXPLANATION HTTP] status={resp.status_code}")
            result = resp.json()
            message = result.get("choices", [{}])[0].get("message", {})
            content = message.get("content") or message.get("reasoning_content")
            return str(content).strip() if content else "explanation_unavailable"

        except Exception as e:
            print(f"[TTS EXPLANATION ERROR] Attempt {attempt}/{max_retries}: {e}")
            if attempt < max_retries:
                print(f"[TTS EXPLANATION] Retrying in {backoff}s...")
                time.sleep(backoff)
                backoff *= 2  # 5 -> 10 -> 20
            if state is not None:
                state["debug_log"].append(f"TTS explanation attempt {attempt} failed: {e}")

    return "explanation_unavailable"


def tts_agent(state: TradingState):
    state["debug_log"].append("TTS agent: starting analysis")

    pair = state.get("currency_pair", "USDJPY").upper()
    target_date = state.get("target_date")

    project_root = Path(__file__).resolve().parents[1]
    if state.get("backtest_mode"):
        file_path = project_root / "data" / "backtesting" / "forex_pairs" / f"{pair}.json"
    else:
        file_path = project_root / "data" / "calibration" / "forex_pair" / f"{pair}.json"

    try:
        full_df, precomputed = _load_ohlcv(file_path)
        tech = calculate_technical_indicators(full_df, target_date, precomputed=precomputed)

        if not tech:
            raise ValueError("No technical data")

        # =========================
        # FIX: only flag insufficient if data is stale
        # ema_200_reliable being False just means low confidence,
        # not that TTS cannot contribute — explanation covers this
        # =========================
        tts_insufficient = tech.get("data_stale", False)

        state["debug_log"].append(
            f"TTS: rows={tech.get('rows_available')} | "
            f"ema_200_confidence={tech.get('ema_200_confidence')} | "
            f"ema_200_reliable={tech.get('ema_200_reliable')} | "
            f"data_stale={tech.get('data_stale')} | "
            f"tts_insufficient={tts_insufficient}"
        )

        # =========================
        # STRATEGY ENGINE (DETERMINISTIC)
        # =========================

        ema_vote = 1 if tech["trend"] == "BULLISH" else -1 if tech["trend"] == "BEARISH" else 0
        ema_strength = tech["trend_strength"]
        ema_score = ema_vote * min(ema_strength, 1.0) if ema_vote else 0.0

        rsi = tech["rsi"]
        rsi_score = max(-1.0, min((rsi - 50) / 50, 1.0))

        if tech["bb_signal"] == "OVERSOLD":
            bb_score = min(tech["bb_strength"], 1.0)
        elif tech["bb_signal"] == "OVERBOUGHT":
            bb_score = -min(tech["bb_strength"], 1.0)
        else:
            bb_score = 0.0

        filtered_df = full_df[full_df["timestamp"] <= pd.to_datetime(target_date)]
        recent_high = filtered_df["high"].tail(20).max()
        recent_low = filtered_df["low"].tail(20).min()
        price = tech["price"]

        breakout_score = (
            1.0 if price > recent_high else -1.0 if price < recent_low else 0.0
        )

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

        if ema_score * rsi_score < 0:
            total_score *= 0.7

        total_score = max(-1.0, min(total_score, 1.0))

        # =========================
        # HARD RULE ENFORCEMENT
        # =========================

        if total_score > 0.15:
            decision = "BUY"
        elif total_score < -0.15:
            decision = "SELL"
        else:
            decision = "HOLD"

        # =========================
        # BUILD RESULT + EXPLANATION
        # =========================

        tts_result = {
            "decision": decision,
            "total_score": float(round(total_score, 4)),
            "ema_trend": tech["trend"],
            "ema_score": float(round(ema_score, 4)),
            "rsi_value": float(round(rsi, 2)),
            "rsi_score": float(round(rsi_score, 4)),
            "bb_signal": tech["bb_signal"],
            "bb_score": float(round(bb_score, 4)),
            "breakout_score": float(round(breakout_score, 4)),
            "price": float(round(price, 5)),
            "ema_200_confidence": float(tech.get("ema_200_confidence", 1.0)),
            "ema_200_reliable": bool(tech.get("ema_200_reliable", True)),
            "data_stale": bool(tech.get("data_stale", False)),
            "rows_available": int(tech.get("rows_available", 0)),
            "tts_insufficient": bool(tts_insufficient),
            "error": None,
            "explanation": "pending"
        }

        tts_result["explanation"] = call_tts_explanation(tts_result, state=state)
        print(f"\n[TTS EXPLANATION]\n{tts_result['explanation']}")

        return {"tts_output": tts_result}

    except Exception as e:
        print(f"[TTS AGENT ERROR] {e}")
        return {
            "tts_output": {
                "decision": "HOLD",
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
                "error": str(e),
                "explanation": "error_no_data"
            }
        }