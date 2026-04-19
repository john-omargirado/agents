import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pathlib import Path
import json
import numpy as np
import pandas as pd
from state.trading_state import TradingState
from tools.tts_tools import calculate_technical_indicators


# FIX: numpy types (np.float64 etc.) are not JSON-serializable and print ugly in CSV
def _sanitize(obj):
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def tts_agent(state: TradingState):
    state["debug_log"].append("TTS agent: starting analysis")

    pair = state.get("currency_pair", "USDJPY").upper()
    target_date = state.get("target_date")

    project_root = Path(__file__).resolve().parents[1]
    file_path = project_root / "data" / "calibration" / "forex_pair" / f"{pair}.json"

    try:
        with open(file_path, "r") as f:
            raw_data = json.load(f)

        full_df = pd.DataFrame(raw_data["data"])
        full_df.columns = [c.lower() for c in full_df.columns]
        full_df["timestamp"] = pd.to_datetime(full_df["timestamp"])

        tech = calculate_technical_indicators(full_df, target_date)

        if not tech:
            raise ValueError(f"No technical data for {target_date}")

        # ✅ Extract and propagate data quality
        tts_insufficient = (
            not tech.get("ema_200_reliable", True) or
            tech.get("data_stale", False) or
            tech.get("ema_200_confidence", 1.0) < 0.5
        )

        state["debug_log"].append(
            f"TTS data quality: {tech['rows_available']} rows, "
            f"EMA200 confidence={tech['ema_200_confidence']:.2f}, "
            f"stale={tech['data_stale']}"
        )

        # =========================
        # ✅ EMA (normalized, no constant bias)
        # =========================
        ema_vote = 1 if tech["trend"] == "BULLISH" else -1 if tech["trend"] == "BEARISH" else 0
        ema_strength = tech["trend_strength"]

        if ema_vote != 0:
            ema_score = ema_vote * min(ema_strength, 1.0)
        else:
            ema_score = 0.0


        # =========================
        # ✅ RSI (continuous contribution)
        # =========================
        rsi = tech["rsi"]

        rsi_score = (rsi - 50) / 50  # positive = bullish, negative = bearish
        rsi_score = max(-1.0, min(rsi_score, 1.0))



        # =========================
        # ✅ Bollinger (still using your signal, but improved)
        # =========================
        if tech["bb_signal"] == "OVERSOLD":
            bb_score = min(tech["bb_strength"], 1.0)
        elif tech["bb_signal"] == "OVERBOUGHT":
            bb_score = -min(tech["bb_strength"], 1.0)
        else:
            bb_score = 0.0


        # =========================
        # ✅ REAL Breakout (20-period range)
        # =========================
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


        # =========================
        # ✅ WEIGHTED FUSION (balanced)
        # =========================
        weights = {
            "ema": 0.30,
            "rsi": 0.30,
            "bb": 0.20,
            "breakout": 0.20
        }

        total_score = (
            weights["ema"] * ema_score +
            weights["rsi"] * rsi_score +
            weights["bb"] * bb_score +
            weights["breakout"] * breakout_score
        )

        # =========================
        # ✅ Conflict dampening (no directional bias)
        # =========================
        if ema_score * rsi_score < 0:
            total_score *= 0.7


        # =========================
        # ✅ Clamp score (important for stability)
        # =========================
        total_score = max(-1.0, min(total_score, 1.0))


        # =========================
        # ✅ Decision (more responsive)
        # =========================
        if total_score > 0.15:
            decision = "BUY"
        elif total_score < -0.15:
            decision = "SELL"
        else:
            decision = "HOLD"


        return {
            "tts_output": {
                "decision": decision,
                "total_score": round(float(total_score), 4),
                "ema_trend": tech["trend"],
                "ema_score": round(float(ema_score), 4),
                "rsi_value": round(float(rsi), 2),
                "rsi_score": round(float(rsi_score), 4),
                "bb_signal": tech["bb_signal"],
                "bb_score": round(float(bb_score), 4),
                "breakout_score": round(float(breakout_score), 4),
                "price": round(float(price), 5),
                "ema_200_confidence": float(tech.get("ema_200_confidence", 1.0)),
                "ema_200_reliable": bool(tech.get("ema_200_reliable", True)),
                "data_stale": bool(tech.get("data_stale", False)),
                "rows_available": int(tech.get("rows_available", 0)),
                "tts_insufficient": tts_insufficient,
                "error": None
            }
        }

    except Exception as e:
        state["debug_log"].append(f"TTS Error: {str(e)}")
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
                "error": str(e)
            }
        }