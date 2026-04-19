import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pathlib import Path
import json
import numpy as np
import pandas as pd
from llm.ollama_client import tts_llm as llm
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
        data_quality = tech.get("data_quality", {})
        tts_insufficient = (
            not data_quality.get("ema_200_reliable", True) or
            data_quality.get("data_stale", False)
        )

        state["debug_log"].append(
            f"TTS data quality: {data_quality['rows_available']} rows, "
            f"EMA200 confidence={data_quality['ema_200_confidence']:.2f}, "
            f"stale={data_quality['data_stale']}"
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

        reasoning = {
            "components": {
                "ema": {
                    "trend": tech["trend"],
                    "strength": round(ema_strength, 4),
                    "score": round(ema_score, 4)
                },
                "rsi": {
                    "value": round(rsi, 2),
                    "score": round(rsi_score, 4)
                },
                "bollinger": {
                    "signal": tech["bb_signal"],
                    "strength": round(tech["bb_strength"], 4),
                    "score": round(bb_score, 4)
                },
                "breakout": {
                    "recent_high": round(float(recent_high), 5),
                    "recent_low": round(float(recent_low), 5),
                    "price": round(float(price), 5),
                    "score": round(breakout_score, 4)
                }
            },
            "weights": weights,
            "total_score": round(total_score, 4),
            "decision": decision
            }

        prompt = f"""
            You are a trading explanation module.

            Explain the decision using ONLY the provided indicator scores.

            Decision: {decision}

            EMA Trend: {tech["trend"]} (strength={ema_strength:.4f}, score={ema_score:.4f})
            RSI Value: {rsi:.2f} (score={rsi_score:.4f})
            Bollinger Signal: {tech["bb_signal"]} (strength={tech["bb_strength"]:.4f}, score={bb_score:.4f})
            Breakout: price={price:.5f}, high={recent_high:.5f}, low={recent_low:.5f}, (score={breakout_score:.4f})

            Total Score: {total_score:.4f}

            Rules:
            - Explain how each indicator contributed to the final score
            - Focus on alignment or conflict between indicators
            - Do NOT suggest alternative trades
            - Do NOT question the decision
            - Keep it factual and mechanical

            Output:
            Maximum 4 sentences, concise, no filler.
        """

        response = llm.invoke(prompt)

        return {
            "tts_output": {
                "decision": decision,
                "total_score": round(float(total_score), 4),  # FIX: explicit float cast
                "reasoning": _sanitize(reasoning),            # FIX: strip np.float64 from dict
                "explanation": response.content,
                "indicators": _sanitize(tech)                 # FIX: strip np.float64 from indicators
            }
        }

    except Exception as e:
        state["debug_log"].append(f"TTS Error: {str(e)}")
        return {
            "tts_output": {
                "decision": "HOLD",
                "error": str(e)
            }
        }
