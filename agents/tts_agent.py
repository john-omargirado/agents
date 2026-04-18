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

        ema_vote = 1 if tech["trend"] == "BULLISH" else -1 if tech["trend"] == "BEARISH" else 0
        ema_strength = tech["trend_strength"]

        rsi = tech["rsi"]
        if rsi < 30:
            rsi_vote = 1
        elif rsi > 70:
            rsi_vote = -1
        else:
            rsi_vote = 0
        rsi_strength = tech["rsi_strength"]

        if tech["bb_signal"] == "OVERSOLD":
            bb_vote = 1
        elif tech["bb_signal"] == "OVERBOUGHT":
            bb_vote = -1
        else:
            bb_vote = 0
        bb_strength = tech["bb_strength"]

        breakout_vote = 1 if ema_vote == 1 and rsi > 50 else -1 if ema_vote == -1 and rsi < 50 else 0

        ema_score = ema_vote * 0.35 * (1 + ema_strength)
        rsi_score = rsi_vote * 0.25 * (1 + rsi_strength)
        bb_score = bb_vote * 0.25 * (1 + bb_strength)
        breakout_score = breakout_vote * 0.15

        total_score = ema_score + rsi_score + bb_score + breakout_score

        if total_score >= 0.35:
            decision = "BUY"
        elif total_score <= -0.35:
            decision = "SELL"
        else:
            decision = "HOLD"

        reasoning = {
            "ema_vote": ema_vote,
            "ema_strength": ema_strength,
            "rsi_vote": rsi_vote,
            "rsi_strength": rsi_strength,
            "bb_vote": bb_vote,
            "bb_strength": bb_strength,
            "breakout_vote": breakout_vote,
            "scores": {
                "ema_score": round(ema_score, 4),
                "rsi_score": round(rsi_score, 4),
                "bb_score": round(bb_score, 4),
                "breakout_score": round(breakout_score, 4),
                "total_score": round(total_score, 4)
            }
        }

        prompt = f"""
You are a trading explanation module.

Explain ONLY using the provided strategy votes and scores.

Decision: {decision}

EMA: {ema_vote} (score={ema_score:.4f})
RSI: {rsi_vote} (score={rsi_score:.4f})
BB: {bb_vote} (score={bb_score:.4f})
Breakout: {breakout_vote} (score={breakout_score:.4f})

Total Score: {total_score:.4f}

Rules:
- No alternative trades
- No corrections
- Only explain alignment of strategies

Output:
4 sentences max, concise, factual, no filler.
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