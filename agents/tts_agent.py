import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import requests
import os
from datetime import datetime
from typing import Optional, List

from state.trading_state import TradingState
from tools.tts_tools import calculate_technical_indicators
from utils.credentials import get_do_model_key


URL = "https://inference.do-ai.run/v1/chat/completions"


def call_gpt_mini(payload, state: Optional[TradingState] = None, candidate_models: Optional[List[str]] = None):
    """Call DO LLM with fallback models and detailed debug logging.

    - `state`: optional TradingState to append debug messages into `state['debug_log']`.
    - `candidate_models`: ordered list of model ids to try. If not provided,
      a sensible default list will be used.
    Returns the LLM content string on success or raises ValueError on final failure.
    """
    key = get_do_model_key()
    project_root = Path(__file__).resolve().parents[1]

    if candidate_models is None:
        candidate_models = [
            "alibaba-qwen3-32b",
        ]

    last_err = None
    save_debug = os.getenv("TTS_DEBUG_SAVE", "0") in ("1", "true", "True")

    for model in candidate_models:
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"

        data = {
            "model": model,
            "messages": [{"role": "user", "content": json.dumps(payload)}],
            "max_tokens": 200,
        }

        try:
            resp = requests.post(URL, headers=headers, json=data, timeout=30)
        except Exception as e:
            msg = f"LLM request failed for model {model}: {e}"
            last_err = msg
            if state is not None:
                state["debug_log"].append(msg)
            continue

        status = resp.status_code
        body_text = resp.text

        try:
            result = resp.json()
        except Exception as e:
            msg = f"Invalid JSON from model {model} (status {status}): {body_text}"
            last_err = msg
            if state is not None:
                state["debug_log"].append(msg)
            if save_debug:
                dbg_path = project_root / "logs"
                dbg_path.mkdir(exist_ok=True)
                (dbg_path / f"tts_debug_{model}_{datetime.utcnow().isoformat()}.txt").write_text(body_text)
            continue

        # Successful content
        if isinstance(result, dict) and "choices" in result:
            try:
                content = result["choices"][0]["message"]["content"]
                if state is not None:
                    state["debug_log"].append(f"LLM success model={model} status={status}")
                if save_debug:
                    dbg_path = project_root / "logs"
                    dbg_path.mkdir(exist_ok=True)
                    (dbg_path / f"tts_debug_success_{model}_{datetime.utcnow().isoformat()}.json").write_text(json.dumps({"model": model, "status": status, "request": payload, "response": result}, default=str))
                return content
            except Exception:
                msg = f"Malformed choices structure from model {model}: {result}"
                last_err = msg
                if state is not None:
                    state["debug_log"].append(msg)
                continue

        # If there's an error block from the API, record and try next model when applicable
        if isinstance(result, dict) and "error" in result:
            err = result["error"]
            msg = f"LLM error for model {model}: {err}"
            last_err = msg
            if state is not None:
                state["debug_log"].append(msg)
            # if it's clearly a model-availability or unauthorized error, continue to fallback
            continue

        # Unknown format — save and continue
        msg = f"Unknown response format from model {model}: {result}"
        last_err = msg
        if state is not None:
            state["debug_log"].append(msg)

    # all models exhausted
    raise ValueError(f"LLM error: {last_err}")


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
            raise ValueError("No technical data")

        tts_insufficient = (
            not tech.get("ema_200_reliable", True)
            or tech.get("data_stale", False)
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
        # LLM PAYLOAD (GPT-MINI)
        # =========================

        payload = {
            "total_score": round(total_score, 4),
            "ema_trend": tech["trend"],
            "rsi": round(rsi, 2),
            "bb_signal": tech["bb_signal"],
            "breakout_score": round(breakout_score, 4),
            "rule": {
                "BUY": "> 0.15",
                "SELL": "< -0.15",
                "HOLD": "otherwise"
            }
        }

        raw = call_gpt_mini(payload, state=state)

        try:
            parsed = json.loads(raw)
            decision = parsed.get("decision", "HOLD")

        except:
            decision = "HOLD"

        # =========================
        # HARD RULE ENFORCEMENT
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
                "error": None
            }
        }

    except Exception as e:
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