import json
import requests
from utils.credentials import get_do_model_key

URL = "https://inference.do-ai.run/v1/chat/completions"


def call_qwen(payload):
    key = get_do_model_key()
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    data = {
        "model": "alibaba-qwen3-32b",
        "messages": [
            {"role": "user", "content": json.dumps(payload)}
        ],
        "max_tokens": 250
    }

    response = requests.post(URL, headers=headers, json=data)
    return response.json()["choices"][0]["message"]["content"]


def verdict_agent(state):
    state["debug_log"].append("VERDICT agent: starting")

    tts = state.get("tts_output", {})
    ce  = state.get("ce_output", {})
    siv = state.get("siv_output", {})

    # =========================
    # HARD BLOCK
    # =========================
    if siv.get("signal") == "INCOHERENT":
        return {
            "verdict": "HOLD",
            "weighted_score": 0.0,
            "verdict_reasoning": "SIV INCOHERENT",
            "risk_multiplier": 0.0,
            "action": "NONE"
        }

    # =========================
    # SCORE COMPUTATION
    # =========================
    ce_map = {
        "BULLISH": 1.0,
        "BEARISH": -1.0,
        "NEUTRAL": 0.0
    }

    ce_signal = ce_map.get(ce.get("sentiment", "NEUTRAL"), 0.0)
    tts_signal = float(tts.get("total_score", 0.0))

    weighted_score = (0.6 * ce_signal) + (0.4 * tts_signal)

    # =========================
    # DETERMINISTIC DECISION (FINAL AUTHORITY)
    # =========================
    if weighted_score > 0.15:
        final_verdict = "BUY"
    elif weighted_score < -0.15:
        final_verdict = "SELL"
    else:
        final_verdict = "HOLD"

    # =========================
    # LLM CONTEXT PAYLOAD (FOR REASONING ONLY)
    # =========================
    payload = {
        "task": "Explain trading decision based on multi-signal system",

        "decision_rule": {
            "BUY": "> 0.15",
            "SELL": "< -0.15",
            "HOLD": "otherwise"
        },

        "weighted_score": round(weighted_score, 4),

        "tts": {
            "decision": tts.get("decision"),
            "score": tts.get("total_score"),
            "ema_trend": tts.get("ema_trend"),
            "rsi": tts.get("rsi_value"),
            "bb_signal": tts.get("bb_signal"),
            "breakout_score": tts.get("breakout_score"),
            "insufficient": tts.get("tts_insufficient")
        },

        "ce": {
            "sentiment": ce.get("sentiment"),
            "confidence": ce.get("confidence"),
            "article_count": ce.get("article_count"),
            "sentiment_score": ce.get("sentiment_score")
        },

        "siv": {
            "signal": siv.get("signal"),
            "conflict_type": siv.get("conflict_type"),
            "data_quality_ok": siv.get("data_quality_ok"),
            "issues": siv.get("issues")
        },

        "instruction": "Explain WHY this trade resulted in the final decision. Do NOT change the decision."
    }

    raw = call_qwen(payload)

    try:
        parsed = json.loads(raw)
        reasoning = parsed.get("reasoning", parsed.get("explanation", ""))

    except Exception:
        reasoning = "LLM failed to parse reasoning"

    # =========================
    # RISK (still optional LLM-free logic)
    # =========================
    risk = 0.5
    if abs(weighted_score) > 0.5:
        risk = 0.8
    elif abs(weighted_score) > 0.2:
        risk = 0.6
    else:
        risk = 0.4

    return {
        "verdict": final_verdict,
        "weighted_score": round(weighted_score, 4),
        "verdict_reasoning": reasoning,
        "risk_multiplier": round(risk, 3),
        "action": "NONE"
    }