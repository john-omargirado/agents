import sys
from pathlib import Path
import json
import re

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from llm.ollama_client import verdict_llm as llm


def verdict_agent(state):
    state["debug_log"].append("VERDICT agent: starting")

    tts = state.get("tts_output", {})
    ce  = state.get("ce_output", {})
    siv = state.get("siv_output", {})

    retry_count = state.get("retry_count", 0)

    # =========================
    # HARD BLOCK
    # =========================
    if siv.get("signal") == "INCOHERENT":
        if retry_count >= 2:
            return {
                "verdict": "HOLD",
                "weighted_score": 0.0,
                "verdict_reasoning": "Retry limit reached. System fallback HOLD.",
                "risk_multiplier": 0.0,
                "action": "NONE"
            }
        return {
            "verdict": "HOLD",
            "weighted_score": 0.0,
            "verdict_reasoning": "SIV INCOHERENT — triggering retry",
            "risk_multiplier": 0.0,
            "action": "RETRY_TTS_CE",
            "retry_count": retry_count + 1
        }

    # =========================
    # WEIGHTED SCORE (deterministic)
    # =========================
    ce_map  = {"BULLISH": 1.0, "BEARISH": -1.0, "NEUTRAL": 0.0}
    tts_map = {"BUY": 1.0, "SELL": -1.0, "HOLD": 0.0}

    ce_signal   = ce_map.get(ce.get("sentiment", "NEUTRAL"), 0.0)
    tts_raw     = tts_map.get(tts.get("decision", "HOLD"), 0.0)
    tts_signal  = tts_raw * max(0.0, min(abs(tts.get("total_score", 0.0)), 1.0))

    weighted_score = (0.6 * ce_signal) + (0.4 * tts_signal)

    if   weighted_score >  0.15: pre_decision = "BUY"
    elif weighted_score < -0.15: pre_decision = "SELL"
    else:                        pre_decision = "HOLD"

    # =========================
    # RISK PRE-SCALING (deterministic)
    # =========================
    base_risk = 1.0
    if tts.get("tts_insufficient", False):        base_risk *= 0.5
    if ce.get("article_count", 0) < 5:            base_risk *= 0.7
    if ce.get("confidence") == "LOW":             base_risk *= 0.8
    if siv.get("signal") == "PARTIAL":            base_risk *= 0.85

    # =========================
    # EARLY EXIT — weak signal
    # =========================
    if abs(weighted_score) < 0.15:
        print("\n================ VERDICT RESULT ================")
        print("Decision: HOLD | Reason: Weak combined signal")
        print("================================================\n")
        return {
            "verdict": "HOLD",
            "weighted_score": round(weighted_score, 4),
            "verdict_reasoning": "Weak combined CE/TTS signal.",
            "risk_multiplier": 0.2,
            "action": "NONE"
        }

    # =========================
    # LLM CALL — risk + reasoning only
    # =========================
    prompt = f"""You are a forex trading risk assessor. All scores are -1.0 to +1.0.

INPUTS:
- tts_decision: {tts.get("decision")}
- tts_score: {tts.get("total_score", 0):.3f}
- tts_ema_trend: {tts.get("ema_trend")}
- tts_ema_200_reliable: {tts.get("ema_200_reliable")}
- tts_rsi: {tts.get("rsi_value", 50):.2f}
- ce_sentiment: {ce.get("sentiment")}
- ce_sentiment_score: {ce.get("sentiment_score", 0):.3f}
- ce_confidence: {ce.get("confidence")}
- ce_article_count: {ce.get("article_count", 0)}
- siv_signal: {siv.get("signal")}
- siv_conflict: {siv.get("conflict_type")}
- weighted_score: {weighted_score:.3f}
- pre_decision: {pre_decision}

TASK: Assess conviction and risk for this trade.

RESPOND WITH ONLY VALID JSON — no markdown, no extra text:
{{
  "verdict": "{pre_decision}",
  "risk_multiplier": <float 0.0-1.0>,
  "reasoning": "<one sentence max 20 words>"
}}

Rules:
- verdict MUST be exactly "{pre_decision}" — do not change it
- risk_multiplier: 0.8-1.0 strong alignment, 0.5-0.7 moderate, 0.0-0.4 weak or conflicted
"""

    response = llm.invoke(prompt)
    raw = getattr(response, "content", str(response)).strip()

    # =========================
    # PARSE — with fallback
    # =========================
    try:
        parsed    = json.loads(raw)
        verdict   = pre_decision                # never trust LLM to change direction
        risk      = float(parsed.get("risk_multiplier", 0.5))
        reasoning = str(parsed.get("reasoning", ""))
    except (json.JSONDecodeError, KeyError, ValueError):
        verdict   = pre_decision
        risk      = 0.5
        reasoning = (
            f"CE={ce.get('sentiment')} TTS={tts.get('decision')} "
            f"score={weighted_score:.2f}"
        )

    risk = max(0.0, min(risk * base_risk, 1.0))

    print("\n================ VERDICT RESULT ================")
    print(f"Decision: {verdict}")
    print(f"Weighted Score: {weighted_score:.2f} (CE 60% / TTS 40%)")
    print(f"Risk: {risk:.2f}")
    print(f"Reasoning: {reasoning}")
    print("================================================\n")

    return {
        "verdict": verdict,
        "weighted_score": round(weighted_score, 4),
        "verdict_reasoning": reasoning,
        "risk_multiplier": round(risk, 3),
        "action": "NONE"
    }