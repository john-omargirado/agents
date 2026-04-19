import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from tools.verdict_tools import (
    normalize_signal,
    normalize_sentiment,
    extract_llm_fields,
    build_reason
)
from llm.ollama_client import verdict_llm as llm
import re



def verdict_agent(state):
    state["debug_log"].append("VERDICT agent: starting Agentic Synthesis")

    # =========================
    # OPTIONAL FAKE MODE
    # =========================
    fake_mode = state.get("verdict_fake_mode", False)

    if fake_mode:
        state["tts_output"] = {
            "decision": "BUY",
            "total_score": 0.78,
            "reasoning": {
                "ema_vote": 1,
                "rsi_vote": 1,
                "bb_vote": 1,
                "breakout_vote": 1
            },
            "explanation": (
                "All major indicators are aligned bullish. "
                "EMA confirms trend continuation. RSI is healthy. "
                "Bollinger Bands show expansion supporting upward move."
            )
        }

        state["ce_output"] = {
            "overall_sentiment": "BULLISH",
            "articles_analyzed": 12,
            "mean_score": 0.73,
            "reasoning": {
                "headline_bias": "positive",
                "keyword_strength": "strong bullish terms",
                "volume": "high coverage"
            }
        }

        state["siv_output"] = {
            "integrity_signal": "COHERENT",
            "conflict_type": "ALIGNED",
            "tts_insufficient": False,
            "explanation": (
                "TTS and CE signals are aligned in bullish direction. "
                "No data integrity issues detected. "
                "Sufficient indicators and news volume support coherence."
            )
        }

    # =========================
    # LOAD STATE
    # =========================
    siv_output = state.get("siv_output", {})
    siv_signal = siv_output.get("integrity_signal", "INCOHERENT")
    tts_insufficient = siv_output.get("tts_insufficient", False)

    tts_output = state.get("tts_output", {})
    ce_output = state.get("ce_output", {})

    retry_count = state.get("retry_count", 0)

    # =========================
    # HARD BLOCK (SIV FAILURE)
    # =========================
    if siv_signal == "INCOHERENT":
        if retry_count >= 2:
            return {
                "verdict": "HOLD",
                "verdict_reasoning": "Retry limit reached. System fallback HOLD.",
                "risk_multiplier": 0.0,
                "action": "NONE"
            }

        return {
            "verdict": "HOLD",
            "verdict_reasoning": "SIV INCOHERENT → triggering retry cycle",
            "risk_multiplier": 0.0,
            "action": "RETRY_TTS_CE",
            "retry_count": retry_count + 1
        }

    # =========================
    # NORMALIZATION
    # =========================
    tts_decision = normalize_signal(tts_output.get("decision"))
    ce_sentiment = normalize_sentiment(ce_output.get("overall_sentiment"))

    tts_score = float(tts_output.get("total_score", 0))
    article_count = int(ce_output.get("articles_analyzed", 0))

    # Weighted verdict influence: CE 60%, TTS 40%
    ce_signal_map = {
        "BULLISH": 1.0,
        "BEARISH": -1.0,
        "NEUTRAL": 0.0
    }
    tts_direction_map = {
        "BUY": 1.0,
        "SELL": -1.0,
        "HOLD": 0.0
    }

    ce_signal = ce_signal_map.get(ce_sentiment, 0.0)
    tts_direction = tts_direction_map.get(tts_decision, 0.0)
    tts_signal = tts_direction * max(0.0, min(abs(tts_score), 1.0))

    weighted_score = (0.6 * ce_signal) + (0.4 * tts_signal)

    if weighted_score > 0.15:
        weighted_decision = "BUY"
    elif weighted_score < -0.15:
        weighted_decision = "SELL"
    else:
        weighted_decision = "HOLD"

    # =========================
    # EARLY SAFETY GUARD
    # =========================
    if abs(weighted_score) < 0.15:
        print("\n================ VERDICT RESULT ================")
        print("Decision: HOLD")
        print("Risk: 0.15")
        print("Reasoning: Weak combined CE/TTS signal (60/40 weighting)")
        print("===============================================\n")

        return {
            "verdict": "HOLD",
            "verdict_reasoning": "Weak combined CE/TTS signal (60/40 weighting)",
            "risk_multiplier": 0.2,
            "action": "NONE"
        }

    # =========================
    # RISK PRE-SCALING
    # =========================
    base_risk = 1.0
    if tts_insufficient:
        base_risk *= 0.5
    if article_count < 5:
        base_risk *= 0.7

    # =========================
    # LLM PROMPT
    # =========================
    prompt = f"""
You are a trading decision engine.

TTS_DECISION: {tts_decision}
TTS_SCORE: {tts_score}
CE_SENTIMENT: {ce_sentiment}
ARTICLE_COUNT: {article_count}
SIV_STATUS: {siv_signal}
CE_WEIGHT: 0.6
TTS_WEIGHT: 0.4
WEIGHTED_SCORE: {weighted_score}
WEIGHTED_DECISION: {weighted_decision}

Return:
REASONING: short explanation
RISK_MULTIPLIER: 0.0 to 1.0
"""

    response = llm.invoke(prompt)
    res_text = getattr(response, "content", str(response))

    # =========================
    # PARSING
    # =========================
    _, reasoning = extract_llm_fields(res_text)

    risk_match = re.search(
        r"RISK_MULTIPLIER:\s*([0-9]*\.?[0-9]+)",
        res_text
    )

    risk = float(risk_match.group(1)) if risk_match else 0.5

    # clamp + scale
    risk = max(0.0, min(risk, 1.0)) * base_risk
    risk = max(0.0, min(risk, 1.0))

    # =========================
    # DEBUG OUTPUT
    # =========================
    print("\n================ VERDICT RESULT ================")
    print(f"Decision: {weighted_decision}")
    print(f"Weighted Score: {weighted_score:.2f} (CE 60% / TTS 40%)")
    print(f"Risk: {risk:.2f}")
    print(f"Reasoning: {reasoning}")
    print("===============================================\n")

    # =========================
    # FINAL OUTPUT
    # =========================
    return {
        "verdict": weighted_decision,
        "verdict_reasoning": build_reason(
            mode="AGENTIC_SYNTHESIS",
            tts_decision=tts_decision,
            ce_sentiment=ce_sentiment,
            article_count=article_count,
            conflict=("MISMATCH" in siv_output.get("conflict_type", "")),
            risk_multiplier=risk,
            extra=reasoning
        ),
        "risk_multiplier": risk,
        "action": "NONE",
        "fake_mode": fake_mode
    }


# =========================
# STANDALONE TEST
# =========================
if __name__ == "__main__":
    test_state = {
        "debug_log": [],
        "retry_count": 0
    }

    result = verdict_agent(test_state)

    print("\n================ FINAL RESULT ================")
    print(result)