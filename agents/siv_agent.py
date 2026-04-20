import json
import requests
from typing import Any, Dict, Tuple
from utils.credentials import get_do_model_key
from utils.formatters import prepare_siv_payload

URL = "https://inference.do-ai.run/v1/chat/completions"


# =========================
# LLM ONLY FOR EXPLANATION
# =========================
def call_llm(payload: Dict[str, Any]) -> str:
    key = get_do_model_key()

    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    prompt = f"""
You are a financial system explanation engine.

Do NOT output structured format.

Explain briefly:
- CE vs TTS alignment
- price mismatch if any
- data quality issues
- why signals conflict or align

Be concise and factual.

INPUT:
{json.dumps(payload)}
"""

    data = {
        "model": "alibaba-qwen3-32b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 300,
        "temperature": 0.2
    }

    try:
        response = requests.post(URL, headers=headers, json=data, timeout=30)

        if response.status_code != 200:
            return "LLM_ERROR"

        result = response.json()

        # SAFE extraction
        choice = result.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content")

        if not content:
            return "LLM_EMPTY"

        return str(content).strip()

    except Exception:
        return "LLM_ERROR"


# =========================
# PURE DETERMINISTIC CORE
# =========================
def compute_siv(payload: Dict[str, Any]) -> Tuple[str, list]:
    ce = payload.get("ce_signal")
    tts = payload.get("tts_signal")

    actual_price = payload.get("actual_price")
    tts_price = payload.get("tts_price")

    issues = []

    # PRICE INTEGRITY (HIGHEST PRIORITY)
    if actual_price is None or tts_price is None:
        return "INCOHERENT", ["missing_price"]

    try:
        if float(actual_price) != float(tts_price):
            return "INCOHERENT", ["price_mismatch"]
    except Exception:
        return "INCOHERENT", ["price_parse_error"]

    # SIGNAL COMPARISON
    if ce == tts:
        return "UNANIMOUS", []

    return "PARTIAL", ["signal_mismatch"]


# =========================
# SIV AGENT
# =========================
def siv_agent(state):
    state.setdefault("debug_log", [])
    state["debug_log"].append("SIV agent running (deterministic core)")

    llm_input = prepare_siv_payload(state)

    # =========================
    # STEP 1: DETERMINISTIC DECISION
    # =========================
    signal, issues = compute_siv(llm_input)

    # =========================
    # STEP 2: LLM EXPLANATION ONLY
    # =========================
    explanation = call_llm(llm_input)

    # =========================
    # SAFE NORMALIZATION
    # =========================
    if not explanation or explanation in ["LLM_ERROR", "LLM_EMPTY"]:
        explanation = "fallback_explanation_used"

    # extra safety (never allow None)
    explanation = str(explanation).strip()
    if not explanation:
        explanation = "fallback_explanation_used"

    # =========================
    # DEBUG OUTPUT
    # =========================
    print(f"\n[SIV OUTPUT] {signal}")
    print(f"[SIV ISSUES] {issues}")
    print(f"[SIV EXPLANATION] {explanation}\n")

    # =========================
    # OUTPUT
    # =========================
    return {
        "siv_output": {
            "signal": signal,
            "issues": issues,

            "ce_signal": llm_input.get("ce_signal"),
            "tts_signal": llm_input.get("tts_signal"),
            "actual_price": llm_input.get("actual_price"),
            "tts_price": llm_input.get("tts_price"),

            "explanation": explanation
        }
    }