import json
import requests
import time
from typing import Any, Dict, Tuple
from utils.credentials import get_do_model_key
from utils.formatters import prepare_siv_payload

URL = "https://inference.do-ai.run/v1/chat/completions"


# =========================
# LLM ONLY FOR EXPLANATION
# =========================

def call_qwen(payload: Dict[str, Any]) -> str:
    key = get_do_model_key()

    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    prompt = f"""
You are a financial system integrity explanation engine.

Agent definitions:
- CE (Comparative Economic Agent): analyzes macroeconomic news and sentiment from articles to produce a directional signal
- TTS (Traditional Trading Strategies Agent): analyzes price action and technical indicators (EMA, RSI, Bollinger Bands, breakout) to produce a directional signal

Do NOT output structured format.

Explain briefly:
- CE (Comparative Economic Agent) vs TTS (Traditional Trading Strategies Agent) alignment
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
        "max_tokens": 700,
        "temperature": 0.2
    }

    max_retries = 3
    backoff = 5  # seconds

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(URL, headers=headers, json=data, timeout=90)
            print(f"[SIV EXPLANATION HTTP] status={response.status_code}")

            if response.status_code != 200:
                print(f"[SIV EXPLANATION ERROR] HTTP {response.status_code}: {response.text[:200]}")
                return "LLM_ERROR"

            result = response.json()
            choice = result.get("choices", [{}])[0]
            message = choice.get("message", {})
            content = message.get("content")

            if not content:
                return "LLM_EMPTY"

            return str(content).strip()

        except Exception as e:
            print(f"[SIV EXPLANATION ERROR] Attempt {attempt}/{max_retries}: {e}")
            if attempt < max_retries:
                print(f"[SIV EXPLANATION] Retrying in {backoff}s...")
                time.sleep(backoff)
                backoff *= 2  # exponential: 5 -> 10 -> 20

    return "LLM_ERROR"


# =========================
# PURE DETERMINISTIC CORE
# =========================
def compute_siv(payload: Dict[str, Any]) -> Tuple[str, list]:
    ce = payload.get("ce_signal")
    tts = payload.get("tts_signal")

    actual_price = payload.get("actual_price")
    tts_price = payload.get("tts_price")

    # PRICE INTEGRITY (HIGHEST PRIORITY)
    if actual_price is None or tts_price is None:
        return "INCOHERENT", ["missing_price"]

    try:
        if float(actual_price) != float(tts_price):
            return "INCOHERENT", ["price_mismatch"]
    except Exception:
        return "INCOHERENT", ["price_parse_error"]

    # =========================
    # NORMALIZE TO DIRECTION
    # CE uses BULLISH/BEARISH/NEUTRAL
    # TTS uses BUY/SELL/HOLD
    # =========================
    direction_map = {
        "BULLISH": "UP", "BUY":  "UP",
        "BEARISH": "DOWN", "SELL": "DOWN",
        "NEUTRAL": "FLAT", "HOLD": "FLAT",
    }

    ce_dir  = direction_map.get(str(ce).upper(),  "UNKNOWN")
    tts_dir = direction_map.get(str(tts).upper(), "UNKNOWN")

    if "UNKNOWN" in (ce_dir, tts_dir):
        return "PARTIAL", ["unrecognized_signal"]

    if ce_dir == tts_dir:
        return "COHERENT", []

    if "FLAT" in (ce_dir, tts_dir):
        return "PARTIAL", ["one_signal_neutral"]

    return "PARTIAL", ["signal_mismatch"]


# =========================
# SIV AGENT
# =========================
def siv_agent(state):
    state.setdefault("debug_log", [])
    state["debug_log"].append("SIV agent running (deterministic core)")

    llm_input = prepare_siv_payload(state)
    backtest_mode = state.get("backtest_mode", False)

    signal, issues = compute_siv(llm_input)

    # STEP 2: LLM EXPLANATION ONLY
    if not backtest_mode:
        explanation = call_qwen(llm_input)
    else:
        explanation = "skipped_backtest"

    # =========================
    # SAFE NORMALIZATION
    # =========================
    if not explanation or explanation in ["LLM_ERROR", "LLM_EMPTY"]:
        explanation = "fallback_explanation_used"

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