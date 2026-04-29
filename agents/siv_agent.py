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
    print("🔥 CALL_QWEN HIT FROM:", __file__)
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

    attempt = 0
    while True:
        attempt += 1
        try:
            response = requests.post(URL, headers=headers, json=data, timeout=90)

            if response.status_code == 429:
                print(f"[SIV] Rate limited on attempt {attempt}. Waiting 15s...")
                time.sleep(15)
                continue

            if response.status_code != 200:
                print(f"[SIV ERROR] HTTP {response.status_code} on attempt {attempt} — retrying in 10s...")
                time.sleep(10)
                continue

            result = response.json()
            choice = result.get("choices", [{}])[0]
            message = choice.get("message", {})
            content = message.get("content")

            if not content:
                print(f"[SIV ERROR] Empty content on attempt {attempt} — retrying in 10s...")
                time.sleep(10)
                continue

            return str(content).strip()

        except Exception as e:
            print(f"[SIV ERROR] Attempt {attempt}: {e} — retrying in 10s...")
            time.sleep(10)
            continue


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

    # 🔥 HARD SAFETY LOCK (NEW)
    force_skip = state.get("skip_llm", False)

    if backtest_mode or force_skip:
        explanation = "skipped_backtest"
    else:
        explanation = None

    print(f"\n[SIV OUTPUT] {signal}")
    print(f"[SIV ISSUES] {issues}")
    print(f"[SIV EXPLANATION] {explanation}\n")

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