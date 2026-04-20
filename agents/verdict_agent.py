import json
import requests
import re
from utils.credentials import get_do_model_key

URL = "https://inference.do-ai.run/v1/chat/completions"


def call_qwen(prompt):
    key = get_do_model_key()

    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    data = {
        "model": "deepseek-r1-distill-llama-70b",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 500,   # slightly higher for reasoning
        "temperature": 0.2
    }

    try:
        response = requests.post(URL, headers=headers, json=data, timeout=30)
    except Exception as e:
        return f"ERROR: request_failed {e}"

    if response.status_code != 200:
        return f"ERROR: status_{response.status_code} {response.text[:300]}"

    try:
        result = response.json()
    except Exception:
        return f"ERROR: invalid_json {response.text[:300]}"

    # 🔥 SAFE EXTRACTION
    try:
        choice = result["choices"][0]

        if "message" in choice:
            raw = choice["message"].get("content")
        else:
            raw = choice.get("text")

        if raw is None:
            return f"ERROR: empty_content {result}"

        return str(raw)

    except Exception:
        return f"ERROR: malformed_response {result}"


import re

def parse_llm_output(raw):
    if raw is None:
        return "HOLD", "empty_llm_output"

    if not isinstance(raw, str):
        raw = str(raw)

    text = raw.strip()

    lines = text.splitlines()

    # verdict = first valid line
    verdict = "HOLD"
    if lines:
        first = lines[0].strip().upper()
        if first in ["BUY", "SELL", "HOLD"]:
            verdict = first

    # reasoning = everything after
    reasoning = "\n".join(lines[1:]).strip()

    if not reasoning:
        reasoning = text[:300]

    return verdict, reasoning


def verdict_agent(state):
    state["debug_log"].append("VERDICT agent: LLM decision mode")

    tts = state.get("tts_output", {})
    ce  = state.get("ce_output", {})
    siv = state.get("siv_output", {})

    # =========================
    # HARD BLOCK (still deterministic)
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
    # LLM PROMPT (DECISION + REASONING)
    # =========================
    prompt = f"""
You are a strict trading decision engine.

OUTPUT FORMAT (STRICT):
First line: BUY or SELL or HOLD
Second line: explanation

RULES (MANDATORY):
- BUY if weighted_score > 0.15
- SELL if weighted_score < -0.15
- HOLD otherwise
- You MUST follow the rule exactly

REASONING REQUIREMENTS:
- Mention weighted_score explicitly
- Mention CE sentiment and TTS signal
- Mention SIV signal
- Explain alignment or conflict
- Be specific, no vague statements

INPUT:

weighted_score: {round(weighted_score, 4)}

TTS:
decision: {tts.get("decision")}
score: {tts.get("total_score")}
ema_trend: {tts.get("ema_trend")}
rsi: {tts.get("rsi_value")}
bb_signal: {tts.get("bb_signal")}

CE:
sentiment: {ce.get("sentiment")}
confidence: {ce.get("confidence")}
articles: {ce.get("article_count")}

SIV:
signal: {siv.get("signal")}
conflict: {siv.get("conflict_type")}
data_quality: {siv.get("data_quality_ok")}
"""

    raw = call_qwen(prompt)

    verdict, reasoning = parse_llm_output(raw)

    # =========================
    # RISK LOGIC (still deterministic)
    # =========================
    risk = 0.5
    if abs(weighted_score) > 0.5:
        risk = 0.8
    elif abs(weighted_score) > 0.2:
        risk = 0.6
    else:
        risk = 0.4

    print(f"\n[VERDICT] {verdict}")
    print(f"[REASONING] {reasoning}\n")

    return {
        "verdict": verdict,
        "weighted_score": round(weighted_score, 4),
        "verdict_reasoning": reasoning,
        "risk_multiplier": round(risk, 3),
        "action": "NONE"
    }