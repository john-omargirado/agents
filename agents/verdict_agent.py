import json
import requests
import time
import re
from utils.credentials import get_do_model_key
from utils.trade_config import get_pair_config

URL = "https://inference.do-ai.run/v1/chat/completions"


def call_qwen(prompt):
    key = get_do_model_key()

    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    data = {
        "model": "alibaba-qwen3-32b",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 2048,
        "temperature": 0.2
    }

    max_retries = 3
    backoff = 5  # seconds

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(URL, headers=headers, json=data, timeout=120)
        except Exception as e:
            print(f"[VERDICT ERROR] Attempt {attempt}/{max_retries}: request_failed {e}")
            if attempt < max_retries:
                print(f"[VERDICT] Retrying in {backoff}s...")
                time.sleep(backoff)
                backoff *= 2  # exponential: 5 -> 10 -> 20
            continue

        if response.status_code != 200:
            return f"ERROR: status_{response.status_code} {response.text[:300]}"

        try:
            result = response.json()
        except Exception:
            return f"ERROR: invalid_json {response.text[:300]}"

        try:
            choice = result["choices"][0]

            if "message" in choice:
                message = choice["message"]
                raw = message.get("content") or message.get("reasoning_content")
            else:
                raw = choice.get("text")

            if raw is None:
                return f"ERROR: empty_content {result}"

            return str(raw)

        except Exception:
            return f"ERROR: malformed_response {result}"

    return "ERROR: request_failed max_retries_exceeded"


def parse_llm_output(raw):
    if raw is None:
        return "HOLD", "empty_llm_output"

    if not isinstance(raw, str):
        raw = str(raw)

    text = raw.strip()
    lines = text.splitlines()

    verdict = "HOLD"
    if lines:
        first = lines[0].strip().upper()
        if first in ["BUY", "SELL", "HOLD"]:
            verdict = first

    reasoning = "\n".join(lines[1:]).strip()
    if not reasoning:
        reasoning = text[:300]

    return verdict, reasoning

def verdict_agent(state):
    state["debug_log"].append("VERDICT agent: LLM decision mode")

    tts = state.get("tts_output", {})
    ce  = state.get("ce_output", {})
    siv = state.get("siv_output", {})
    pair = str(state.get("currency_pair", "")).upper()
    atr = float(state.get("atr", 0.0))

    pair_cfg = get_pair_config(pair)
    sl_mult = float(pair_cfg.get("sl_mult", 1.0))
    rr_ratio = float(pair_cfg.get("rr_ratio", 2.0))

    sl_distance = round(atr * sl_mult, 5)
    tp_distance = round(sl_distance * rr_ratio, 5)

    # =========================
    # HARD BLOCK (deterministic)
    # =========================
    if siv.get("signal") == "INCOHERENT":
        return {
            "verdict": "HOLD",
            "weighted_score": 0.0,
            "verdict_reasoning": "SIV INCOHERENT",
            "risk_multiplier": 0.0,
            "atr": atr,
            "sl_distance": sl_distance,
            "tp_distance": tp_distance,
            "action": "NONE"
        }

    # =========================
    # SCORE COMPUTATION
    # =========================
    ce_map = {"BULLISH": 1.0, "BEARISH": -1.0, "NEUTRAL": 0.0}
    ce_signal = ce_map.get(ce.get("sentiment", "NEUTRAL"), 0.0)
    tts_signal = float(tts.get("total_score", 0.0))
    weighted_score = (0.6 * ce_signal) + (0.4 * tts_signal)

    # =========================
    # LLM PROMPT
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
- Mention CE sentiment and its confidence — if CE explanation reveals internal contradiction, reduce trust in CE signal
- Mention TTS signal and whether tts_insufficient is true
- Mention SIV signal and whether signals are aligned or conflicting
- Mention ATR-based SL ({sl_distance}) and TP ({tp_distance}) and whether the risk/reward is acceptable
- If CE and TTS conflict and CE confidence is LOW or MODERATE, lean toward TTS
- Be specific, no vague statements

INPUT:

weighted_score: {round(weighted_score, 4)}
atr: {round(atr, 5)}
sl_distance: {sl_distance}
tp_distance: {tp_distance}

TTS:
decision: {tts.get("decision")}
score: {tts.get("total_score")}
ema_trend: {tts.get("ema_trend")}
rsi: {tts.get("rsi_value")}
bb_signal: {tts.get("bb_signal")}
tts_insufficient: {tts.get("tts_insufficient")}
tts_explanation: {tts.get("explanation", "none")}

CE:
sentiment: {ce.get("sentiment")}
confidence: {ce.get("confidence")}
articles: {ce.get("article_count")}
raw_vibe: {ce.get("raw_vibe")}
sentiment_score: {ce.get("sentiment_score")}
ce_explanation: {ce.get("explanation", "none")}

SIV:
signal: {siv.get("signal")}
issues: {siv.get("issues")}
siv_explanation: {siv.get("explanation", "none")}
"""

    raw = call_qwen(prompt)
    verdict, reasoning = parse_llm_output(raw)

    # =========================
    # RISK LOGIC (ATR-aware)
    # =========================
    if atr == 0.0:
        risk = 0.4  # fallback if ATR missing
    elif abs(weighted_score) > 0.5:
        risk = 0.8
    elif abs(weighted_score) > 0.2:
        risk = 0.6
    else:
        risk = 0.4

    print(f"\n[VERDICT] {verdict} | ATR={atr:.5f} | SL={sl_distance} | TP={tp_distance}")
    print(f"[REASONING] {reasoning}\n")

    return {
        "verdict": verdict,
        "weighted_score": round(weighted_score, 4),
        "verdict_reasoning": reasoning,
        "risk_multiplier": round(risk, 3),
        "atr": atr,
        "sl_distance": sl_distance,
        "tp_distance": tp_distance,
        "action": "NONE"
    }