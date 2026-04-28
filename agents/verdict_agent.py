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
        "max_tokens": 1024,
        "temperature": 0.2
    }

    max_retries = 3
    backoff = 5

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(URL, headers=headers, json=data, timeout=120)
        except Exception as e:
            print(f"[VERDICT ERROR] Attempt {attempt}/{max_retries}: request_failed {e}")
            if attempt < max_retries:
                print(f"[VERDICT] Retrying in {backoff}s...")
                time.sleep(backoff)
                backoff *= 2
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

    tts  = state.get("tts_output", {})
    ce   = state.get("ce_output", {})
    siv  = state.get("siv_output", {})
    pair = str(state.get("currency_pair", "")).upper()
    atr  = float(state.get("atr", 0.0))

    pair_cfg    = get_pair_config(pair)
    sl_mult     = float(pair_cfg.get("sl_mult", 1.0))
    rr_ratio    = float(pair_cfg.get("rr_ratio", 2.0))
    sl_distance = round(atr * sl_mult, 5)
    tp_distance = round(sl_distance * rr_ratio, 5)

    # =========================
    # HARD BLOCK (SIV INCOHERENT
    # still a hard block — price
    # mismatch means data is broken)
    # =========================
    if siv.get("signal") == "INCOHERENT":
        return {
            "verdict": "HOLD",
            "weighted_score": 0.0,
            "verdict_reasoning": "SIV INCOHERENT — price mismatch or missing data, no trade.",
            "risk_multiplier": 0.0,
            "atr": atr,
            "sl_distance": sl_distance,
            "tp_distance": tp_distance,
            "action": "NONE"
        }

    # =========================
    # SCORE COMPUTATION
    # Still computed and passed
    # to LLM as context — but
    # LLM decides the verdict
    # =========================
    ce_map        = {"BULLISH": 1.0, "BEARISH": -1.0, "NEUTRAL": 0.0}
    ce_signal     = ce_map.get(ce.get("sentiment", "NEUTRAL"), 0.0)
    tts_signal    = float(tts.get("total_score", 0.0))
    weighted_score = (0.6 * ce_signal) + (0.4 * tts_signal)

    # =========================
    # LLM PROMPT
    # No hard rules — LLM reasons
    # from weighted score + all
    # agent context + explanations
    # =========================
    prompt = f"""
You are an expert forex trading decision engine.

OUTPUT FORMAT (STRICT):
First line: BUY or SELL or HOLD (one word only)
Second line onwards: your reasoning

YOUR TASK:
Make a trading decision for {pair} based on all available information below.

The weighted_score is the primary directional signal, but it must be interpreted using strict thresholds and signal reliability checks.

------------------------------------------------------------
DECISION FRAMEWORK (STRICT)
------------------------------------------------------------

1. WEIGHTED_SCORE THRESHOLDS (HARD RULE):

- >= 0.15 → BUY bias
- <= -0.15 → SELL bias
- between -0.15 and 0.15 → HOLD (NEUTRAL ZONE)

IMPORTANT:
- If weighted_score is in the neutral zone, you MUST choose HOLD
- Do NOT override neutral zone even if CE or TTS suggest direction

------------------------------------------------------------
2. TRADE DECISION RULES
------------------------------------------------------------

A. BUY:
- weighted_score >= 0.15
- AND at least one supporting condition:
  - CE and TTS align bullish
  - TTS shows bullish technical confirmation (EMA uptrend, RSI supportive, BB not bearish)
  - CE confidence is MODERATE or HIGH with sufficient articles

B. SELL:
- weighted_score <= -0.15
- AND at least one supporting condition:
  - CE and TTS align bearish
  - TTS shows bearish technical confirmation (EMA downtrend, RSI weak, BB bearish)
  - CE confidence is MODERATE or HIGH with sufficient articles

C. HOLD (STRICT):
You MUST choose HOLD if ANY of the following apply:

- -0.15 < weighted_score < 0.15 (neutral zone)
- CE and TTS strongly conflict AND no clear dominance
- CE confidence is LOW AND signals are inconsistent
- SIV signal = "PARTIAL" with "signal_mismatch" AND no strong confirmation
- Risk/reward is unclear or conflicting across agents

------------------------------------------------------------
3. SIGNAL RELIABILITY ADJUSTMENTS (FOR CONVICTION ONLY)
------------------------------------------------------------

Do NOT change BUY/SELL/HOLD from this section.
Only adjust reasoning strength.

Reduce conviction if:
- CE confidence is LOW or article count < 10
- CE and TTS conflict
- SIV shows signal_mismatch

Increase conviction if:
- CE and TTS align
- CE confidence is HIGH or MODERATE with sufficient articles
- EMA_200_confidence >= 0.8
- TTS indicators (EMA, RSI, BB) agree

------------------------------------------------------------
4. IMPORTANT RULES
------------------------------------------------------------

- weighted_score determines direction AND HOLD zone
- HOLD is NOT optional in neutral zone (it is mandatory)
- Do NOT force BUY or SELL inside [-0.15, 0.15]
- Mixed signals outside neutral zone affect confidence only, not direction

------------------------------------------------------------
5. RISK MANAGEMENT (ATR BASED)
------------------------------------------------------------

SL distance: {sl_distance}
TP distance: {tp_distance}

If risk/reward is unclear or contradicts signal strength → HOLD

------------------------------------------------------------
REASONING REQUIREMENTS:
------------------------------------------------------------

- State weighted_score and classify it (BUY zone / SELL zone / HOLD zone)
- Evaluate CE (confidence, article count, sentiment strength)
- Evaluate TTS (EMA, RSI, BB, EMA200 reliability)
- Evaluate SIV (alignment or mismatch and impact)
- Justify decision strictly based on threshold logic
- Always reference actual values

------------------------------------------------------------
INPUT DATA
------------------------------------------------------------

weighted_score: {round(weighted_score, 4)}
atr: {round(atr, 5)}
sl_distance: {sl_distance}
tp_distance: {tp_distance}

TTS:
decision: {tts.get("decision")}
total_score: {tts.get("total_score")}
ema_trend: {tts.get("ema_trend")}
rsi: {tts.get("rsi")}
bb_signal: {tts.get("bb_signal")}
ema_200_confidence: {tts.get("ema_200_confidence")}
ema_200_reliable: {tts.get("ema_200_reliable")}
data_stale: {tts.get("data_stale")}
tts_explanation: {tts.get("explanation", "none")}

CE:
sentiment: {ce.get("sentiment")}
confidence: {ce.get("confidence")}
articles: {ce.get("article_count")}
raw_vibe: {ce.get("raw_vibe")}
sentiment_score: {ce.get("sentiment_score")}
mean_score: {ce.get("mean_score")}
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
        risk = 0.4
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