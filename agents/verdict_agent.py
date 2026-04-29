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

        if response.status_code == 429:
            wait = backoff * attempt
            print(f"[VERDICT] Rate limited. Waiting {wait}s before retry {attempt}/{max_retries}...")
            time.sleep(wait)
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
    for line in lines[:3]:  # check first 3 lines, not just line 0
        candidate = line.strip().upper()
        if candidate in ["BUY", "SELL", "HOLD"]:
            verdict = candidate
            break

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
    # SCORE COMPUTATION (OLD LOGIC RESTORED)
    # =========================

    ce_map = {"BULLISH": 1.0, "BEARISH": -1.0, "NEUTRAL": 0.0}
    ce_signal = ce_map.get(ce.get("sentiment", "NEUTRAL"), 0.0)

    tts_signal = float(tts.get("total_score", 0.0))

    # OLD SIMPLE WEIGHTED SCORE (RESTORED)
    weighted_score = (0.6 * ce_signal) + (0.4 * tts_signal)

    # clamp safety (kept from newer system)
    weighted_score = max(-1.0, min(weighted_score, 1.0))

        # =========================
    # FIX 2: DYNAMIC SL/TP (VOLATILITY REGIME IMPROVED)
    # =========================

    price = float(tts.get("price", 1e-9))
    atr_pct = atr / price if price > 0 else 0

    # volatility regime (ATR-based, more stable than price-dependent scaling)
    if atr_pct < 0.002:
        volatility = "LOW"
        vol_mult = 1.6
    elif atr_pct < 0.005:
        volatility = "MEDIUM"
        vol_mult = 2.2
    else:
        volatility = "HIGH"
        vol_mult = 3.0

    ema_trend = tts.get("ema_trend", "NEUTRAL")
    trend_regime = "TRENDING" if ema_trend in ["BULLISH", "BEARISH"] else "RANGING"

    # SL stays ATR-based (unchanged core logic)
    sl_distance = round(atr * sl_mult, 5)

    # base TP
    base_tp = sl_distance * rr_ratio * vol_mult

    # trend adjustment (kept but normalized so it doesn't explode TP)
    if trend_regime == "TRENDING":
        base_tp *= 1.15
    else:
        base_tp *= 0.9

    tp_distance = round(base_tp, 5)

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

------------------------------------------------------------
DECISION FRAMEWORK (STRICT)
------------------------------------------------------------

1. WEIGHTED_SCORE IS THE ONLY DIRECTION RULE:

weighted_score = (0.6 * CE_signal) + (0.4 * TTS_signal)

- >= 0.15  → BUY
- <= -0.15 → SELL
- between -0.15 and 0.15 → HOLD (mandatory, no override)

The weighted_score already accounts for CE/TTS conflict mathematically.
Do NOT override BUY or SELL to HOLD because CE and TTS disagree directionally.
Signal conflict is already priced into the score.

------------------------------------------------------------
2. CONVICTION ADJUSTMENTS (DO NOT CHANGE VERDICT)
------------------------------------------------------------

These only affect reasoning strength, never the BUY/SELL/HOLD decision:

Reduce conviction if:
- CE confidence is LOW or article count < 10
- CE and TTS point in opposite directions
- SIV signal = PARTIAL with signal_mismatch

Increase conviction if:
- CE and TTS align directionally
- CE confidence HIGH or MODERATE with sufficient articles
- EMA_200_confidence >= 0.8

------------------------------------------------------------
3. REASONING REQUIREMENTS
------------------------------------------------------------

- State weighted_score and its zone
- Evaluate CE (confidence, article count, sentiment strength)
- Evaluate TTS (EMA, RSI, BB, MACD, EMA200 reliability)
- Evaluate SIV (alignment or mismatch — conviction impact only)
- Justify decision from weighted_score threshold only
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