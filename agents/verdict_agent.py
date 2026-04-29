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
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
        "temperature": 0.2
    }

    attempt = 0
    backoff = 10
    max_backoff = 60

    while True:
        attempt += 1
        try:
            response = requests.post(URL, headers=headers, json=data, timeout=90)

            if response.status_code == 429:
                print(f"[VERDICT] Rate limited on attempt {attempt}. Waiting {backoff}s...")
                time.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)
                continue

            if response.status_code in [400, 401, 403]:
                print(f"[VERDICT FATAL] HTTP {response.status_code}: {response.text}")
                return "explanation_unavailable"

            if response.status_code != 200:
                print(f"[VERDICT ERROR] HTTP {response.status_code} on attempt {attempt} — retrying...")
                time.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)
                continue

            result = response.json()
            choice = result.get("choices", [{}])[0]
            message = choice.get("message", {})
            content = message.get("content") or message.get("reasoning_content")

            if not content:
                print(f"[VERDICT ERROR] Empty content on attempt {attempt} — retrying...")
                time.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)
                continue

            return str(content).strip()

        except Exception as e:
            print(f"[VERDICT ERROR] Attempt {attempt}: {e} — retrying...")
            time.sleep(backoff)
            backoff = min(backoff * 2, max_backoff)


def parse_llm_output(raw):
    if raw is None:
        return "HOLD", "empty_llm_output"

    if not isinstance(raw, str):
        raw = str(raw)

    text = raw.strip()
    lines = text.splitlines()

    verdict = "HOLD"
    for line in lines[:3]:
        candidate = line.strip().upper()
        if candidate in ["BUY", "SELL", "HOLD"]:
            verdict = candidate
            break

    reasoning = "\n".join(lines[1:]).strip()
    if not reasoning:
        reasoning = text[:300]

    return verdict, reasoning


# =========================
# DETERMINISTIC VERDICT
# Used in backtest mode — no LLM call.
# Mirrors the exact thresholds in the LLM prompt so results are consistent.
# =========================
def compute_verdict_deterministic(weighted_score: float) -> tuple[str, str]:
    if weighted_score >= 0.15:
        verdict = "BUY"
    elif weighted_score <= -0.15:
        verdict = "SELL"
    else:
        verdict = "HOLD"

    reasoning = (
        f"[backtest deterministic] weighted_score={round(weighted_score, 4)} → {verdict} "
        f"(thresholds: >=0.15 BUY, <=-0.15 SELL, else HOLD)"
    )
    return verdict, reasoning


def verdict_agent(state):
    state["debug_log"].append("VERDICT agent: LLM decision mode")

    tts  = state.get("tts_output", {})
    ce   = state.get("ce_output", {})
    siv  = state.get("siv_output", {})
    pair = str(state.get("currency_pair", "")).upper()
    atr  = float(state.get("atr", 0.0))

    # =========================
    # READ BACKTEST FLAG EARLY
    # This is the key fix: skip LLM entirely in backtest mode
    # =========================
    backtest_mode = bool(state.get("backtest_mode", False))

    pair_cfg    = get_pair_config(pair)
    sl_mult     = float(pair_cfg.get("sl_mult", 1.0))
    rr_ratio    = float(pair_cfg.get("rr_ratio", 2.0))
    sl_distance = round(atr * sl_mult, 5)
    tp_distance = round(sl_distance * rr_ratio, 5)

    # =========================
    # HARD BLOCK (SIV INCOHERENT)
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
    # =========================
    ce_map = {"BULLISH": 1.0, "BEARISH": -1.0, "NEUTRAL": 0.0}
    ce_signal = ce_map.get(ce.get("sentiment", "NEUTRAL"), 0.0)
    tts_signal = float(tts.get("total_score", 0.0))

    weighted_score = (0.6 * ce_signal) + (0.4 * tts_signal)
    weighted_score = max(-1.0, min(weighted_score, 1.0))

    # =========================
    # DYNAMIC SL/TP
    # =========================
    price = float(tts.get("price", 1e-9))
    atr_pct = atr / price if price > 0 else 0

    if atr_pct < 0.002:
        vol_mult = 1.6
    elif atr_pct < 0.005:
        vol_mult = 2.2
    else:
        vol_mult = 3.0

    ema_trend = tts.get("ema_trend", "NEUTRAL")
    trend_regime = "TRENDING" if ema_trend in ["BULLISH", "BEARISH"] else "RANGING"

    sl_distance = round(atr * sl_mult, 5)
    base_tp = sl_distance * rr_ratio * vol_mult
    base_tp *= 1.15 if trend_regime == "TRENDING" else 0.9
    tp_distance = round(base_tp, 5)

    # =========================
    # VERDICT — BACKTEST: deterministic, LIVE: LLM
    # =========================
    if backtest_mode:
        verdict, reasoning = compute_verdict_deterministic(weighted_score)
        print(f"\n[VERDICT] {verdict} | BACKTEST DETERMINISTIC | score={round(weighted_score, 4)}")
    else:
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
        print(f"\n[VERDICT] {verdict} | ATR={atr:.5f} | SL={sl_distance} | TP={tp_distance}")
        print(f"[REASONING] {reasoning}\n")

    # =========================
    # RISK LOGIC
    # =========================
    if atr == 0.0:
        risk = 0.4
    elif abs(weighted_score) > 0.5:
        risk = 0.8
    elif abs(weighted_score) > 0.2:
        risk = 0.6
    else:
        risk = 0.4

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