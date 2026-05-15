import json
import requests
import time
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

def compute_verdict_deterministic(weighted_score: float, threshold: float = 0.05) -> tuple[str, str]:
    if weighted_score >= threshold:
        verdict = "BUY"
    elif weighted_score <= -threshold:
        verdict = "SELL"
    else:
        verdict = "HOLD"

    reasoning = (
        f"[deterministic] weighted_score={round(weighted_score, 4)} → {verdict} "
        f"(thresholds: >={threshold} BUY, <=-{threshold} SELL)"
    )
    return verdict, reasoning


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


def verdict_agent(state):
    state["debug_log"].append("VERDICT agent: LLM decision mode")

    tts  = state.get("tts_output", {})
    ce   = state.get("ce_output", {})
    siv  = state.get("siv_output", {})
    pair = str(state.get("currency_pair", "")).upper()
    atr_raw = tts.get("atr") or state.get("atr") or 0.0
    atr = max(float(atr_raw), 0.0001)

    backtest_mode = bool(state.get("backtest_mode", False))
    skip_llm      = bool(state.get("skip_llm", False))

    print("CALIBRATION THRESHOLD:", state.get("calibration_threshold"))
    print("BACKTEST MODE:", backtest_mode)
    print("SKIP LLM:", skip_llm)


    print(f"[VERDICT ENTRY] risk_per_trade={state.get('risk_per_trade')} | account_capital={state.get('account_capital')}")
    pair_cfg    = get_pair_config(pair)
    sl_mult     = float(pair_cfg.get("sl_mult", 1.0))
    rr_ratio    = float(pair_cfg.get("rr_ratio", 2.0))


    capital = float(state["account_capital"])

    risk_pct = float(state.get("risk_per_trade", 1)) / 100

    leverage_str = state.get("leverage", "1:1")
    leverage = float(leverage_str.split(":")[1])

    risk_amount = 0.0
    position_size = 0.0
    max_exposure = capital * leverage

    # =========================
    # HARD BLOCK (SIV INCOHERENT)
    # =========================
    if siv.get("signal") == "INCOHERENT":
        risk_scale  = float(state.get("risk_per_trade", 1.0))
        sl_distance = round(atr * sl_mult * risk_scale, 5)
        tp_distance = round(sl_distance * rr_ratio, 5)

        return {
            "verdict": "HOLD",
            "weighted_score": 0.0,
            "verdict_reasoning": "SIV INCOHERENT — price mismatch or missing data.",
            "risk_multiplier": 0.0,
            "atr": atr,
            "sl_distance": sl_distance,
            "tp_distance": tp_distance,
            "position_size": 0.0,
            "risk_amount": 0.0,
            "max_exposure": max_exposure,
            "action": "NONE"
        }

    # =========================
    # SCORE COMPUTATION
    # =========================
    siv_multiplier = float(siv.get("score_multiplier", 1.0))
    ce_score_raw = float(ce.get("ce_score", 0.0))
    ce_conf      = float(ce.get("ce_confidence", 0.0))
    tts_score    = float(tts.get("total_score", 0.0))

    ce_weight  = 0.35 + (0.30 * ce_conf)
    tts_weight = 1.0 - ce_weight

    weighted_score = (ce_weight * ce_score_raw) + (tts_weight * tts_score)

    # =========================
    # SIGNAL QUALITY GATE
    # =========================
    ce_article_count = int(ce.get("article_count", 0))
    ce_strong  = ce_article_count >= 5 and abs(ce_score_raw) >= 0.05
    tts_strong = abs(tts_score) >= 0.05
    siv_issues = siv.get("issues", [])

    tradeable = True
    skip_reason = None

    if siv.get("signal") == "INCOHERENT":
        tradeable = False
        skip_reason = "signal_mismatch — need CE strong (TTS unreliable)"
    elif siv.get("signal") == "PARTIAL":
        weighted_score *= 0.7
    elif not ce_strong and not tts_strong:
        tradeable   = False
        skip_reason = f"weak signals — articles={ce_article_count} |ce|={abs(ce_score_raw):.3f} |tts|={abs(tts_score):.3f}"
    else:
        tradeable   = True
        skip_reason = None

    if not tradeable:
        return {
            "verdict": "HOLD",
            "weighted_score": 0.0,
            "verdict_reasoning": f"SKIP: {skip_reason}",
            "risk_multiplier": 0.0,
            "atr": atr,
            "sl_distance": round(atr * sl_mult * float(state.get("risk_per_trade", 1.0)), 5),
            "tp_distance": 0.0,
            "position_size": 0.0,
            "risk_amount": 0.0,
            "max_exposure": max_exposure,
            "action": "SKIP"
        }

    # Apply SIV score multiplier
    
    weighted_score = weighted_score * siv_multiplier
    weighted_score = max(-1.0, min(weighted_score, 1.0))

    # =========================
    # DYNAMIC SL/TP
    # =========================
    ema_trend    = tts.get("ema_trend", "NEUTRAL")
    trend_regime = "TRENDING" if ema_trend in ["BULLISH", "BEARISH"] else "RANGING"

    price = float(tts.get("price", 1e-9))

    risk_scale  = float(state.get("risk_per_trade", 1.0))
    sl_distance = round(atr * sl_mult * risk_scale, 5)
    print(f"[RISK_SCALE DEBUG] raw={state.get('risk_per_trade')} | type={type(state.get('risk_per_trade'))} | risk_scale={risk_scale}")
    if sl_distance <= 0:
        return {
            "verdict": "HOLD",
            "weighted_score": 0.0,
            "verdict_reasoning": "Invalid SL distance (ATR too small)",
            "risk_multiplier": 0.0,
            "atr": atr,
            "sl_distance": 0,
            "tp_distance": 0,
            "position_size": 0.0,
            "risk_amount": 0.0,
            "max_exposure": max_exposure,
            "action": "NONE"
        }
    
    risk_amount = capital * risk_pct
    print(f"\n[VERDICT] risk_amount={risk_amount} | capital={capital} risk_pct={risk_pct*100}%")
    position_size = risk_amount / (sl_distance * 100000)
    base_tp  = sl_distance * rr_ratio
    base_tp *= 1.15 if trend_regime == "TRENDING" else 0.9
    tp_distance = round(base_tp, 5)
    print(f"[SANITY] ATR={atr:.5f} | SL={sl_distance:.5f} | TP={tp_distance:.5f} | price={price:.5f}")

    max_exposure = capital * leverage
    trade_exposure = position_size * price * 100000

    if trade_exposure > max_exposure:
        position_size = max_exposure / (price * 100000)
    
    max_allowed_lot = max_exposure / (price * 100000)
    position_size = min(position_size, max_allowed_lot)

    # =========================
    # MODE SWITCH (UPDATED)
    # =========================
    use_deterministic = backtest_mode or skip_llm
    if use_deterministic:
        deterministic_threshold = float(state.get("calibration_threshold", 0.05))
        verdict, reasoning = compute_verdict_deterministic(weighted_score, deterministic_threshold)
        print(f"\n[VERDICT] {verdict} | DETERMINISTIC | score={round(weighted_score, 4)}")
    else:
        experience_level = str(state.get("experience_level", "intermediate") or "intermediate").lower()

        if experience_level == "beginner":
            tone_instruction = """\
STYLE — BEGINNER (MANDATORY, NO EXCEPTIONS):
You are explaining this to someone who has NEVER heard of forex trading before.
Write exactly 2-3 plain, warm sentences — like texting a friend who asked what happened.

FORBIDDEN — never use these in your reasoning:
weighted_score, ce_weight, tts_score, ce_score, siv_multiplier, score_multiplier,
atr, sl, tp, COHERENT, INCOHERENT, PARTIAL, EMA, RSI, BB, MACD, regime, adx,
ema_trend, bb_signal, ce_confidence, article_count, threshold

INSTEAD say things like:
"the charts", "the news", "market conditions", "price direction", "both sides agreed",
"conditions look good", "signals weren't strong enough", "the system wasn't sure"

EXAMPLE — correct beginner BUY reasoning:
"Both the news and the charts were pointing upward today, and they agreed with each other — so the system flagged this as a potential buying opportunity. Think of it like a green light: most signals lined up in the same direction. Just keep in mind this is for learning, not a guarantee — always be careful with real money."

EXAMPLE — correct beginner HOLD reasoning:
"The system decided to sit this one out. The news and the charts weren't sending a strong enough signal in either direction, so it chose to wait rather than guess. That's actually healthy — not every moment is the right time to trade, and patience is part of learning."""

        elif experience_level == "basic":
            tone_instruction = """\
STYLE — BASIC (MANDATORY):
Write 2-3 conversational sentences for someone who knows pips, charts, and basic indicators.
You may mention RSI, EMA trend, and news sentiment — but explain what each contributed, don't just list them.
Do not use raw field names like weighted_score, siv_multiplier, ce_weight, or atr.
Keep it readable, not like a data dump."""

        else:
            tone_instruction = """\
STYLE — INTERMEDIATE:
Write 2-3 concise technical sentences.
You may reference weighted_score, ce_weight, SIV signal, regime, and indicator values directly.
Cover what drove the decision and any relevant risk factors."""

        decision_word = "BUY" if weighted_score >= 0.05 else "SELL" if weighted_score <= -0.05 else "HOLD"
        ce_direction  = ce.get("sentiment", "NEUTRAL")
        tts_direction = tts.get("decision", "HOLD")
        siv_status    = siv.get("signal", "UNKNOWN")
        market_regime = tts.get("regime", "UNKNOWN")
        ema_dir       = tts.get("ema_trend", "NEUTRAL")
        rsi_val       = tts.get("rsi", "N/A")
        bb_val        = tts.get("bb_signal", "N/A")
        art_count     = ce.get("article_count", 0)
        ce_conf_tier  = ce.get("confidence", "LOW")

        prompt = f"""/no_think
{tone_instruction}

Now apply the style above to write the reasoning for this decision.

DECISION (already computed — do not change it): {decision_word}

OUTPUT FORMAT:
Line 1: {decision_word}
Line 2 onwards: your reasoning using ONLY the style described above

CONTEXT FOR YOUR REASONING (use to inform your words, do not copy these field names):
- Overall direction: news was {ce_direction}, charts said {tts_direction}
- News confidence: {ce_conf_tier} ({art_count} articles)
- Chart market condition: {market_regime} (trend direction: {ema_dir})
- Price momentum (RSI): {rsi_val} — above 60 means overbought, below 40 means oversold, middle is neutral
- Bollinger Bands: {bb_val}
- Signal agreement check: {siv_status}
- Combined score: {round(weighted_score, 4)} (above 0.05 = buy, below -0.05 = sell, in between = hold)
"""
        raw = call_qwen(prompt)
        verdict, reasoning = parse_llm_output(raw)

        # Safety: if model still returns technical output for beginners, replace it
        if experience_level == "beginner":
            forbidden = ["weighted_score", "ce_weight", "siv_multiplier", "atr=", "ce_score=",
                         "tts_score", "COHERENT", "INCOHERENT", "ema=", "bb=", "regime="]
            if any(term in reasoning for term in forbidden):
                direction_map = {
                    "BUY":  "Both the charts and the news were leaning upward today and agreed with each other, so the system flagged a potential buying opportunity. Think of it like a green light — most signals pointed the same way. Remember, this is for learning purposes only, so always be careful before making any real decisions.",
                    "SELL": "Both the charts and the news were leaning downward today, so the system flagged a potential selling opportunity. Think of it like a red light — the signals suggested caution about the price going higher. This is for learning only, so always do your own research before acting.",
                    "HOLD": "The system decided to sit this one out — the signals weren't strong or clear enough in either direction to feel confident. Think of it like a yellow light: it's saying wait and see. That's perfectly normal, and patience is an important part of learning to trade.",
                }
                reasoning = direction_map.get(verdict, direction_map["HOLD"])

        print(f"\n[VERDICT] {verdict} | LLM | score={round(weighted_score, 4)}")

    # =========================
    # RISK MULTIPLIER
    # =========================
    if atr == 0.0:
        risk = 0.4
    elif abs(weighted_score) > 0.5:
        risk = 0.8
    elif abs(weighted_score) > 0.2:
        risk = 0.6
    else:
        risk = 0.4

    trade_output = {
        "position_size": position_size,
        "risk_amount": risk_amount,
        "max_exposure": max_exposure,
        "sl_distance": sl_distance,
        "tp_distance": tp_distance,
        "atr": atr
    }

    return {
        "verdict": verdict,
        "weighted_score": weighted_score,
        "risk_multiplier": risk,
        "verdict_reasoning": reasoning,
        "trade_output": trade_output,
        "atr": atr,                    
        "action": verdict if verdict in ["BUY", "SELL"] else "NONE"
    }