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
    atr  = float(state.get("atr", 0.0))

    backtest_mode = bool(state.get("backtest_mode", False))
    skip_llm      = bool(state.get("skip_llm", False))

    pair_cfg    = get_pair_config(pair)
    sl_mult     = float(pair_cfg.get("sl_mult", 1.0))
    rr_ratio    = float(pair_cfg.get("rr_ratio", 2.0))

    # =========================
    # HARD BLOCK (SIV INCOHERENT)
    # =========================
    if siv.get("signal") == "INCOHERENT":
        sl_distance = round(atr * sl_mult, 5)
        tp_distance = round(sl_distance * rr_ratio, 5)
        return {
            "verdict": "HOLD",
            "weighted_score": 0.0,
            "verdict_reasoning": "SIV INCOHERENT — price mismatch or missing data.",
            "risk_multiplier": 0.0,
            "atr": atr,
            "sl_distance": sl_distance,
            "tp_distance": tp_distance,
            "action": "NONE"
        }

    # =========================
    # SCORE COMPUTATION — continuous ce_score + adaptive weight
    # =========================
    ce_score_raw = float(ce.get("ce_score", 0.0))
    ce_conf      = float(ce.get("ce_confidence", 0.0))
    tts_score    = float(tts.get("total_score", 0.0))

    ce_weight  = 0.35 + (0.30 * ce_conf)   # 0.35 → 0.65 based on data quality
    tts_weight = 1.0 - ce_weight

    weighted_score = (ce_weight * ce_score_raw) + (tts_weight * tts_score)

    # =========================
    # SIGNAL QUALITY GATE
    # =========================
    ce_article_count = int(ce.get("article_count", 0))
    ce_strong  = ce_article_count >= 10 and abs(ce_score_raw) >= 0.05
    tts_strong = abs(tts_score) >= 0.08
    siv_issues = siv.get("issues", [])

    if siv.get("signal") == "PARTIAL" and "signal_mismatch" in siv_issues:
        # ✅ FIX: was (ce_strong and tts_strong) — TTS is 0 for 83% of days
        # so this gate was blocking almost every PARTIAL trade
        tradeable   = ce_strong          # ← only require CE when TTS is unavailable
        skip_reason = "signal_mismatch — need CE strong (TTS unreliable)"
    elif not ce_strong and not tts_strong:
        tradeable   = False
        skip_reason = f"weak signals — articles={ce_article_count} |ce|={abs(ce_score_raw):.3f} |tts|={abs(tts_score):.3f}"
    else:
        tradeable   = True
        skip_reason = None

    if not tradeable:
        print(f"\n[VERDICT] SKIP — {skip_reason}")
        return {
            "verdict":           "HOLD",
            "weighted_score":    0.0,
            "verdict_reasoning": f"SKIP: {skip_reason}",
            "risk_multiplier":   0.0,
            "atr":               atr,
            "sl_distance":       round(atr * sl_mult, 5),
            "tp_distance":       0.0,
            "action":            "SKIP"
        }

    # Apply SIV score multiplier
    siv_multiplier = float(siv.get("score_multiplier", 1.0))
    weighted_score = weighted_score * siv_multiplier
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

    ema_trend    = tts.get("ema_trend", "NEUTRAL")
    trend_regime = "TRENDING" if ema_trend in ["BULLISH", "BEARISH"] else "RANGING"

    sl_distance = round(atr * sl_mult, 5)
    base_tp     = sl_distance * rr_ratio * vol_mult
    base_tp    *= 1.15 if trend_regime == "TRENDING" else 0.9
    tp_distance = round(base_tp, 5)

    # =========================
    # VERDICT
    # =========================
    if backtest_mode or skip_llm:
        deterministic_threshold = float(state.get("calibration_threshold", 0.20))
        verdict, reasoning = compute_verdict_deterministic(weighted_score, deterministic_threshold)
        print(f"\n[VERDICT] {verdict} | DETERMINISTIC | score={round(weighted_score, 4)}")
    else:
        prompt = f"""
You are an expert forex trading decision engine.

OUTPUT FORMAT (STRICT):
First line: BUY or SELL or HOLD (one word only)
Second line onwards: your reasoning

DECISION RULE (NON-NEGOTIABLE):
weighted_score >= 0.20  → BUY
weighted_score <= -0.20 → SELL
between -0.20 and 0.20  → HOLD

weighted_score: {round(weighted_score, 4)}
ce_weight used: {round(ce_weight, 2)} (based on ce_confidence={ce_conf})
atr: {round(atr, 5)} | sl: {sl_distance} | tp: {tp_distance}

TTS: decision={tts.get("decision")} score={tts.get("total_score")} ema={tts.get("ema_trend")} rsi={tts.get("rsi")} bb={tts.get("bb_signal")} regime={tts.get("regime", "UNKNOWN")}
CE:  sentiment={ce.get("sentiment")} confidence={ce.get("confidence")} articles={ce.get("article_count")} ce_score={ce_score_raw}
SIV: signal={siv.get("signal")} issues={siv.get("issues")} multiplier={siv_multiplier}
"""
        raw = call_qwen(prompt)
        verdict, reasoning = parse_llm_output(raw)
        print(f"\n[VERDICT] {verdict} | LLM | score={round(weighted_score, 4)}")

    # =========================
    # RISK MULTIPLIER (clean, no variable collision)
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
        "verdict":           verdict,
        "weighted_score":    weighted_score,
        "verdict_reasoning": reasoning,
        "risk_multiplier":   risk,
        "atr":               atr,
        "sl_distance":       sl_distance,
        "tp_distance":       tp_distance,
        "action":            verdict if verdict in ["BUY", "SELL"] else "NONE"
    }


def compute_verdict_deterministic(weighted_score: float, threshold: float = 0.10) -> tuple[str, str]:
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