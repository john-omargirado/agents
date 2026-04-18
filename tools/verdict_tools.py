import re


def normalize_signal(value, default="HOLD"):
    if value is None:
        return default

    value = str(value).strip().upper()

    # exact matches first (strong priority)
    if value == "BUY":
        return "BUY"
    if value == "SELL":
        return "SELL"
    if value == "HOLD":
        return "HOLD"

    # fuzzy matches
    if "BUY" in value or "LONG" in value:
        return "BUY"
    if "SELL" in value or "SHORT" in value:
        return "SELL"
    if "HOLD" in value or "WAIT" in value:
        return "HOLD"

    return default

def normalize_sentiment(value):
    if value is None:
        return "NEUTRAL"

    value = str(value).strip().upper()

    if value == "BULLISH":
        return "BULLISH"
    if value == "BEARISH":
        return "BEARISH"
    if value == "NEUTRAL":
        return "NEUTRAL"

    if "BULL" in value or "POSITIVE" in value:
        return "BULLISH"
    if "BEAR" in value or "NEGATIVE" in value:
        return "BEARISH"

    return "NEUTRAL"


def extract_llm_fields(text):
    decision = None
    reasoning = None

    if not text:
        return "HOLD", "No output from LLM"

    text = str(text)

    # --- DECISION (strict) ---
    decision_match = re.search(
        r"DECISION:\s*(BUY|SELL|HOLD)",
        text,
        re.IGNORECASE
    )

    if decision_match:
        decision = decision_match.group(1).upper()

    # --- REASONING (non-greedy, stops before next field) ---
    reasoning_match = re.search(
        r"REASONING:\s*(.*?)(?:\n[A-Z_]+:|\Z)",
        text,
        re.IGNORECASE | re.DOTALL
    )

    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()

    # --- Fallbacks ---
    if not decision:
        decision = normalize_signal(text)

    if not reasoning:
        reasoning = "No clear reasoning provided"

    return decision, reasoning


def build_reason(
    mode,
    tts_decision,
    ce_sentiment,
    article_count,
    conflict,
    risk_multiplier,
    extra=None
):
    parts = []

    parts.append(f"mode={mode}")
    parts.append(f"tech={tts_decision}")
    parts.append(f"sentiment={ce_sentiment}")
    parts.append(f"articles={article_count}")

    if conflict:
        parts.append("conflict=1")
    else:
        parts.append("conflict=0")

    parts.append(f"risk={round(risk_multiplier, 2)}")

    if extra:
        # clean extra text (avoid pipes breaking logs)
        clean_extra = str(extra).replace("|", "/").strip()
        parts.append(f"note={clean_extra}")

    return " | ".join(parts)