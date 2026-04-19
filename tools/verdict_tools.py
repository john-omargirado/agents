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