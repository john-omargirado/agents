def check_data_integrity(state):
    tts_data = state.get("tts_output", {})
    indicators = tts_data.get("indicators", {})

    tts_price = indicators.get("price")
    actual_price = state.get("price")

    issues = []

    # --- Presence checks ---
    if tts_price is None:
        issues.append("missing_tts_price")
    if actual_price is None:
        issues.append("missing_actual_price")

    if issues:
        return {
            "pass": False,
            "deviation": 1.0,
            "issues": issues
        }

    try:
        tts_price = float(tts_price)
        actual_price = float(actual_price)

        # --- Sanity checks ---
        if actual_price <= 0:
            return {
                "pass": False,
                "deviation": 1.0,
                "issues": ["invalid_actual_price"]
            }

        if tts_price <= 0:
            return {
                "pass": False,
                "deviation": 1.0,
                "issues": ["invalid_tts_price"]
            }

        # --- Deviation check ---
        deviation = abs(tts_price - actual_price) / actual_price

        return {
            "pass": deviation < 0.0015,  # 0.15%
            "deviation": deviation,
            "issues": [] if deviation < 0.0015 else ["price_mismatch"]
        }

    except Exception:
        return {
            "pass": False,
            "deviation": 1.0,
            "issues": ["type_conversion_error"]
        }

def calculate_technical_conflict(tts_decision, ce_sentiment):
    decision = str(tts_decision).upper()
    sentiment = str(ce_sentiment).upper()

    is_buy = "BUY" in decision
    is_sell = "SELL" in decision
    is_hold = "HOLD" in decision

    is_bullish = "BULLISH" in sentiment
    is_bearish = "BEARISH" in sentiment
    is_neutral = "NEUTRAL" in sentiment

    # --- Opposite directional signals ---
    if (is_buy and is_bearish) or (is_sell and is_bullish):
        return "DIRECTIONAL_MISMATCH"

    # --- One active, one neutral ---
    if (is_buy or is_sell) and is_neutral:
        return "TECHNICAL_ONLY"

    if is_hold and (is_bullish or is_bearish):
        return "SENTIMENT_ONLY"

    # --- Both aligned ---
    if (is_buy and is_bullish) or (is_sell and is_bearish):
        return "ALIGNED"

    # --- Both inactive / neutral ---
    if is_hold and is_neutral:
        return "NO_SIGNAL"

    return "UNCLEAR"