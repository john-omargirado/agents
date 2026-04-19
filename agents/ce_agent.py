from state.trading_state import TradingState
from tools.ce_tools import get_news_sentiment


def ce_agent(state: TradingState):
    state["debug_log"].append("CE agent: Starting sentiment audit via tools")

    target_date = state.get("target_date")
    pair = state.get("currency_pair")

    sentiment_data = get_news_sentiment(target_date, pair)

    # =========================
    # SAFE DEFAULT STRUCTURE
    # =========================
    safe_default = {
        "sentiment": "NEUTRAL",
        "raw_vibe": "NEUTRAL",
        "mean_score": 0.0,
        "sentiment_score": 0.0,
        "article_count": 0,
        "confidence": "LOW",
        "error": None
    }

    # =========================
    # VALIDATION GUARD
    # =========================
    if not sentiment_data:
        state["debug_log"].append(f"CE agent: No data for {target_date}")
        return {"ce_output": safe_default}

    if sentiment_data.get("article_count", 0) == 0:
        state["debug_log"].append(f"CE agent: Empty sentiment result for {target_date}")
        return {"ce_output": safe_default}


    # =========================
    # NORMALIZATION LAYER
    # =========================
    mean_score    = float(sentiment_data.get("mean_score", 0.0))
    article_count = int(sentiment_data.get("article_count", 0))

    if article_count >= 20 and abs(mean_score) >= 0.5:
        confidence = "HIGH"
    elif article_count >= 10 or abs(mean_score) >= 0.2:
        confidence = "MODERATE"
    else:
        confidence = "LOW"

    normalized = {
        "sentiment":       sentiment_data.get("sentiment", "NEUTRAL"),
        "raw_vibe":        sentiment_data.get("raw_vibe", "NEUTRAL"),
        "mean_score":      mean_score,
        "sentiment_score": float(sentiment_data.get("sentiment_score", 0.0)),
        "article_count":   article_count,
        "confidence":      confidence,
        "error":           None
    }
    

    

    state["debug_log"].append(
        f"CE: {normalized['article_count']} articles | "
        f"{normalized['sentiment']} sentiment | "
        f"confidence={normalized['confidence']}"
    )

    return {"ce_output": normalized}