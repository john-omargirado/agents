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
        "overall_sentiment": "neutral",
        "raw_vibe": "neutral",
        "mean_score": 0.0,
        "sentiment_score": 0.0,
        "articles_analyzed": 0,
        "titles": [],
        "fake_mode": False
    }

    # =========================
    # VALIDATION GUARD
    # =========================
    if not sentiment_data:
        state["debug_log"].append(f"CE agent: No data for {target_date}")
        return {"ce_output": safe_default}

    if sentiment_data.get("articles_analyzed", 0) == 0:
        state["debug_log"].append(f"CE agent: Empty sentiment result for {target_date}")
        return {"ce_output": safe_default}

    # =========================
    # NORMALIZATION LAYER
    # =========================
    normalized = {
        "overall_sentiment": str(sentiment_data.get("overall_sentiment", "neutral")).lower(),
        "raw_vibe": str(sentiment_data.get("raw_vibe", "neutral")).lower(),
        "mean_score": float(sentiment_data.get("mean_score", 0.0)),
        "sentiment_score": float(sentiment_data.get("sentiment_score", 0.0)),
        "articles_analyzed": int(sentiment_data.get("articles_analyzed", 0)),
        "titles": sentiment_data.get("titles", []),
        "fake_mode": sentiment_data.get("fake_mode", False)
    }

    state["debug_log"].append(
        f"CE: {normalized['articles_analyzed']} articles | "
        f"{normalized['overall_sentiment']} sentiment"
    )

    return {"ce_output": normalized}