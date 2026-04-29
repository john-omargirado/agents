import json
import requests
import time
from state.trading_state import TradingState
from tools.ce_tools import get_news_sentiment
from utils.credentials import get_do_model_key

CE_URL = "https://inference.do-ai.run/v1/chat/completions"

_explanation_cache = {}


def call_ce_explanation(ce_data: dict):
    key = get_do_model_key()
    headers = {"Content-Type": "application/json"}

    if key:
        headers["Authorization"] = f"Bearer {key}"

    prompt = f"""/no_think
Explain briefly:
- sentiment meaning
- confidence: {ce_data.get('confidence')}
- reliability based on article count
- final sentiment: {ce_data.get('sentiment')}

INPUT:
{json.dumps(ce_data)}
"""

    payload = {
        "model": "alibaba-qwen3-32b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 500,
        "temperature": 0.2,
    }

    for attempt in range(3):
        try:
            resp = requests.post(CE_URL, headers=headers, json=payload, timeout=60)
            result = resp.json()

            msg = result.get("choices", [{}])[0].get("message", {})

            content = (
                msg.get("content")
                or msg.get("reasoning_content")
                or "explanation_unavailable"
            )

            return str(content).strip()

        except Exception as e:
            print(f"[CE EXPLANATION ERROR] {e}")
            time.sleep(3 * (attempt + 1))

    return "explanation_unavailable"


def ce_agent(state: TradingState):

    state["debug_log"].append("CE agent started")

    target_date = state.get("target_date")
    pair = state.get("currency_pair")

    # ALWAYS SAFE BOOLEAN (important for backtesting stability)
    backtest_mode = bool(state.get("backtest_mode"))

    # =========================
    # CORE SENTIMENT PIPELINE
    # =========================
    sentiment_data = get_news_sentiment(
        target_date,
        pair,
        backtest_mode=backtest_mode
    )

    if not sentiment_data or sentiment_data.get("article_count", 0) == 0:
        return {
            "ce_output": {
                "sentiment": "NEUTRAL",
                "raw_vibe": "NEUTRAL",
                "mean_score": 0.0,
                "sentiment_score": 0.0,
                "article_count": 0,
                "raw_article_count": 0,
                "confidence": "LOW",
                "explanation": "no_data"
            }
        }

    mean_score = sentiment_data.get("mean_score", 0.0)
    sentiment_score = sentiment_data.get("sentiment_score", 0.0)
    article_count = sentiment_data.get("article_count", 0)
    raw_article_count = sentiment_data.get("raw_article_count", 0)

    # =========================
    # CLASSIFICATION
    # =========================
    confidence = (
        "HIGH" if article_count >= 25 else
        "MODERATE" if article_count >= 15 else
        "LOW"
    )

    sentiment = (
        "BULLISH" if sentiment_score > 0.05 else
        "BEARISH" if sentiment_score < -0.05 else
        "NEUTRAL"
    )

    normalized = {
        "sentiment": sentiment,
        "raw_vibe": sentiment_data.get("raw_vibe", "NEUTRAL"),
        "mean_score": mean_score,
        "sentiment_score": sentiment_score,
        "article_count": article_count,
        "raw_article_count": raw_article_count,
        "confidence": confidence,
        "explanation": "pending"
    }

    # =========================
    # EXPLANATION (LIVE ONLY)
    # =========================
    if not backtest_mode:

        cache_key = (
            sentiment,
            confidence,
            article_count,
            round(sentiment_score, 3)
        )

        if cache_key in _explanation_cache:
            explanation = _explanation_cache[cache_key]
        else:
            explanation = call_ce_explanation(normalized)
            _explanation_cache[cache_key] = explanation

        normalized["explanation"] = explanation

    else:
        normalized["explanation"] = "skipped_backtest"

    return {"ce_output": normalized}