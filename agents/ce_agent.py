import json
import requests
import time
from state.trading_state import TradingState
from tools.ce_tools import get_news_sentiment
from utils.credentials import get_do_model_key


CE_URL = "https://inference.do-ai.run/v1/chat/completions"

_explanation_cache = {}


# =========================
# EXPLANATION ENGINE
# =========================
def call_ce_explanation(ce_data: dict):

    key = get_do_model_key()
    headers = {"Content-Type": "application/json"}

    if key:
        headers["Authorization"] = f"Bearer {key}"

    prompt = f"""/no_think
You are a sentiment analysis explanation engine.

Explain briefly:
- What the article sentiment distribution suggests
- Why confidence is {ce_data.get('confidence')}
- Whether the signal is reliable given article count
- Why the final sentiment is {ce_data.get('sentiment')}

Be concise and factual. No structured format.

INPUT:
{json.dumps(ce_data)}
"""

    payload = {
        "model": "alibaba-qwen3-32b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 700,
        "temperature": 0.2,
    }

    for attempt in range(3):
        try:
            resp = requests.post(CE_URL, headers=headers, json=payload, timeout=90)
            result = resp.json()

            msg = result.get("choices", [{}])[0].get("message", {})
            content = msg.get("content") or msg.get("reasoning_content")

            return str(content).strip() if content else "explanation_unavailable"

        except Exception as e:
            print(f"[CE EXPLANATION ERROR] {e}")
            time.sleep(5 * (attempt + 1))

    return "explanation_unavailable"


# =========================
# CE AGENT
# =========================
def ce_agent(state: TradingState):

    state["debug_log"].append("CE agent started")

    target_date = state.get("target_date")
    pair = state.get("currency_pair")
    backtest_mode = state.get("backtest_mode", False)

    sentiment_data = get_news_sentiment(target_date, pair, backtest_mode=backtest_mode)

    # =========================
    # DEBUG NEWS TRACE
    # =========================
    debug_titles = sentiment_data.get("debug_titles", [])

    print("\n[CE NEWS TRACE]")
    print(f"Date: {target_date}")
    print(f"Pair: {pair}")
    print(f"Articles used: {sentiment_data.get('article_count', 0)}")

    for i, title in enumerate(debug_titles):
        print(f"{i+1}. {title}")

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

    mean_score = float(sentiment_data["mean_score"])
    sentiment_score = float(sentiment_data["sentiment_score"])
    article_count = int(sentiment_data["article_count"])
    raw_article_count = int(sentiment_data["raw_article_count"])

    # =========================
    # CONFIDENCE
    # =========================
    if article_count >= 20:
        confidence = "HIGH"
    elif article_count >= 10:
        confidence = "MODERATE"
    else:
        confidence = "LOW"

    # =========================
    # SENTIMENT
    # =========================
    if sentiment_score > 0.05:
        sentiment = "BULLISH"
    elif sentiment_score < -0.05:
        sentiment = "BEARISH"
    else:
        sentiment = "NEUTRAL"

    normalized = {
        "sentiment": sentiment,
        "raw_vibe": sentiment_data["raw_vibe"],
        "mean_score": mean_score,
        "sentiment_score": sentiment_score,
        "article_count": article_count,
        "raw_article_count": raw_article_count,
        "confidence": confidence,
        "explanation": "pending"
    }

    cache_key = (
        sentiment,
        confidence,
        article_count,
        round(mean_score, 3),
        round(sentiment_score, 3)
    )

    if cache_key in _explanation_cache:
        explanation = _explanation_cache[cache_key]
    else:
        explanation = call_ce_explanation(normalized)
        _explanation_cache[cache_key] = explanation

    normalized["explanation"] = explanation

    print(f"\n[CE EXPLANATION]\n{explanation}")
    state["debug_log"].append("CE explanation generated")

    return {"ce_output": normalized}