import json
import requests
import time
from state.trading_state import TradingState
from tools.ce_tools import get_news_sentiment
from utils.credentials import get_do_model_key


CE_URL = "https://inference.do-ai.run/v1/chat/completions"


def call_ce_explanation(ce_data: dict) -> str:
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

    data = {
        "model": "alibaba-qwen3-32b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
        "temperature": 0.2,
    }

    max_retries = 3
    backoff = 5

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(CE_URL, headers=headers, json=data, timeout=90)
            print(f"[CE EXPLANATION HTTP] status={resp.status_code}")
            result = resp.json()

            message = result.get("choices", [{}])[0].get("message", {})
            content = message.get("content") or message.get("reasoning_content")

            return str(content).strip() if content else "explanation_unavailable"

        except Exception as e:
            print(f"[CE EXPLANATION ERROR] Attempt {attempt}/{max_retries}: {e}")
            if attempt < max_retries:
                print(f"[CE EXPLANATION] Retrying in {backoff}s...")
                time.sleep(backoff)
                backoff *= 2  # 5 -> 10 -> 20

    return "explanation_unavailable"


def ce_agent(state: TradingState):
    state["debug_log"].append("CE agent: Starting sentiment audit via tools")

    target_date = state.get("target_date")
    pair = state.get("currency_pair")

    backtest_mode = state.get("backtest_mode", False)
    sentiment_data = get_news_sentiment(target_date, pair, backtest_mode=backtest_mode)

    # =========================
    # SAFE DEFAULT
    # =========================
    safe_default = {
        "sentiment": "NEUTRAL",
        "raw_vibe": "NEUTRAL",
        "mean_score": 0.0,
        "sentiment_score": 0.0,
        "article_count": 0,
        "raw_article_count": 0,
        "confidence": "LOW",
        "error": None,
        "explanation": "no_data"
    }

    if not sentiment_data:
        return {"ce_output": safe_default}

    if sentiment_data.get("article_count", 0) == 0:
        return {"ce_output": safe_default}

    # =========================
    # NORMALIZATION
    # =========================
    mean_score = float(sentiment_data.get("mean_score", 0.0))
    sentiment_score = float(sentiment_data.get("sentiment_score", 0.0))
    article_count = int(sentiment_data.get("article_count", 0))
    raw_article_count = int(sentiment_data.get("raw_article_count", article_count))

    # =========================
    # CONFIDENCE LOGIC
    # =========================
    if article_count >= 20 and abs(mean_score) >= 0.5:
        confidence = "HIGH"
    elif article_count >= 10 or abs(mean_score) >= 0.2:
        confidence = "MODERATE"
    else:
        confidence = "LOW"

    # =========================
    # MARKET SENTIMENT (NOW HERE, NOT IN TOOL)
    # =========================
    if sentiment_score > 0.05:
        sentiment = "BULLISH"
    elif sentiment_score < -0.05:
        sentiment = "BEARISH"
    else:
        sentiment = "NEUTRAL"

    normalized = {
        "sentiment": sentiment,
        "raw_vibe": sentiment_data.get("raw_vibe", "NEUTRAL"),
        "mean_score": mean_score,
        "sentiment_score": sentiment_score,
        "article_count": article_count,
        "raw_article_count": raw_article_count,
        "confidence": confidence,
        "error": None,
        "explanation": "pending"
    }

    state["debug_log"].append(
        f"CE: {article_count} articles | "
        f"{sentiment} | confidence={confidence}"
    )

    normalized["explanation"] = call_ce_explanation(normalized)

    print(f"\n[CE EXPLANATION]\n{normalized['explanation']}")
    state["debug_log"].append("CE explanation generated")

    return {"ce_output": normalized}