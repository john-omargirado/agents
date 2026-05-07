import json
import requests
import time
import os
from state.trading_state import TradingState
from tools.ce_tools import get_news_sentiment
from utils.credentials import get_do_model_key

CE_URL = "https://inference.do-ai.run/v1/chat/completions"
_explanation_cache = {}

print(f"[CE AGENT MODULE] Loaded from: {os.path.abspath(__file__)}")

def log(stage: str, start: float):
    elapsed = (time.perf_counter() - start) * 1000
    print(f"[CE TIMER] {stage}: {elapsed:.2f} ms")

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

    attempt = 0
    while True:
        attempt += 1
        try:
            t0 = time.perf_counter()
            resp = requests.post(CE_URL, headers=headers, json=payload, timeout=90)
            log("LLM REQUEST", t0)
            if resp.status_code == 429:
                print(f"[CE] Rate limited attempt {attempt}, waiting 15s...")
                time.sleep(15)
                continue
            if resp.status_code != 200:
                print(f"[CE ERROR] HTTP {resp.status_code}, retrying...")
                time.sleep(10)
                continue
            result = resp.json()
            message = result.get("choices", [{}])[0].get("message", {})
            content = message.get("content") or message.get("reasoning_content")
            if not content:
                print(f"[CE ERROR] Empty response, retrying...")
                time.sleep(10)
                continue
            return str(content).strip()
        except Exception as e:
            print(f"[CE ERROR] {e} — retrying...")
            time.sleep(10)


def ce_agent(state: TradingState):
    state["debug_log"].append("CE agent started")

    target_date = state.get("target_date")
    pair        = state.get("currency_pair")

    live_mode     = bool(state.get("live_mode", True))
    backtest_mode = not live_mode
    skip_llm      = bool(state.get("skip_llm", False))

    print(f"[CE AGENT] live_mode={live_mode} | backtest_mode={backtest_mode} | date={target_date} | pair={pair}")

    sentiment_data = get_news_sentiment(
        target_date,
        pair,
        backtest_mode=backtest_mode,
        live_mode=live_mode,
    )

    print(f"[CE AGENT] sentiment_data={sentiment_data}")

    if not sentiment_data or sentiment_data.get("article_count", 0) == 0:
        return {
            "ce_output": {
                "sentiment": "NEUTRAL",
                "raw_vibe": "NEUTRAL",
                "ce_score": 0.0,
                "ce_confidence": 0.0,
                "article_count": 0,
                "raw_article_count": sentiment_data.get("raw_article_count", 0) if sentiment_data else 0,
                "confidence": "LOW",
                "explanation": "no_data",
                "error": "no_data"
            }
        }

    article_count     = sentiment_data["article_count"]
    raw_article_count = sentiment_data["raw_article_count"]
    ce_final          = sentiment_data["ce_score"]
    ce_confidence     = sentiment_data["ce_confidence"]

    confidence = (
        "HIGH" if article_count >= 25 else
        "MODERATE" if article_count >= 15 else
        "LOW"
    )

    sentiment = (
        "BULLISH" if ce_final > 0.05 else
        "BEARISH" if ce_final < -0.05 else
        "NEUTRAL"
    )

    normalized = {
        "sentiment": sentiment,
        "raw_vibe": sentiment_data.get("raw_vibe", "NEUTRAL"),
        "ce_score": ce_final,
        "ce_confidence": ce_confidence,
        "article_count": article_count,
        "raw_article_count": raw_article_count,
        "confidence": confidence,
        "explanation": "pending",
        "error": None,
    }

    if live_mode and not skip_llm:
        cache_key = (ce_final, confidence, article_count)
        if cache_key in _explanation_cache:
            explanation = _explanation_cache[cache_key]
        else:
            explanation = call_ce_explanation(normalized)
            _explanation_cache[cache_key] = explanation
        normalized["explanation"] = explanation
    else:
        normalized["explanation"] = "skipped"

    return {"ce_output": normalized}