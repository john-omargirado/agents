import os
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

MODEL_ID = "ProsusAI/finbert"

load_dotenv()
_hf_token = os.environ.get("HF_TOKEN")

if not _hf_token:
    raise EnvironmentError(
        "HF_TOKEN environment variable is not set."
    )

client = InferenceClient(
    provider="hf-inference",
    api_key=_hf_token,
)


def _map_label(label: str, score: float) -> float:
    """
    Convert FinBERT output into signed sentiment score.
    """
    label = (label or "").lower()

    if label == "positive":
        return float(score)
    if label == "negative":
        return -float(score)
    return 0.0


def get_news_sentiment(target_date: str, pair: str):
    """
    Backtest-safe sentiment tool.
    Deterministic, stable output format.
    """

    repo_root = Path(__file__).resolve().parents[2]
    data_path = repo_root / "data" / "calibration" / "news"

    if not data_path.is_dir():
        return {
            "overall_sentiment": "neutral",
            "raw_vibe": "neutral",
            "mean_score": 0.0,
            "articles_analyzed": 0,
            "titles": []
        }

    rows = []

    # =========================
    # LOAD DATA
    # =========================
    for file in os.listdir(data_path):
        if not file.endswith(".json"):
            continue

        with open(data_path / file, "r", encoding="utf-8") as f:
            data = json.load(f)

            for article in data.get("articles", []):
                try:
                    dt = datetime.strptime(article["seendate"], "%Y%m%dT%H%M%SZ")
                    if dt.strftime("%m/%d/%Y") == target_date:
                        title = article.get("title", "").strip()
                        if title:
                            rows.append(title)
                except:
                    continue

    if not rows:
        return {
            "overall_sentiment": "neutral",
            "raw_vibe": "neutral",
            "mean_score": 0.0,
            "articles_analyzed": 0,
            "titles": []
        }

    # remove duplicates for stability
    rows = list(set(rows))

    sentiments = []
    scores = []

    # =========================
    # INFERENCE
    # =========================
    for title in rows:
        try:
            results = client.text_classification(title, model=MODEL_ID)
            top = results[0]

            sentiments.append(top.label)
            scores.append(float(top.score))

        except Exception as e:
            print(f"[CE ERROR] FinBERT failed: {e}")
            continue

    n = len(scores)

    if n == 0:
        return {
            "overall_sentiment": "neutral",
            "raw_vibe": "neutral",
            "mean_score": 0.0,
            "articles_analyzed": 0,
            "titles": rows
        }

    # =========================
    # STABLE SCORING
    # =========================
    mapped_scores = [
        _map_label(label, score)
        for label, score in zip(sentiments, scores)
    ]

    mean_score = sum(scores) / n
    sentiment_score = sum(mapped_scores) / n

    # =========================
    # VIBE LOGIC
    # =========================
    if sentiment_score > 0.05:
        raw_vibe = "positive"
    elif sentiment_score < -0.05:
        raw_vibe = "negative"
    else:
        raw_vibe = "neutral"

    # =========================
    # CURRENCY ADJUSTMENT
    # =========================
    pair = pair.upper()

    if pair == "USDJPY":
        if raw_vibe == "negative":
            final = "bullish"
        elif raw_vibe == "positive":
            final = "bearish"
        else:
            final = "neutral"
    else:
        if raw_vibe == "positive":
            final = "bullish"
        elif raw_vibe == "negative":
            final = "bearish"
        else:
            final = "neutral"

    # =========================
    # OUTPUT (BACKTEST STABLE)
    # =========================
    return {
        "overall_sentiment": final,
        "raw_vibe": raw_vibe,
        "mean_score": round(float(mean_score), 4),
        "sentiment_score": round(float(sentiment_score), 4),
        "articles_analyzed": n,
        "titles": rows
    }