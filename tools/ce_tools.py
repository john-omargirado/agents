import os
import json
from datetime import datetime
from pathlib import Path
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

MODEL_ID = "ProsusAI/finbert"

load_dotenv()
_hf_token = os.environ.get("HF_TOKEN")

if not _hf_token:
    raise EnvironmentError("HF_TOKEN environment variable is not set.")

client = InferenceClient(
    provider="hf-inference",
    api_key=_hf_token,
)


def _map_label(label: str, score: float) -> float:
    label = (label or "").lower()

    if label == "positive":
        return float(score)
    if label == "negative":
        return -float(score)
    return 0.0


def get_news_sentiment(target_date: str, pair: str):

    repo_root = Path(__file__).resolve().parents[1]
    data_path = repo_root / "data" / "calibration" / "news"

    if not data_path.is_dir():
        return {
            "sentiment": "NEUTRAL",
            "raw_vibe": "NEUTRAL",
            "mean_score": 0.0,
            "sentiment_score": 0.0,
            "article_count": 0,
            "raw_article_count": 0,
            "titles": []
        }

    raw_rows = []

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
                            raw_rows.append(title)

                except:
                    continue

    # =========================
    # RAW COUNT (IMPORTANT)
    # =========================
    raw_article_count = len(raw_rows)

    if raw_article_count == 0:
        return {
            "sentiment": "NEUTRAL",
            "raw_vibe": "NEUTRAL",
            "mean_score": 0.0,
            "sentiment_score": 0.0,
            "article_count": 0,
            "raw_article_count": 0,
            "titles": []
        }

    # =========================
    # DEDUP FOR STABILITY
    # =========================
    inference_rows = list(set(raw_rows))

    sentiments = []
    scores = []

    # =========================
    # INFERENCE
    # =========================
    for title in inference_rows:
        try:
            results = client.text_classification(title, model=MODEL_ID)
            top = results[0]

            sentiments.append(top.label)
            scores.append(float(top.score))

        except Exception as e:
            print(f"[CE ERROR] FinBERT failed: {e}")
            continue

    article_count = len(scores)

    if article_count == 0:
        return {
            "sentiment": "NEUTRAL",
            "raw_vibe": "NEUTRAL",
            "mean_score": 0.0,
            "sentiment_score": 0.0,
            "article_count": 0,
            "raw_article_count": raw_article_count,
            "titles": inference_rows
        }

    # =========================
    # SCORING
    # =========================
    mapped_scores = [
        _map_label(label, score)
        for label, score in zip(sentiments, scores)
    ]

    mean_score = sum(scores) / article_count
    sentiment_score = sum(mapped_scores) / article_count

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
    # OUTPUT
    # =========================
    return {
        "sentiment": final.upper(),
        "raw_vibe": raw_vibe.upper(),
        "mean_score": round(float(mean_score), 4),
        "sentiment_score": round(float(sentiment_score), 4),

        "article_count": article_count,
        "raw_article_count": raw_article_count,

        "titles": inference_rows
    }