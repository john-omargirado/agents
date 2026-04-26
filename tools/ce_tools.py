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


def get_news_sentiment(target_date: str, pair: str, backtest_mode: bool = False):
    repo_root = Path(__file__).resolve().parents[1]

    if backtest_mode:
        data_path = repo_root / "data" / "backtesting" / "news"
        # Split USDJPY → ["usd", "jpy"]
        base = pair[:3].lower()
        quote = pair[3:].lower()
        target_files = [f"{base}_news_backtesting.json", f"{quote}_news_backtesting.json"]
    else:
        data_path = repo_root / "data" / "calibration" / "news"
        target_files = [f for f in os.listdir(data_path) if f.endswith(".json")] if data_path.is_dir() else []

    if not data_path.is_dir():
        return {"raw_vibe": "NEUTRAL", "mean_score": 0.0, "sentiment_score": 0.0,
                "article_count": 0, "raw_article_count": 0, "titles": []}

    raw_rows = []

    for file in target_files:
        filepath = data_path / file
        if not filepath.exists():
            continue

        with open(filepath, "r", encoding="utf-8") as f:
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

    raw_article_count = len(raw_rows)

    if raw_article_count == 0:
        return {
            "raw_vibe": "NEUTRAL",
            "mean_score": 0.0,
            "sentiment_score": 0.0,
            "article_count": 0,
            "raw_article_count": 0,
            "titles": []
        }

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
    # RAW VIBE (PURE SENTIMENT ONLY)
    # =========================
    if sentiment_score > 0.05:
        raw_vibe = "POSITIVE"
    elif sentiment_score < -0.05:
        raw_vibe = "NEGATIVE"
    else:
        raw_vibe = "NEUTRAL"

    # =========================
    # OUTPUT (NO FX LOGIC)
    # =========================
    return {
        "raw_vibe": raw_vibe,
        "mean_score": round(float(mean_score), 4),
        "sentiment_score": round(float(sentiment_score), 4),
        "article_count": article_count,
        "raw_article_count": raw_article_count,
        "titles": inference_rows
    }