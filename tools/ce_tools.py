import os
from pathlib import Path
from datetime import datetime
import time
import pandas as pd
from transformers import pipeline
from dotenv import load_dotenv

MODEL_ID = "ProsusAI/finbert"

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "backtesting" / "news_cleaned"
PARQUET_FILE = DATA_PATH / "processed_news.parquet"

_news_df_cache: dict = {}
_finbert_cache = {}

DEBUG_CE = True

# CPU optimization (i5 11th gen sweet spot)
BATCH_SIZE = 16

# IMPORTANT: CPU only
pipe = pipeline(
    "text-classification",
    model=MODEL_ID,
    device=-1,  # CPU
    truncation=True
)


def log_time(label: str, start: float):
    if DEBUG_CE:
        print(f"[CE TIMER] {label}: {time.perf_counter() - start:.4f}s")


def _load_news_df(parquet_file: Path):
    key = str(parquet_file)
    if key in _news_df_cache:
        return _news_df_cache[key]

    df = pd.read_parquet(parquet_file)
    df["date"]     = df["date"].astype(str)
    df["currency"] = df["currency"].astype(str)
    df["title"]    = df["title"].fillna("").astype(str)

    _news_df_cache[key] = df
    return df


def _map_label(label: str, score: float) -> float:
    label = (label or "").lower()

    if label == "positive":
        return float(score)
    if label == "negative":
        return -float(score)
    return 0.0


# =========================
# SAFE BATCH PREDICTION
# =========================
def batch_predict(texts: list[str]):
    results = pipe(texts, batch_size=BATCH_SIZE)

    sentiments = []
    scores = []

    for r in results:
        sentiments.append(r["label"])
        scores.append(float(r["score"]))

    return sentiments, scores


# =========================
# MAIN FUNCTION
# =========================
def get_news_sentiment(target_date: str, pair: str, backtest_mode: bool = False):

    total_start = time.perf_counter()

    if backtest_mode:
        parquet_file = PROJECT_ROOT / "data" / "backtesting" / "news_cleaned" / "processed_news.parquet"
    else:
        parquet_file = PROJECT_ROOT / "data" / "calibration" / "news_cleaned" / "processed_news.parquet"

    df = _load_news_df(parquet_file)

    base = pair[:3].upper()
    quote = pair[3:].upper()

    try:
        dt = datetime.strptime(target_date, "%m/%d/%Y")
        date_key = dt.strftime("%Y-%m-%d")
    except:
        return {
            "raw_vibe": "NEUTRAL",
            "mean_score": 0.0,
            "sentiment_score": 0.0,
            "article_count": 0,
            "raw_article_count": 0,
            "titles": []
        }

    # =========================
    # FILTER
    # =========================
    filtered = df[
        (df["date"] == date_key) &
        (df["currency"].isin([base, quote]))
    ]

    raw_rows = filtered["title"].tolist()
    raw_rows = [t for t in raw_rows if isinstance(t, str) and t.strip()]

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

    # =========================
    # CACHE SPLIT
    # =========================
    uncached = [t for t in inference_rows if t not in _finbert_cache]

    # =========================
    # BATCH INFERENCE (FIXED)
    # =========================
    if uncached:
        try:
            sentiments, scores = batch_predict(uncached)

            for t, l, s in zip(uncached, sentiments, scores):
                _finbert_cache[t] = (l, s)

        except Exception as e:
            print(f"[CE BATCH ERROR] {e}")

    # =========================
    # BUILD FINAL OUTPUT
    # =========================
    sentiments = []
    scores = []

    for t in inference_rows:
        if t in _finbert_cache:
            l, s = _finbert_cache[t]
            sentiments.append(l)
            scores.append(s)

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
        _map_label(l, s)
        for l, s in zip(sentiments, scores)
    ]

    mean_score = sum(scores) / article_count
    sentiment_score = sum(mapped_scores) / article_count

    if sentiment_score > 0.05:
        raw_vibe = "POSITIVE"
    elif sentiment_score < -0.05:
        raw_vibe = "NEGATIVE"
    else:
        raw_vibe = "NEUTRAL"

    if DEBUG_CE:
        print(f"[CE TOTAL TIME] {time.perf_counter() - total_start:.4f}s | articles={article_count}")

    return {
        "raw_vibe": raw_vibe,
        "mean_score": round(mean_score, 4),
        "sentiment_score": round(sentiment_score, 4),
        "article_count": article_count,
        "raw_article_count": raw_article_count,
        "titles": inference_rows,
        "debug_titles": inference_rows[:5],
    }