import os
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
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

# =========================
# PROJECT ROOT FIXED
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "backtesting" / "news_cleaned"
PARQUET_FILE = DATA_PATH / "processed_news.parquet"

_news_df_cache = None


# =========================
# LOAD PARQUET ONCE
# =========================
def _load_news_df():
    global _news_df_cache

    if _news_df_cache is not None:
        return _news_df_cache

    if not PARQUET_FILE.exists():
        raise FileNotFoundError(f"Missing parquet file: {PARQUET_FILE}")

    df = pd.read_parquet(PARQUET_FILE)

    # ensure clean types
    df["date"] = df["date"].astype(str)
    df["currency"] = df["currency"].astype(str)
    df["title"] = df["title"].fillna("").astype(str)

    _news_df_cache = df
    return df


# =========================
# LABEL MAPPING
# =========================
def _map_label(label: str, score: float) -> float:
    label = (label or "").lower()

    if label == "positive":
        return float(score)
    if label == "negative":
        return -float(score)
    return 0.0


# =========================
# MAIN FUNCTION
# =========================
def get_news_sentiment(target_date: str, pair: str, backtest_mode: bool = False):

    df = _load_news_df()

    base = pair[:3].upper()
    quote = pair[3:].upper()

    # normalize date
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
    # FILTER (FAST VECTOR OPS)
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

    sentiments = []
    scores = []

    def classify(title: str):
        result = client.text_classification(title, model=MODEL_ID)
        top = result[0]
        return top.label, float(top.score)

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(classify, t): t for t in inference_rows}

        for f in as_completed(futures):
            try:
                label, score = f.result()
                sentiments.append(label)
                scores.append(score)
            except Exception as e:
                print(f"[CE ERROR] {futures[f]} -> {e}")

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

    return {
        "raw_vibe": raw_vibe,
        "mean_score": round(mean_score, 4),
        "sentiment_score": round(sentiment_score, 4),
        "article_count": article_count,
        "raw_article_count": raw_article_count,
        "titles": inference_rows,
        "debug_titles": inference_rows,  # NEW (for logging only)
        
    }