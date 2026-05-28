import os
from pathlib import Path
from datetime import datetime
import time
import pandas as pd
from transformers import pipeline
from dotenv import load_dotenv
import json

MODEL_ID = "ProsusAI/finbert"

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "backtesting" / "news_cleaned"
PARQUET_FILE = DATA_PATH / "processed_news.parquet"

_news_df_cache: dict = {}
_finbert_cache: dict = {}

DEBUG_CE = True
BATCH_SIZE = 16

pipe = pipeline(
    "text-classification",
    model=MODEL_ID,
    device=-1,
    truncation=True
)
print(f"[CE MODULE] Loaded from: {os.path.abspath(__file__)}")


# =========================
# DATE NORMALIZER
# =========================
def _normalize_date(date_str: str) -> str | None:
    if not date_str:
        return None

    formats = [
        "%Y-%m-%d",
        "%m-%d-%Y",
        "%m/%d/%Y",
        "%d-%m-%Y"
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except Exception:
            continue

    return None


# =========================
# CONFIG
# =========================
CURRENCY_RELEVANCE_TERMS: dict[str, list[str]] = {
    "USD": ["dollar", "usd", "fed", "fomc", "federal reserve"],
    "JPY": ["yen", "jpy", "boj", "bank of japan"],
    "EUR": ["euro", "eur", "ecb"],
    "GBP": ["pound", "gbp", "boe"],
    "AUD": ["aussie", "aud", "rba"],
    "CAD": ["loonie", "cad", "boc", "oil"],
    "CHF": ["franc", "chf", "snb"],
    "PHP": ["peso", "php", "bsp"]
}


def _pair_currencies(pair: str) -> tuple[str, str]:
    pair = pair.upper().replace("/", "").replace("_", "")
    return pair[:3], pair[3:]


def is_relevant(title: str, base: str, quote: str) -> bool:
    t = title.lower()
    terms = CURRENCY_RELEVANCE_TERMS.get(base, []) + CURRENCY_RELEVANCE_TERMS.get(quote, [])
    return any(term in t for term in terms)


def _load_news_df(parquet_file: Path) -> pd.DataFrame:
    key = str(parquet_file)
    if key in _news_df_cache:
        return _news_df_cache[key]

    df = pd.read_parquet(parquet_file)
    df["date"] = df["date"].astype(str)
    df["currency"] = df["currency"].astype(str).str.upper().str.strip()
    df["title"] = df["title"].fillna("").astype(str)

    if "language" in df.columns:
        df["language"] = df["language"].fillna("English").astype(str)
    else:
        df["language"] = "English"

    if DEBUG_CE:
        print("[CE] Loaded parquet")

    _news_df_cache[key] = df
    return df


def batch_predict(texts: list[str]):
    results = pipe(texts, batch_size=BATCH_SIZE)
    return [r["label"] for r in results], [float(r["score"]) for r in results]


def _sentiment_to_score(label, confidence, currency, base, quote):
    label = label.lower()

    if label == "positive":
        raw = confidence
    elif label == "negative":
        raw = -confidence
    else:
        raw = 0.0

    if currency == base:
        return raw
    elif currency == quote:
        return -raw
    return 0.0


def get_news_sentiment(
    target_date: str,
    pair: str,
    backtest_mode=False,
    live_mode=False,
    skip_llm=False,
) -> dict:

    print(f"[CE DEBUG] called | live_mode={live_mode} | date={target_date} | pair={pair}")

    start = time.perf_counter()
    base, quote = _pair_currencies(pair)

    df = _load_news_df(PARQUET_FILE)
    pair_df = df[df["currency"].isin([base, quote])]

    if len(pair_df) == 0:
        print(f"[CE] No data at all for {base}/{quote}")
        return {
            "raw_vibe": "NEUTRAL",
            "ce_score": 0.0,
            "ce_confidence": 0.0,
            "article_count": 0,
            "raw_article_count": 0,
        }

    # ── RESOLVE DATE ──────────────────────────────────────────────────
    date_key = _normalize_date(target_date)

    if not date_key:
        # unparseable date → use latest
        date_key = pair_df["date"].max()
        print(f"[CE] Unparseable date '{target_date}' — fallback to latest: {date_key}")
    else:
        # check if data exists for this exact date
        has_data = len(pair_df[pair_df["date"] == date_key]) > 0

        if not has_data:
            # no data for requested date → find nearest earlier date
            earlier = pair_df[pair_df["date"] <= date_key]
            if len(earlier) > 0:
                date_key = earlier["date"].max()
                print(f"[CE] No data for requested date — using nearest earlier: {date_key}")
            else:
                date_key = pair_df["date"].max()
                print(f"[CE] No earlier data found — fallback to latest: {date_key}")
        else:
            print(f"[CE] Using exact date = {date_key}")

    # If the fallback date is still outside the available range, ensure we use a real date
    if date_key not in set(pair_df["date"].astype(str).tolist()):
        available_dates = pair_df["date"].dropna().astype(str)
        if len(available_dates) > 0:
            date_key = available_dates.max()
            print(f"[CE] Final fallback — using latest available date: {date_key}")
    # ─────────────────────────────────────────────────────────────────

    filtered = df[
        (df["date"] == date_key) &
        (df["currency"].isin([base, quote])) &
        (df["language"] == "English")
    ]

    raw_article_count = len(filtered)
    print(f"[CE] Filtered rows for {date_key}: {raw_article_count}")

    raw_rows = []
    article_currency = {}

    for _, row in filtered.iterrows():
        title = row["title"]
        currency = row["currency"]

        if not is_relevant(title, base, quote):
            continue

        raw_rows.append(title)
        article_currency[title] = currency

    print(f"[CE] Relevant articles after filter: {len(raw_rows)} / {raw_article_count}")

    if not raw_rows:
        return {
            "raw_vibe": "NEUTRAL",
            "ce_score": 0.0,
            "ce_confidence": 0.0,
            "article_count": 0,
            "raw_article_count": raw_article_count,
        }

    uncached = [t for t in raw_rows if t not in _finbert_cache]

    if uncached:
        labels, scores = batch_predict(uncached)
        for t, l, s in zip(uncached, labels, scores):
            _finbert_cache[t] = (l, s)

    scores = []
    for t in raw_rows:
        if t not in _finbert_cache:
            continue
        label, conf = _finbert_cache[t]
        currency = article_currency[t]
        scores.append(_sentiment_to_score(label, conf, currency, base, quote))

    article_count = len(scores)

    if article_count == 0:
        return {
            "raw_vibe": "NEUTRAL",
            "ce_score": 0.0,
            "ce_confidence": 0.0,
            "article_count": 0,
            "raw_article_count": raw_article_count,
        }

    ce_score = sum(scores) / article_count
    vibe = (
        "POSITIVE" if ce_score > 0.05 else
        "NEGATIVE" if ce_score < -0.05 else
        "NEUTRAL"
    )

    print(f"[CE] {date_key} | score={round(ce_score,4)} | articles={article_count} | raw={raw_article_count}")

    return {
        "raw_vibe": vibe,
        "ce_score": round(ce_score, 4),
        "ce_confidence": round(min(len(scores) / 25, 1.0), 4),
        "article_count": article_count,
        "raw_article_count": raw_article_count,
        "titles": raw_rows,
    }