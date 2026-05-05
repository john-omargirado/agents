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
_calib_news_cache: dict = {}

DEBUG_CE = True
BATCH_SIZE = 16

pipe = pipeline(
    "text-classification",
    model=MODEL_ID,
    device=-1,   # CPU only
    truncation=True
)

# Map every supported currency to its calibration JSON filename.
# Add new currencies here as data files become available.
CURRENCY_FILENAME_MAP = {
    "USD": "usd_news_calibration.json",
    "JPY": "jpy_news_calibration.json",
    "EUR": "eur_news_calibration.json",
    "GBP": "gbp_news_calibration.json",
    "AUD": "aud_news_calibration.json",
    "CAD": "cad_news_calibration.json",
    "CHF": "chf_news_calibration.json",
    # Exotic / additional currencies (no calibration files yet — live mode only)
    "PHP": "php_news_calibration.json",
}

# Relevance terms per currency — articles must mention at least one.
# Extend this dict when adding new pairs.
CURRENCY_RELEVANCE_TERMS: dict[str, list[str]] = {
    "USD": ["dollar", "usd", "fed", "fomc", "federal reserve", "treasury",
            "us economy", "us gdp", "us inflation", "us jobs", "nonfarm",
            "powell", "interest rate", "us trade"],
    "JPY": ["yen", "jpy", "boj", "bank of japan", "kuroda", "ueda",
            "japan economy", "japan gdp", "japan inflation", "tankan",
            "tokyo cpi", "japanese trade", "yield curve control"],
    "EUR": ["euro", "eur", "ecb", "european central bank", "lagarde",
            "eurozone", "eu economy", "eu inflation", "eu gdp"],
    "GBP": ["pound", "gbp", "sterling", "bank of england", "boe",
            "uk economy", "uk inflation", "uk gdp", "uk trade"],
    "AUD": ["aussie", "aud", "rba", "reserve bank of australia",
            "australia economy", "australia inflation", "australia gdp"],
    "CAD": ["loonie", "cad", "bank of canada", "boc",
            "canada economy", "canada inflation", "canada gdp", "oil"],
    "CHF": ["franc", "chf", "snb", "swiss national bank",
            "switzerland economy", "swiss inflation"],
    # Exotic currencies
    "PHP": ["peso", "php", "bsp", "bangko sentral", "philippines economy",
            "philippine inflation", "philippine gdp", "manila"]
}


def _pair_currencies(pair: str) -> tuple[str, str]:
    """Return (base, quote) upper-cased from a 6-char pair string e.g. 'USDJPY'."""
    pair = pair.upper().replace("/", "").replace("_", "")
    return pair[:3], pair[3:]


def is_relevant(title: str, base: str, quote: str) -> bool:
    """
    Return True if the article title mentions at least one term
    relevant to either the base or quote currency.
    No noise-keyword blocking — only positive relevance check.
    """
    t = title.lower()
    base_terms  = CURRENCY_RELEVANCE_TERMS.get(base,  [])
    quote_terms = CURRENCY_RELEVANCE_TERMS.get(quote, [])
    all_terms   = base_terms + quote_terms
    return any(term in t for term in all_terms)


def log_time(label: str, start: float) -> None:
    if DEBUG_CE:
        print(f"[CE TIMER] {label}: {time.perf_counter() - start:.4f}s")


# =========================
# DATA LOADERS
# =========================

def _load_news_df(parquet_file: Path) -> pd.DataFrame:
    key = str(parquet_file)
    if key in _news_df_cache:
        return _news_df_cache[key]

    df = pd.read_parquet(parquet_file)
    df["date"]     = df["date"].astype(str)
    df["currency"] = df["currency"].astype(str).str.upper().str.strip()
    df["title"]    = df["title"].fillna("").astype(str)

    # Normalise language column if present — fill missing as English
    if "language" in df.columns:
        df["language"] = df["language"].fillna("English").astype(str)
    else:
        df["language"] = "English"

    if DEBUG_CE:
        unique_currencies = sorted(df["currency"].unique().tolist())
        print(f"[CE] Parquet loaded — unique currencies in data: {unique_currencies}")

    _news_df_cache[key] = df
    return df


def _load_calibration_news(pair: str) -> list[dict]:
    """Load and merge calibration JSON files for both currencies in the pair."""
    key = pair.upper()
    if key in _calib_news_cache:
        return _calib_news_cache[key]

    base, quote = _pair_currencies(pair)
    news_dir    = PROJECT_ROOT / "data" / "calibration" / "news"
    all_articles: list[dict] = []

    for currency in [base, quote]:
        filename = CURRENCY_FILENAME_MAP.get(currency)
        if not filename:
            print(f"[CE CALIB] No calibration file mapped for currency: {currency} — skipping")
            continue

        path = news_dir / filename
        if not path.exists():
            print(f"[CE CALIB] File not found: {path} — skipping")
            continue

        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        articles = raw.get("articles", [])
        for a in articles:
            all_articles.append({
                "title":    a.get("title", ""),
                "seendate": a.get("seendate", ""),
                "language": a.get("language", "English"),
                "currency": currency,
            })

        print(f"[CE CALIB] Loaded {len(articles)} articles for {currency}")

    _calib_news_cache[key] = all_articles
    return all_articles


def _parse_seendate(seendate: str) -> str:
    """Convert '20180122T143000Z' → '2018-01-22'. Returns '' on failure."""
    try:
        return datetime.strptime(seendate[:8], "%Y%m%d").strftime("%Y-%m-%d")
    except Exception:
        return ""


def _recency_weight(seendate_str: str, date_key: str) -> float:
    """1.0 same day → 0.3 floor at 3+ days old. Future articles → 0.0."""
    try:
        seen   = datetime.strptime(seendate_str[:8], "%Y%m%d")
        target = datetime.strptime(date_key, "%Y-%m-%d")
        delta  = (target - seen).days
        if delta < 0:
            return 0.0
        return max(0.3, 1.0 - (delta * 0.25))
    except Exception:
        return 0.5


# =========================
# FINBERT INFERENCE
# =========================

def batch_predict(texts: list[str]) -> tuple[list[str], list[float]]:
    results    = pipe(texts, batch_size=BATCH_SIZE)
    sentiments = [r["label"]        for r in results]
    scores     = [float(r["score"]) for r in results]
    return sentiments, scores


def _sentiment_to_score(
    label: str,
    confidence: float,
    currency: str,
    base: str,
    quote: str,
) -> float:
    """
    Convert a FinBERT label + confidence into a directional score relative to
    the pair's base currency.

    Convention: positive score = bullish for BASE (pair price goes UP).

    - Base  currency positive news → pair goes UP    → +score
    - Base  currency negative news → pair goes DOWN  → -score
    - Quote currency positive news → pair goes DOWN  → -score  (inverted)
    - Quote currency negative news → pair goes UP    → +score  (inverted)
    - Unknown currency             → score = 0.0     (safe discard)
    """
    label = (label or "").lower()

    if label == "positive":
        raw = float(confidence)
    elif label == "negative":
        raw = -float(confidence)
    else:
        raw = 0.0

    currency = (currency or "").upper().strip()

    if currency == base:
        return raw
    elif currency == quote:
        return -raw
    else:
        # Currency doesn't belong to this pair — neutral, don't bias either way
        if DEBUG_CE:
            print(f"[CE WARN] Unexpected currency '{currency}' for pair {base}/{quote} — scoring as 0.0")
        return 0.0


# =========================
# MAIN FUNCTION
# =========================

def get_news_sentiment(
    target_date: str,
    pair: str,
    backtest_mode: bool = False,
    skip_llm: bool = False,
) -> dict:

    total_start = time.perf_counter()

    base, quote = _pair_currencies(pair)

    # ── DATE PARSING ──────────────────────────────────────────
    try:
        dt       = datetime.strptime(target_date, "%m/%d/%Y")
        date_key = dt.strftime("%Y-%m-%d")
    except Exception:
        return {
            "raw_vibe": "NEUTRAL", "ce_score": 0.0, "ce_confidence": 0.0,
            "article_count": 0, "raw_article_count": 0, "titles": []
        }

    # ── DATA SOURCE ROUTING ───────────────────────────────────
    article_currencies: dict[str, str] = {}

    if backtest_mode:
        # ── BACKTEST: load from parquet ───────────────────────
        parquet_file = (
            PROJECT_ROOT / "data" / "backtesting" / "news_cleaned"
            / "processed_news.parquet"
        )
        df = _load_news_df(parquet_file)

        # Strictly filter to only articles tagged for base or quote currency.
        # The parquet's `currency` column (e.g. "USD", "JPY", "PHP") is the
        # authoritative tag — articles for unrelated currencies are excluded here.
        filtered = df[
            (df["date"]     == date_key) &
            (df["currency"].isin([base, quote])) &
            (df["language"] == "English")
        ]

        if DEBUG_CE:
            total_on_date = df[df["date"] == date_key].shape[0]
            kept          = filtered.shape[0]
            print(
                f"[CE BACKTEST] {date_key} | pair={pair} | "
                f"total articles on date={total_on_date} | "
                f"kept after currency+language filter={kept}"
            )

        raw_rows: list[str]      = []
        raw_weights: list[float] = []

        for _, row in filtered.iterrows():
            title    = str(row["title"]).strip()
            currency = str(row["currency"]).upper().strip()

            if not title:
                continue

            # Secondary relevance check — catches mislabelled articles
            if not is_relevant(title, base, quote):
                continue

            raw_rows.append(title)
            raw_weights.append(1.0)
            article_currencies[title] = currency   # e.g. "USD" or "JPY"

    else:
        # ── CALIBRATION: load from JSON files ─────────────────
        all_articles = _load_calibration_news(pair)

        try:
            target_dt = datetime.strptime(date_key, "%Y-%m-%d")
        except Exception:
            target_dt = None

        seen_weights: dict[str, float] = {}

        if target_dt:
            for a in all_articles:
                # English-only filter
                if a.get("language", "English") != "English":
                    continue

                title = a.get("title", "")
                if not isinstance(title, str) or not title.strip():
                    continue

                # Relevance filter
                if not is_relevant(title, base, quote):
                    continue

                seen_date_str = _parse_seendate(a["seendate"])
                if not seen_date_str:
                    continue

                try:
                    seen_dt = datetime.strptime(seen_date_str, "%Y-%m-%d")
                    delta   = (target_dt - seen_dt).days
                    if not (0 <= delta <= 1):    # same day or 1 day old
                        continue
                except Exception:
                    continue

                w = _recency_weight(a["seendate"], date_key)

                # Dedup: keep highest recency weight per title
                if title not in seen_weights or w > seen_weights[title]:
                    seen_weights[title]       = w
                    article_currencies[title] = a["currency"]

        raw_rows    = list(seen_weights.keys())
        raw_weights = [seen_weights[t] for t in raw_rows]

    # ── SHARED PATH ───────────────────────────────────────────

    raw_article_count = len(raw_rows)

    if raw_article_count == 0:
        return {
            "raw_vibe": "NEUTRAL", "ce_score": 0.0, "ce_confidence": 0.0,
            "article_count": 0, "raw_article_count": 0, "titles": []
        }

    # Dedup for backtest path (calibration already deduped above)
    if backtest_mode:
        seen_set: set[str]       = set()
        inference_rows: list[str]    = []
        deduped_weights: list[float] = []
        for t, w in zip(raw_rows, raw_weights):
            if t not in seen_set:
                seen_set.add(t)
                inference_rows.append(t)
                deduped_weights.append(w)
        raw_weights = deduped_weights
    else:
        inference_rows = raw_rows   # already deduped

    # ── FINBERT INFERENCE (cached) ────────────────────────────
    uncached = [t for t in inference_rows if t not in _finbert_cache]
    if uncached:
        try:
            sentiments, scores = batch_predict(uncached)
            for t, l, s in zip(uncached, sentiments, scores):
                _finbert_cache[t] = (l, s)
        except Exception as e:
            print(f"[CE BATCH ERROR] {e}")

    # ── BUILD SENTIMENT SCORES ────────────────────────────────
    sentiment_probs: list[float] = []
    valid_weights:   list[float] = []

    for t, w in zip(inference_rows, raw_weights):
        if t not in _finbert_cache:
            continue

        label, confidence = _finbert_cache[t]

        # Use parquet's currency tag directly — no guessing
        currency = article_currencies.get(t, "").upper().strip()

        score = _sentiment_to_score(label, confidence, currency, base, quote)

        # Discard articles whose currency didn't match base or quote
        # (score == 0.0 from unknown currency — still include in count but
        #  they contribute nothing, so we can safely skip to keep stats clean)
        if currency not in (base, quote):
            continue

        sentiment_probs.append(score)
        valid_weights.append(w)

    article_count = len(sentiment_probs)

    if article_count == 0:
        return {
            "raw_vibe": "NEUTRAL", "ce_score": 0.0, "ce_confidence": 0.0,
            "article_count": 0, "raw_article_count": raw_article_count,
            "titles": inference_rows
        }

    # ── WEIGHTED SCORING ──────────────────────────────────────
    weight_sum    = sum(valid_weights)
    ce_score      = sum(v * w for v, w in zip(sentiment_probs, valid_weights)) / weight_sum
    ce_confidence = min(article_count / 25, 1.0)

    raw_vibe = (
        "POSITIVE" if ce_score > 0.05 else
        "NEGATIVE" if ce_score < -0.05 else
        "NEUTRAL"
    )

    if DEBUG_CE:
        print(
            f"[CE] {target_date} | pair={pair} | "
            f"articles={article_count} (raw={raw_article_count}) | "
            f"ce_score={round(ce_score, 4)} | vibe={raw_vibe} | "
            f"elapsed={time.perf_counter() - total_start:.3f}s"
        )

    return {
        "raw_vibe":          raw_vibe,
        "ce_score":          round(ce_score, 4),
        "ce_confidence":     round(ce_confidence, 4),
        "article_count":     article_count,
        "raw_article_count": raw_article_count,
        "titles":            inference_rows,
    }