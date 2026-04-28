import json
from pathlib import Path
from datetime import datetime
import pandas as pd

# =========================
# PROJECT ROOT
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# =========================
# DATA PATHS
# =========================
DATA_PATH = PROJECT_ROOT / "data" / "backtesting" / "news"
OUTPUT_FILE = DATA_PATH / "processed_news.parquet"

CURRENCIES = {
    "USD": "usd_news_backtesting.json",
    "JPY": "jpy_news_backtesting.json",
    "EUR": "euro_news_backtesting.json",   # FIXED (was euro_news_backtesting.json)
    "GBP": "gbp_news_backtesting.json",
    "AUD": "aud_news_backtesting.json",
    "CAD": "cad_news_backtesting.json",
    "CHF": "chf_news_backtesting.json",
    "PHP": "php_news_backtesting.json",
}


# =========================
# DATE PARSER
# =========================
def parse_date(seendate: str):
    return datetime.strptime(seendate, "%Y%m%dT%H%M%SZ")


# =========================
# LOAD ARTICLES (ROBUST)
# =========================
def load_articles(currency: str, filename: str):
    file_path = DATA_PATH / filename

    print(f"[INFO] Checking {file_path}")

    if not file_path.exists():
        print(f"[WARN] Missing file: {file_path}")
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # =========================
    # FIX: handle multiple formats
    # =========================
    if isinstance(data, dict) and "articles" in data:
        data = data["articles"]

    if not isinstance(data, list):
        print(f"[ERROR] Invalid format in {file_path}")
        return []

    rows = []

    for a in data:
        try:
            seendate = a.get("seendate")
            if not seendate:
                continue

            dt = parse_date(seendate)

            # skip weekends
            if dt.weekday() >= 5:
                continue

            rows.append({
                "date": dt.date().isoformat(),
                "datetime": dt,
                "currency": currency,
                "title": (a.get("title") or "").strip(),
                "domain": a.get("domain", ""),
                "sourcecountry": a.get("sourcecountry", "")
            })

        except Exception:
            continue

    print(f"[OK] {currency}: {len(rows)} articles")
    return rows


# =========================
# BUILD DATASET
# =========================
def build_dataset():
    all_rows = []

    for currency, file in CURRENCIES.items():
        print(f"\nProcessing {currency}...")
        all_rows.extend(load_articles(currency, file))

    print(f"\nTOTAL ROWS COLLECTED: {len(all_rows)}")

    if not all_rows:
        print("No data found")
        return

    df = pd.DataFrame(all_rows)

    df = df.sort_values(["date", "currency"])

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_FILE, index=False)

    print("\nDONE")
    print(f"Saved: {OUTPUT_FILE}")
    print(f"Rows: {len(df)}")


if __name__ == "__main__":
    build_dataset()