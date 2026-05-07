# run this as a quick script: check_parquet.py
from pathlib import Path
import pandas as pd

PARQUET = Path("data/backtesting/news_cleaned/processed_news.parquet")
df = pd.read_parquet(PARQUET)

print("=== COLUMNS ===")
print(df.columns.tolist())

print("\n=== CURRENCIES IN DATA ===")
print(df["currency"].unique())

print("\n=== DATE RANGE ===")
print(f"Min: {df['date'].min()}")
print(f"Max: {df['date'].max()}")

print("\n=== 2023-04-02 EUR/USD ===")
print(len(df[(df["date"] == "2023-04-02") & (df["currency"].isin(["EUR", "USD"]))]))

print("\n=== LATEST DATE WITH EUR/USD DATA ===")
eur_usd = df[df["currency"].isin(["EUR", "USD"])]
print(eur_usd["date"].max())

print("\n=== EUR/USD RELEVANT ROWS ===")
eur_usd = df[df["currency"].isin(["EUR", "USD"])]
print(f"Total rows: {len(eur_usd)}")
print(eur_usd["date"].value_counts().head(10))