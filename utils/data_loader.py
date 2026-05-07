"""
Data loading utilities for backtesting and analysis.
Handles loading OHLCV data and news from parquet/JSON files.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta


DATA_DIR = Path(__file__).parent.parent / "data"
BACKTESTING_DIR = DATA_DIR / "backtesting"
FOREX_PAIRS_DIR = BACKTESTING_DIR / "forex_pairs"
NEWS_PARQUET = BACKTESTING_DIR / "news_cleaned" / "processed_news.parquet"

# Cache for loaded data
_ohlcv_cache: Dict[str, pd.DataFrame] = {}
_news_cache: Optional[pd.DataFrame] = None

def get_next_candles(pair: str, after_date: str, n: int = 5) -> list:
    """
    Return the next `n` OHLCV candles strictly after `after_date`.
    Compatible with the (df, is_stale) tuple returned by load_ohlcv_data().
    """
    df, _ = load_ohlcv_data(pair)          # ← unpack the tuple
    if df is None or df.empty:
        return []

    # 'timestamp' is already a datetime column (set in load_ohlcv_data)
    target = pd.Timestamp(after_date)
    future = df[df['timestamp'] > target].head(n)

    result = []
    for _, row in future.iterrows():
        result.append({
            "date":  str(row['timestamp'].date()),
            "open":  float(row['open']),
            "high":  float(row['high']),
            "low":   float(row['low']),
            "close": float(row['close']),
        })
    return result



def normalize_pair(pair: str) -> str:
    """Normalize pair format: EUR/USD -> EURUSD, EURUSD -> EURUSD"""
    return pair.replace("/", "").upper()


def get_forex_file(pair: str) -> Path:
    """Get path to forex pair JSON file"""
    normalized = normalize_pair(pair)
    return FOREX_PAIRS_DIR / f"{normalized}.json"


def load_ohlcv_data(pair: str, target_date: Optional[str] = None) -> Tuple[pd.DataFrame, bool]:
    """
    Load OHLCV data for a pair.
    
    Args:
        pair: Currency pair (e.g., 'EUR/USD' or 'EURUSD')
        target_date: Optional target date to check data availability
        
    Returns:
        Tuple of (DataFrame with OHLCV data, is_data_stale bool)
    """
    normalized_pair = normalize_pair(pair)
    cache_key = normalized_pair
    
    # Check cache
    if cache_key in _ohlcv_cache:
        df = _ohlcv_cache[cache_key].copy()
    else:
        # Load from JSON file
        file_path = get_forex_file(normalized_pair)
        
        if not file_path.exists():
            raise FileNotFoundError(f"OHLCV data not found for {pair} at {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data.get("data", []))
        
        if df.empty:
            raise ValueError(f"No OHLCV data in {file_path}")
        
        # Standardize column names
        df.columns = [c.lower() for c in df.columns]
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort and cache
        df = df.sort_values('timestamp').reset_index(drop=True)
        _ohlcv_cache[cache_key] = df.copy()
    
    # Check if data is stale (older than 1 day if target_date provided)
    is_stale = False
    if target_date and not df.empty:
        latest_date = df['timestamp'].max().date()
        target = pd.to_datetime(target_date).date()
        is_stale = (target - latest_date).days > 1
    
    return df, is_stale


def load_news_for_currency(currency: str, target_date: Optional[str] = None) -> List[Dict]:
    """
    Load news articles for a currency from parquet.
    
    Args:
        currency: Currency code (e.g., 'EUR', 'AUD', 'USD')
        target_date: Optional date filter (format: 'YYYY-MM-DD')
        
    Returns:
        List of news article dictionaries
    """
    global _news_cache
    
    # Load parquet if not cached
    if _news_cache is None:
        if not NEWS_PARQUET.exists():
            raise FileNotFoundError(f"News parquet file not found at {NEWS_PARQUET}")
        
        _news_cache = pd.read_parquet(NEWS_PARQUET)
    
    df = _news_cache.copy()
    
    # Filter by currency
    currency_upper = currency.upper()
    df = df[df['currency'] == currency_upper]
    
    # Filter by date if provided
    if target_date:
        target = pd.to_datetime(target_date).date()
        df = df[df['date'] == str(target)]
    
    # Convert to list of dicts
    articles = df[['date', 'title', 'domain', 'sourcecountry']].to_dict('records')
    
    return articles


def load_news_for_pair(pair: str, target_date: Optional[str] = None) -> Dict[str, List[Dict]]:
    """
    Load news for both currencies in a pair.
    
    Args:
        pair: Currency pair (e.g., 'EUR/USD')
        target_date: Optional date filter
        
    Returns:
        Dict with base and quote currency news
    """
    normalized = normalize_pair(pair)
    base = normalized[:3]  # EUR from EURUSD
    quote = normalized[3:]  # USD from EURUSD
    
    return {
        "base": load_news_for_currency(base, target_date),
        "quote": load_news_for_currency(quote, target_date),
    }


def get_available_dates_for_pair(pair: str) -> List[str]:
    """Get sorted list of available dates for a pair"""
    df, _ = load_ohlcv_data(pair)
    dates = df['timestamp'].dt.strftime('%Y-%m-%d').unique().tolist()
    return sorted(dates)


def clear_cache():
    """Clear all data caches"""
    global _ohlcv_cache, _news_cache
    _ohlcv_cache.clear()
    _news_cache = None

