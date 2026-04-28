import pandas as pd
from typing import Optional
import numpy as np




def precompute_indicators(full_df: pd.DataFrame) -> pd.DataFrame:
    df = full_df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col])

    # EMA
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean()

    # RSI
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / (loss + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))

    # Bollinger
    df["sma_20"] = df["close"].rolling(window=20).mean()
    df["std_20"] = df["close"].rolling(window=20).std()
    df["upper_bb"] = df["sma_20"] + 2 * df["std_20"]
    df["lower_bb"] = df["sma_20"] - 2 * df["std_20"]

    df = df.set_index("timestamp")
    return df


def calculate_technical_indicators(
    full_df: pd.DataFrame,
    target_date: str,
    precomputed: Optional[pd.DataFrame] = None,
):
    target_dt = pd.to_datetime(target_date)

    df_slice = full_df[full_df["timestamp"] <= target_dt]
    row_count = len(df_slice)

    if precomputed is not None:
        df_pre = precomputed.loc[:target_dt]
    else:
        df_pre = None

    if df_slice.empty or row_count < 30:
        return None

    row: pd.Series

    # =========================
    # SAFE ROW EXTRACTION
    # =========================
    if precomputed is not None:
        available = precomputed.loc[:target_dt]

        if available.empty:
            return _calculate_from_scratch(df_slice, row_count, target_dt)

        row = available.iloc[-1]
    else:
        return _calculate_from_scratch(df_slice, row_count, target_dt)

    # =========================
    # FORCE SCALAR VALUES
    # =========================
    last_close = float(row["close"])
    ema_50 = float(row["ema_50"])
    ema_200 = float(row["ema_200"])
    rsi = float(row["rsi"])
    upper_bb = float(row["upper_bb"])
    lower_bb = float(row["lower_bb"])

    band_width = upper_bb - lower_bb
    ema_200_confidence = min(row_count / 400, 1.0)

    # =========================
    # STALENESS
    # =========================
    last_candle = df_slice["timestamp"].iloc[-1]
    days_stale = (target_dt - last_candle).days
    data_stale = days_stale > 5

    # =========================
    # TREND
    # =========================
    trend_diff = (ema_50 - ema_200) / ema_200

    if trend_diff > 0.001:
        trend = "BULLISH"
    elif trend_diff < -0.001:
        trend = "BEARISH"
    else:
        trend = "SIDEWAYS"

    trend_strength = min(abs(trend_diff) / 0.02, 1.0) * ema_200_confidence

    # =========================
    # RSI STRENGTH
    # =========================
    rsi_strength = 0.0
    if rsi > 60:
        rsi_strength = min((rsi - 60) / 40, 1.0)
    elif rsi < 40:
        rsi_strength = min((40 - rsi) / 40, 1.0)

    # =========================
    # BB SIGNAL
    # =========================
    bb_signal = "STABLE"
    bb_strength = 0.0

    if last_close >= upper_bb:
        bb_signal = "OVERBOUGHT"
        bb_strength = min(abs(last_close - upper_bb) / (band_width / 2 + 1e-9), 1.0)
    elif last_close <= lower_bb:
        bb_signal = "OVERSOLD"
        bb_strength = min(abs(lower_bb - last_close) / (band_width / 2 + 1e-9), 1.0)

    return {
        "price": round(last_close, 5),
        "trend": trend,
        "trend_strength": round(trend_strength, 4),
        "rsi": round(rsi, 2),
        "rsi_strength": round(rsi_strength, 4),
        "bb_signal": bb_signal,
        "bb_strength": round(bb_strength, 4),
        "ema_50": round(ema_50, 5),
        "ema_200": round(ema_200, 5),
        "rows_available": row_count,
        "ema_200_confidence": round(ema_200_confidence, 3),
        "ema_200_reliable": row_count >= 400,
        "ema_50_reliable": row_count >= 150,
        "data_stale": data_stale,
        "days_stale": int(days_stale),
    }
def _calculate_from_scratch(df: pd.DataFrame, row_count: int, target_dt):
    """Original logic as fallback."""
    for col in ['open', 'high', 'low', 'close']:
        df[col] = pd.to_numeric(df[col])

    ema_200_confidence = min(row_count / 400, 1.0)
    last_close = float(df['close'].iloc[-1])

    last_candle = df['timestamp'].iloc[-1]
    days_stale  = (target_dt - last_candle).days

    ema_50  = df['close'].ewm(span=50,  adjust=False).mean().iloc[-1]
    ema_200 = df['close'].ewm(span=200, adjust=False).mean().iloc[-1]
    trend_diff = (ema_50 - ema_200) / ema_200

    if trend_diff > 0.001:   trend = "BULLISH"
    elif trend_diff < -0.001: trend = "BEARISH"
    else:                     trend = "SIDEWAYS"

    trend_strength = min(abs(trend_diff) / 0.02, 1.0) * ema_200_confidence

    delta = df['close'].diff()
    gain  = delta.where(delta > 0, 0.0).ewm(alpha=1/14, adjust=False).mean()
    loss  = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/14, adjust=False).mean()
    rsi   = float(100 - (100 / (1 + gain.iloc[-1] / (loss.iloc[-1] + 1e-9))))

    rsi_strength = 0.0
    if rsi > 60:   rsi_strength = min((rsi - 60) / 40, 1.0)
    elif rsi < 40: rsi_strength = min((40 - rsi) / 40, 1.0)

    sma_20   = df['close'].rolling(window=20).mean()
    std_20   = df['close'].rolling(window=20).std()
    upper_bb = (sma_20 + 2 * std_20).iloc[-1]
    lower_bb = (sma_20 - 2 * std_20).iloc[-1]
    band_width = upper_bb - lower_bb

    bb_signal = "STABLE"; bb_strength = 0.0
    if last_close >= upper_bb:
        bb_signal   = "OVERBOUGHT"
        bb_strength = min(abs(last_close - upper_bb) / (band_width / 2 + 1e-9), 1.0)
    elif last_close <= lower_bb:
        bb_signal   = "OVERSOLD"
        bb_strength = min(abs(lower_bb - last_close) / (band_width / 2 + 1e-9), 1.0)

    return {
        "price": round(last_close, 5), "trend": trend,
        "trend_strength": round(trend_strength, 4), "rsi": round(rsi, 2),
        "rsi_strength": round(rsi_strength, 4), "bb_signal": bb_signal,
        "bb_strength": round(bb_strength, 4), "ema_50": round(ema_50, 5),
        "ema_200": round(ema_200, 5), "rows_available": row_count,
        "ema_200_confidence": round(ema_200_confidence, 3),
        "ema_200_reliable": row_count >= 400, "ema_50_reliable": row_count >= 150,
        "data_stale": days_stale > 5, "days_stale": int(days_stale),
    }