import pandas as pd
from typing import Optional
import numpy as np


import pandas as pd
from typing import Optional

def precompute_indicators(full_df: pd.DataFrame) -> pd.DataFrame:
    df = full_df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col])

    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / (loss + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))

    df["sma_20"] = df["close"].rolling(20).mean()
    df["std_20"] = df["close"].rolling(20).std()
    df["upper_bb"] = df["sma_20"] + 2 * df["std_20"]
    df["lower_bb"] = df["sma_20"] - 2 * df["std_20"]

    df = df.set_index("timestamp")

    return df


def calculate_technical_indicators(full_df, target_date, precomputed=None):

    target_dt = pd.to_datetime(target_date)

    if precomputed is None:
        raise ValueError("Precomputed required for performance")

    try:
        row = precomputed.loc[:target_dt].iloc[-1]
    except Exception:
        return None

    last_close = float(row["close"])
    ema_50 = float(row["ema_50"])
    ema_200 = float(row["ema_200"])
    rsi = float(row["rsi"])
    upper_bb = float(row["upper_bb"])
    lower_bb = float(row["lower_bb"])

    row_count = len(precomputed.loc[:target_dt])

    ema_200_conf = min(row_count / 400, 1.0)

    trend_diff = (ema_50 - ema_200) / (ema_200 + 1e-9)

    if trend_diff > 0.001:
        trend = "BULLISH"
    elif trend_diff < -0.001:
        trend = "BEARISH"
    else:
        trend = "SIDEWAYS"

    trend_strength = min(abs(trend_diff) / 0.02, 1.0) * ema_200_conf

    rsi_strength = 0.0
    if rsi > 60:
        rsi_strength = min((rsi - 60) / 40, 1.0)
    elif rsi < 40:
        rsi_strength = min((40 - rsi) / 40, 1.0)

    bb_signal = "STABLE"
    bb_strength = 0.0

    band_width = upper_bb - lower_bb + 1e-9

    if last_close > upper_bb:
        bb_signal = "OVERBOUGHT"
        bb_strength = abs(last_close - upper_bb) / band_width
    elif last_close < lower_bb:
        bb_signal = "OVERSOLD"
        bb_strength = abs(lower_bb - last_close) / band_width

    return {
        "price": last_close,
        "trend": trend,
        "trend_strength": trend_strength,
        "rsi": rsi,
        "rsi_strength": rsi_strength,
        "bb_signal": bb_signal,
        "bb_strength": bb_strength,
        "ema_50": ema_50,
        "ema_200": ema_200,
        "rows_available": row_count,
        "ema_200_confidence": ema_200_conf,
        "ema_200_reliable": row_count >= 400,
        "data_stale": False
    }