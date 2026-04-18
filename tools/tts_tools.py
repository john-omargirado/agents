import pandas as pd
import numpy as np

def calculate_technical_indicators(full_df: pd.DataFrame, target_date: str):
    target_dt = pd.to_datetime(target_date)
    df = full_df[full_df['timestamp'] <= target_dt].copy()

    if df.empty or len(df) < 50:
        return None

    for col in ['open', 'high', 'low', 'close']:
        df[col] = pd.to_numeric(df[col])

    last_close = float(df['close'].iloc[-1])

    # --- EMA TREND STRENGTH ---
    ema_50 = df['close'].ewm(span=50, adjust=False).mean().iloc[-1]
    ema_200 = df['close'].ewm(span=200, adjust=False).mean().iloc[-1]

    trend_diff = (ema_50 - ema_200) / ema_200

    if trend_diff > 0.001:
        trend = "BULLISH"
    elif trend_diff < -0.001:
        trend = "BEARISH"
    else:
        trend = "SIDEWAYS"

    trend_strength = min(abs(trend_diff) * 1000, 1.0)

    # --- RSI (keep numeric only) ---
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()

    rs = gain.iloc[-1] / (loss.iloc[-1] + 1e-9)
    rsi = 100 - (100 / (1 + rs))

    rsi_strength = 0.0
    if rsi > 70:
        rsi_strength = (rsi - 70) / 30
    elif rsi < 30:
        rsi_strength = (30 - rsi) / 30

    rsi_strength = min(max(rsi_strength, 0.0), 1.0)

    # --- Bollinger Bands ---
    sma_20 = df['close'].rolling(window=20).mean()
    std_20 = df['close'].rolling(window=20).std()

    upper_bb = (sma_20 + (2 * std_20)).iloc[-1]
    lower_bb = (sma_20 - (2 * std_20)).iloc[-1]

    bb_signal = "STABLE"
    bb_strength = 0.0

    if last_close >= upper_bb:
        bb_signal = "OVERBOUGHT"
        bb_strength = abs(last_close - upper_bb) / last_close

    elif last_close <= lower_bb:
        bb_signal = "OVERSOLD"
        bb_strength = abs(lower_bb - last_close) / last_close

    bb_strength = min(bb_strength * 100, 1.0)

    return {
        "price": round(last_close, 5),

        # structured trend
        "trend": trend,
        "trend_strength": round(trend_strength, 4),

        # RSI numeric + strength
        "rsi": round(rsi, 2),
        "rsi_strength": round(rsi_strength, 4),

        # BB structured
        "bb_signal": bb_signal,
        "bb_strength": round(bb_strength, 4),

        # optional diagnostics
        "ema_50": round(ema_50, 5),
        "ema_200": round(ema_200, 5)
    }